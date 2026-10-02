#!/usr/bin/env python3
"""GRPO VideoXum su TRL (trl.GRPOTrainer) per Qwen2.5-VL/Qwen3-VL con input video.

Compito, dati, prompt, reward e codifica video sono quelli di train_grpo.py, che
resta come implementazione di riferimento. Qui l'algoritmo (advantage di gruppo,
KL k3 verso il modello con l'adapter disattivato, loss, accumulo, checkpoint e
resume) e' quello di GRPOTrainer: la sottoclasse aggiunge solo l'input video, che
TRL 1.13 non gestisce (costruisce gli input multimodali soltanto dalle immagini).
"""
from __future__ import annotations
import argparse, math, os, sys, traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
import trl
from datasets import Dataset
from peft import LoraConfig, PeftModel
from tqdm.auto import tqdm
from transformers import AutoProcessor, TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from trl import GRPOConfig, GRPOTrainer

import train_grpo
from grpo_rewards import DEFAULT_REWARD_WEIGHTS, SentenceTransformerEncoder, VideoSummaryReward
from sft_train_videoxum import resolve_video_time_mode, write_video_encoding
from train_grpo import (SUMMARY_WORD_RANGE, decode_frames, ensure_reward_cache, evaluate,
                        load_datasets, prompt_inputs, read_grpo_init, subset_indices,
                        write_grpo_init, write_metric)
from vl_model_loader import load_qwen_vl_model


# Gli override usano metodi e attributi privati di GRPOTrainer (_tokenize_prompts,
# _step, _buffered_inputs, ...): vanno riverificati prima di cambiare versione.
TRL_VERSION = "1.13.0"
LOSS_TYPES = ("grpo", "dapo", "dr_grpo", "bnpo")
SCALE_REWARDS = ("group", "batch", "none")
# Tensori visivi del prompt, uno per video: nel batch vanno ripetuti per ogni riga.
VIDEO_KEYS = ("pixel_values_videos", "video_grid_thw", "second_per_grid_ts")
# Componenti della reward mediate nei log di TRL, accanto a rewards/video_summary.
LOGGED_COMPONENTS = ("total_reward_default", "valid", "format_reward", "f_beta_reward",
                     "representativeness_reward", "grounding_reward", "reference_reward",
                     "summary_length_reward", "length_budget_reward", "saliency_rank_reward",
                     "summary_words", "selected_ratio")


def write_sidecars(checkpoint, args):
    """File che infer_grpo_videoxum.py legge accanto all'adapter."""
    write_video_encoding(checkpoint, args); write_grpo_init(checkpoint, args.init_from)


def save_named(trainer, checkpoint, args):
    """Checkpoint con i nomi di train_grpo.py (epoch, best, final): adapter,
    processor e sidecar, senza lo stato dell'optimizer."""
    trainer.save_model(str(checkpoint)); write_sidecars(checkpoint, args)


class _VideoForward:
    """Aggiunge i tensori del video corrente a ogni forward di GRPOTrainer.

    TRL passa al modello solo input_ids, attention_mask e i campi delle immagini:
    senza questo wrapper le logps di policy e reference sarebbero calcolate su
    un prompt con i token video ma senza pixel."""

    def __init__(self, model, video, token_types):
        self.model, self.video, self.token_types = model, video, token_types

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, input_ids, **kwargs):
        rows = input_ids.size(0)
        extra = {key: torch.cat([value] * rows) for key, value in self.video.items()}
        if self.token_types:
            # Stessi valori di processor.create_mm_token_type_ids (immagine 1, video
            # 2), ricostruiti sull'intera sequenza prompt+completion: servono a
            # Qwen3-VL per le posizioni M-RoPE.
            types = torch.zeros_like(input_ids)
            for token_id, value in self.token_types.items(): types[input_ids == token_id] = value
            extra["mm_token_type_ids"] = types
        return self.model(input_ids=input_ids, **kwargs, **extra)


class VideoGRPOTrainer(GRPOTrainer):
    """GRPOTrainer con prompt video, un video (i suoi G rollout) per batch di generazione."""

    def __init__(self, *args, video_dataset, cli_args, **kwargs):
        super().__init__(*args, **kwargs)
        self.video_dataset, self.cli_args = video_dataset, cli_args
        self._batch_indices = self._video = self._token_types = None

    def _generate_and_score_completions(self, inputs):
        # _tokenize_prompts riceve solo i prompt: l'indice serve a ritrovare il video.
        self._batch_indices = [row["sample_index"] for row in inputs]
        return super()._generate_and_score_completions(inputs)

    def _tokenize_prompts(self, prompts):
        if len(set(self._batch_indices)) != 1:
            raise ValueError("ogni batch di generazione deve contenere un solo video: servono "
                             "per_device_train_batch_size=1 e steps_per_generation=num_generations")
        sample = self.video_dataset[self._batch_indices[0]]
        # Stessa decodifica e stesso preprocessing di train_grpo.py (frame a 1 fps,
        # marker temporali, max/min_pixels), eseguiti una volta per gruppo.
        frames = decode_frames(sample, self.cli_args.max_frames)
        inputs = prompt_inputs(self.processing_class, sample, self.cli_args, torch.device("cpu"), frames)
        device = self.accelerator.device
        self._video = {key: inputs[key].to(device) for key in VIDEO_KEYS if key in inputs}
        self._token_types = None
        if "mm_token_type_ids" in inputs:
            self._token_types = {token_id: value for token_id, value in
                                 ((self._image_pad_token_id, 1), (self._video_pad_token_id, 2))
                                 if token_id is not None}
        rows = len(prompts)
        prompt_ids = [inputs["input_ids"][0].tolist() for _ in range(rows)]
        fields = {key: torch.cat([value] * rows) for key, value in self._video.items()}
        if "mm_token_type_ids" in inputs:
            fields["mm_token_type_ids"] = inputs["mm_token_type_ids"].repeat(rows, 1).to(device)
        return prompt_ids, None, fields

    def _restore_gradient_checkpointing(self):
        # _unwrap_model_for_generation di TRL disattiva il checkpointing durante la
        # generazione ma non lo riattiva se questa solleva un'eccezione (manca un
        # try/finally attorno allo yield): senza ripristino il resto del training
        # girerebbe senza checkpointing.
        model = self.accelerator.unwrap_model(self.model)
        if self.args.gradient_checkpointing and not model.is_gradient_checkpointing:
            model.gradient_checkpointing_enable(self.args.gradient_checkpointing_kwargs)

    def _generate_single_turn(self, prompt_ids, images, multimodal_fields, has_tool_images=False):
        try:
            return super()._generate_single_turn(prompt_ids, images, multimodal_fields, has_tool_images)
        except torch.OutOfMemoryError:
            if len(prompt_ids) == 1: raise
        # Come train_grpo.generate: in OOM i rollout del gruppo uno alla volta.
        print("[warn] rollout batch in OOM; uso generazioni singole")
        self._restore_gradient_checkpointing()
        torch.cuda.empty_cache()
        single = dict(self._video)
        if "mm_token_type_ids" in multimodal_fields:
            single["mm_token_type_ids"] = multimodal_fields["mm_token_type_ids"][:1]
        completion_ids = []
        for ids in prompt_ids:
            completion, _ = super()._generate_single_turn([ids], images, single, has_tool_images)
            completion_ids.extend(completion); torch.cuda.empty_cache()
        return completion_ids, None

    def _get_per_token_logps_and_entropies(self, model, *args, **kwargs):
        return super()._get_per_token_logps_and_entropies(
            _VideoForward(model, self._video, self._token_types), *args, **kwargs)

    def training_step(self, model, inputs, num_items_in_batch):
        try:
            return super().training_step(model, inputs, num_items_in_batch)
        except Exception as error:
            # Un rank che salta il backward bloccherebbe gli altri nell'all-reduce.
            if self.accelerator.num_processes > 1: raise
            print(f"[warn] skip micro-step {self._step}: {type(error).__name__}: {error}")
            traceback.print_exc()
        # Come train_grpo.train: si scartano i gradienti della finestra di accumulo,
        # che potrebbero contenere un backward parziale. Poi TRL deve rigenerare
        # (_buffered_inputs=None) e avanzare il proprio contatore, altrimenti
        # riproverebbe per sempre la stessa slice del buffer.
        model.zero_grad(set_to_none=True)
        self._buffered_inputs = None
        self._step += 1
        self._restore_gradient_checkpointing()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return torch.zeros((), device=self.accelerator.device)

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        if self.args.should_save:
            checkpoint = Path(self._get_output_dir(trial=trial)) / f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            write_sidecars(checkpoint, self.cli_args)


def make_reward_function(reward, dataset, processor, log, run_id):
    def video_summary(completion_ids, sample_index, log_metric, trainer_state, **kwargs):
        """Reward VideoXum di train_grpo.py, un valore per rollout."""
        # Stessa decodifica di train_grpo.generate, dai token invece che dal testo di TRL.
        texts = [text.strip() for text in processor.batch_decode(
            completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)]
        samples = {index: dataset[index] for index in dict.fromkeys(sample_index)}
        rows = []
        for text, index in zip(texts, sample_index):
            ensure_reward_cache(reward, samples[index])
            rows.append(reward.score(text, samples[index]["cache_key"]))
        for name in LOGGED_COMPONENTS:
            log_metric(f"reward/{name}", float(np.mean([row[name] for row in rows])))
        memory = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else None
        for index, sample in samples.items():
            group = [i for i, value in enumerate(sample_index) if value == index]
            write_metric(log, run_id=run_id, record_type="sample", split="train",
                epoch=math.floor(trainer_state.epoch or 0) + 1, global_step=trainer_state.global_step,
                video_id=sample["video_id"], count=sample["count"],
                completions=[texts[i] for i in group], reward_components=[rows[i] for i in group],
                total_reward=float(np.mean([rows[i]["total_reward"] for i in group])),
                max_memory_allocated_gb=memory)
        return [row["total_reward"] for row in rows]
    return video_summary


class JsonlMetrics(TrainerCallback):
    """Log di TRL (loss, kl, reward, entropy, ...) nel JSONL delle metriche."""

    def __init__(self, log, run_id): self.log, self.run_id = log, run_id

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # logs contiene gia' "epoch": le chiavi vanno unite, non ripetute.
            write_metric(self.log, run_id=self.run_id, record_type="train_log",
                         **{"global_step": state.global_step, "epoch": state.epoch, **logs})


class EpochValidation(TrainerCallback):
    """Validation di train_grpo.py a fine epoca, con gli stessi checkpoint."""

    def __init__(self, trainer, val_set, reward, args, log, run_id, save_dir):
        self.trainer, self.val_set, self.reward, self.args = trainer, val_set, reward, args
        self.log, self.run_id, self.save_dir, self.best = log, run_id, save_dir, -float("inf")

    def on_epoch_end(self, args, state, control, **kwargs):
        trainer, epoch = self.trainer, round(state.epoch)
        # Salva prima della validation: se SLURM termina il job durante evaluate()
        # resta comunque il modello completo dell'epoca.
        save_named(trainer, self.save_dir / f"checkpoint-epoch{epoch}-preval", self.args)
        metrics = evaluate(trainer.model, trainer.processing_class, self.val_set, self.reward,
                           self.args, trainer.accelerator.device, self.log, self.run_id, epoch)
        trainer.model.train()
        save_named(trainer, self.save_dir / f"checkpoint-epoch{epoch}", self.args)
        # Reward con pesi di default: il best resta confrontabile tra ablation.
        if metrics.get("total_reward_default", -1) > self.best:
            self.best = metrics["total_reward_default"]
            save_named(trainer, self.save_dir / "checkpoint-best", self.args)


def build_reward(args):
    """Stessa reward di train_grpo.train."""
    reward = VideoSummaryReward(
        SentenceTransformerEncoder(args.semantic_model, args.semantic_device),
        f_beta=args.reward_f_beta,
        f_beta_aggregation=args.f_beta_aggregation,
        weights={name: getattr(args, f"w_{name}") for name in DEFAULT_REWARD_WEIGHTS},
        use_length_penalty=args.length_penalty,
        summary_length_sigma=args.summary_length_sigma,
        summary_target_words=args.summary_target_words,
        length_budget_sigma=args.length_budget_sigma,
        reference_cosine_floor=args.reference_cosine_floor,
        reference_cosine_scale=args.reference_cosine_scale,
        timeline_unit=args.timeline_unit,
    )
    print(f"[reward] pesi={reward.weights} length_penalty={reward.use_length_penalty} "
          f"f_beta={args.reward_f_beta} f_beta_aggregation={reward.f_beta_aggregation} "
          f"summary_style={args.summary_style} summary_word_range={SUMMARY_WORD_RANGE} "
          f"summary_length_sigma={reward.summary_length_sigma} "
          f"summary_target_words={reward.summary_target_words or 'per-video'} "
          f"length_budget_sigma={reward.length_budget_sigma} "
          f"reference_cosine=clamp((cos-{reward.reference_cosine_floor})/{reward.reference_cosine_scale}) "
          f"timeline_unit={reward.timeline_unit}")
    return reward


def cache_train_rewards(reward, dataset, indices):
    """Prepara la cache della reward ed esclude i video con feature incompatibili.

    train_grpo.py li saltava a runtime; qui l'esclusione avviene prima del
    training, cosi' TRL non genera rollout per video che non si possono valutare."""
    valid = []
    for index in tqdm(indices, desc="reward cache"):
        sample = dataset[index]
        try:
            ensure_reward_cache(reward, sample)
        except Exception as error:
            print(f"[warn] escluso {sample['video_id']}: {type(error).__name__}: {error}")
            continue
        valid.append(index)
    print(f"[data] video train con reward valutabile: {len(valid)}/{len(indices)}")
    return valid


def train(args):
    if trl.__version__ != TRL_VERSION:
        raise SystemExit(f"[trl] richiesto trl=={TRL_VERSION}, installato {trl.__version__}")
    if not args.model.is_dir(): raise FileNotFoundError(f"modello base non trovato: {args.model}")
    if args.init_from is not None and not args.init_from.is_dir():
        raise FileNotFoundError(f"checkpoint PEFT iniziale non trovato: {args.init_from}")
    # Il Trainer usa la GPU visibile al processo: la scelta passa da CUDA_VISIBLE_DEVICES.
    if args.device not in ("cuda", "cuda:0"): raise SystemExit("--device deve essere cuda:0")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA non disponibile")
    if args.resume_from is not None:
        if not (args.resume_from / "trainer_state.json").is_file():
            raise SystemExit(f"[resume] {args.resume_from} non e' un checkpoint-N di questo script "
                             "(manca trainer_state.json)")
        # Il LoRA da riprendere e' un delta sul modello con l'adapter iniziale fuso.
        found, saved_init = read_grpo_init(args.resume_from)
        if found and args.init_from is None:
            args.init_from = saved_init
        elif found and saved_init is not None and Path(args.init_from) != saved_init:
            raise SystemExit(f"[model] {args.resume_from} parte da {saved_init}, non da {args.init_from}")
    # Il resume eredita l'asse temporale del proprio checkpoint, il warm-start
    # quello del checkpoint SFT/DPO di partenza.
    resolve_video_time_mode(args, args.resume_from or args.init_from, default="seconds")
    train_set, val_set = load_datasets(args)
    reward = build_reward(args)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    save_dir, result_dir = args.saveroot/args.model.name, args.resultsroot/args.model.name
    save_dir.mkdir(parents=True, exist_ok=True); result_dir.mkdir(parents=True, exist_ok=True)
    log = (result_dir/f"metrics-{run_id}.jsonl").open("w", encoding="utf-8", buffering=1)
    train_indices = cache_train_rewards(
        reward, train_set, subset_indices(len(train_set), args.train_samples or args.debug_samples, args.seed))
    print(f"[data] training={len(train_indices)}/{len(train_set)}, validation="
          f"{min(args.val_samples, len(val_set)) if args.val_samples else len(val_set)}/{len(val_set)}")

    model, model_type = load_qwen_vl_model(args.model, torch_dtype=torch.bfloat16,
                                           attn_implementation=args.attn_impl)
    model = model.to(args.device)
    print(f"[model] architecture={model_type}")
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    if args.init_from is not None:
        model = PeftModel.from_pretrained(
            model, args.init_from, is_trainable=False, local_files_only=True
        ).merge_and_unload()
        print(f"[model] warm-start e reference KL dal checkpoint PEFT: {args.init_from}")
    else:
        print("[model] nessun checkpoint PEFT: warm-start e reference KL dal modello base")

    config = GRPOConfig(
        output_dir=str(save_dir),
        # Un micro-step per rollout e un batch di generazione per video: con
        # accum_steps video per update, come in train_grpo.py.
        per_device_train_batch_size=1,
        num_generations=args.num_generations,
        steps_per_generation=args.num_generations,
        gradient_accumulation_steps=args.num_generations*args.accum_steps,
        num_iterations=1,
        num_train_epochs=args.epochs,
        beta=args.beta, loss_type=args.loss_type, scale_rewards=args.scale_rewards,
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        max_completion_length=args.max_completion_length,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=0.1,
        weight_decay=0.01, max_grad_norm=1.0, optim="adamw_torch",
        bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="steps", save_steps=args.save_steps, logging_steps=1,
        eval_strategy="no", report_to="none", remove_unused_columns=False, seed=args.seed,
    )
    write_metric(log, run_id=run_id, record_type="config", arguments=vars(args),
                 trl_version=trl.__version__, grpo_config=config.to_dict())
    dataset = Dataset.from_dict({"prompt": [train_set[i]["prompt"] for i in train_indices],
                                 "sample_index": train_indices})
    trainer = VideoGRPOTrainer(
        model=model,
        reward_funcs=make_reward_function(reward, train_set, processor, log, run_id),
        args=config, train_dataset=dataset, processing_class=processor,
        peft_config=LoraConfig(r=args.lora_r, lora_alpha=2*args.lora_r, lora_dropout=0,
                               target_modules=["q_proj", "v_proj"], bias="none", task_type="CAUSAL_LM"),
        callbacks=[JsonlMetrics(log, run_id)],
        video_dataset=train_set, cli_args=args,
    )
    trainer.add_callback(EpochValidation(trainer, val_set, reward, args, log, run_id, save_dir))
    trainer.model.print_trainable_parameters()
    trainer.train(resume_from_checkpoint=str(args.resume_from) if args.resume_from else None)
    save_named(trainer, save_dir/"final", args); log.close(); print(f"[done] {save_dir/'final'}")


def parse_args():
    """Flag propri della versione TRL, poi la CLI di train_grpo.py sul resto."""
    parser = argparse.ArgumentParser(add_help=False, description="Opzioni TRL di train_grpo_trl.py "
                                     "(le altre sono quelle di train_grpo.py)")
    parser.add_argument("--loss_type", choices=LOSS_TYPES, default="grpo",
        help="aggregazione della loss di TRL; grpo (media per sequenza) e' l'obiettivo di "
             "train_grpo.py, dapo e dr_grpo eliminano il bias sulla lunghezza")
    parser.add_argument("--scale_rewards", choices=SCALE_REWARDS, default="group",
        help="normalizzazione dell'advantage: group come train_grpo.py, none come Dr. GRPO")
    parser.add_argument("--top_k", type=int, default=20,
        help="top-k del campionamento; 20 e' il valore del generation_config di Qwen3-VL, "
             "che train_grpo.py eredita implicitamente (TRL altrimenti usa 0, nessun filtro)")
    if {"-h", "--help"} & set(sys.argv[1:]): parser.print_help(); print()
    trl_args, rest = parser.parse_known_args()
    if trl_args.top_k < 0: parser.error("--top_k deve essere >= 0")
    sys.argv = [sys.argv[0], *rest]
    args = train_grpo.parse_args()
    vars(args).update(vars(trl_args))
    return args


if __name__ == "__main__": train(parse_args())

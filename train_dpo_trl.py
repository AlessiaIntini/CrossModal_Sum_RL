#!/usr/bin/env python3
"""DPO VideoXum su TRL (trl.DPOTrainer) per Qwen2.5-VL/Qwen3-VL con input video.

Dati, prompt, coppie chosen/rejected, codifica video e validation generativa sono
quelli di train_dpo.py, che resta come implementazione di riferimento. Qui loss DPO,
reference, accumulo, scheduler, checkpoint e resume sono quelli di DPOTrainer: il
collator aggiunge l'input video, che TRL 1.13 non gestisce (prepara solo immagini).

Differenze note rispetto a train_dpo.py:
- reference: con --init_from DPOTrainer copia l'adapter SFT in un adapter "ref"
  congelato invece di caricare una seconda copia del modello (stessa reference,
  ~17 GB di VRAM in meno); senza --init_from e' il modello con l'adapter disattivato.
- chosen e rejected passano in un solo forward (batch di due righe).
- dropout disattivato, LoRA dropout compreso (disable_dropout di TRL): il forward
  della reference con l'adapter "ref" avviene in modalita' train e il dropout la
  renderebbe stocastica.
- ordine dei video deciso dal sampler del Trainer (stesso seed, stesso sottoinsieme).
"""
from __future__ import annotations
import argparse, json, math, os, random, shutil, sys, time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import trl
from datasets import Dataset
from peft import LoraConfig, PeftModel
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoProcessor
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from trl import DPOConfig, DPOTrainer

import train_dpo
from sft_train_videoxum import (RESUME_POINTER_NAME, TIME_BUDGET_EXIT_CODE, guard_checkpoint_dir,
                                resolve_best_metric, resolve_video_time_mode, write_video_encoding)
from sft_train_videoxum_trl import SFTSchedule
from train_dpo import decode_video_frames, evaluate, load_datasets, prepare_completion, write_metric
from vl_model_loader import load_qwen_vl_model


# Gli override usano metodi privati di DPOTrainer (_prepare_dataset, _save, _metrics):
# vanno riverificati prima di cambiare versione (vedi train_grpo_trl.py).
TRL_VERSION = "1.13.0"
# Sidecar con l'adapter SFT di partenza: il resume ne ricostruisce la reference.
DPO_INIT_NAME = "dpo_init.json"
# Campi lunghi quanto la sequenza: vanno allineati col padding, gli altri (pixel e
# griglia del video) si concatenano riga per riga.
TOKEN_KEYS = ("input_ids", "attention_mask", "mm_token_type_ids")


def write_sidecars(checkpoint, args):
    """video_encoding.json per inferenza e GRPO, dpo_init.json per il resume."""
    write_video_encoding(checkpoint, args)
    (Path(checkpoint) / DPO_INIT_NAME).write_text(
        json.dumps({"init_from": str(args.init_from) if args.init_from else None}, indent=2) + "\n",
        encoding="utf-8")


def read_dpo_init(checkpoint):
    """(trovato, init_from) registrati nel checkpoint da write_sidecars."""
    path = Path(checkpoint) / DPO_INIT_NAME
    if not path.is_file():
        return False, None
    init_from = json.loads(path.read_text(encoding="utf-8"))["init_from"]
    return True, Path(init_from) if init_from else None


def save_named(trainer, checkpoint, args):
    """Checkpoint con i nomi di train_dpo.py (best, epoch, final): adapter, processor
    e sidecar, senza lo stato dell'optimizer."""
    trainer.save_model(str(checkpoint)); write_sidecars(checkpoint, args)


def full_sequence_forward(module, args, kwargs):
    """Forward pre-hook: sequenza intera al modello, logits solo sulla completion.

    Il collator mette in input_ids e completion_mask solo la coda della sequenza,
    dall'ultimo token del prompt in poi, che e' tutto cio' su cui DPOTrainer calcola le
    log-prob; la sequenza completa, con i token video, sta in full_input_ids. Il
    modello riceve quest'ultima con logits_to_keep pari alla coda: log-prob identiche
    a un forward completo, ma senza i logits (vocabolario ~152k) sulle migliaia di
    token video, che con chosen e rejected nello stesso batch non entrerebbero in
    memoria sui video lunghi. Vale per policy e reference, che DPOTrainer calcola
    entrambe chiamando lo stesso PeftModel."""
    full_input_ids = kwargs.pop("full_input_ids", None)
    if full_input_ids is None:
        return None
    kwargs["logits_to_keep"] = kwargs["input_ids"].size(1)
    kwargs["input_ids"] = full_input_ids
    return args, kwargs


class VideoPreferenceCollator:
    """Una coppia (un video) per batch, nel formato di DataCollatorForVisionPreference:
    righe [chosen, rejected] con padding a destra e completion_mask sulla risposta.

    Stesso input di train_dpo.prepare_pair; lo spostamento sul device lo fa il Trainer."""

    def __init__(self, dataset, processor, args, model_type):
        self.dataset, self.processor, self.args, self.model_type = dataset, processor, args, model_type

    def __call__(self, rows):
        if len(rows) != 1:
            raise ValueError("serve per_device_train_batch_size=1: i video non vengono impaccati")
        index = rows[0]["sample_index"]
        sample = self.dataset[index]
        frames = decode_video_frames(sample, self.args)
        (chosen, prompt_len), (rejected, rejected_prompt_len) = (
            prepare_completion(self.processor, sample, sample[key], self.args, self.model_type, frames)
            for key in ("chosen", "rejected"))
        if rejected_prompt_len != prompt_len or not torch.equal(
                chosen["input_ids"][0, :prompt_len], rejected["input_ids"][0, :prompt_len]):
            raise RuntimeError(f"sample {index}: chosen e rejected hanno prompt diversi")
        batch = {}
        for key, value in chosen.items():
            if key in TOKEN_KEYS:
                pad = self.processor.tokenizer.pad_token_id if key == "input_ids" else 0
                batch[key] = pad_sequence([value[0], rejected[key][0]], batch_first=True, padding_value=pad)
            else:
                batch[key] = torch.cat([value, rejected[key]])
        completion_mask = batch["attention_mask"].clone()
        completion_mask[:, :prompt_len] = 0
        # Coda su cui DPOTrainer calcola le log-prob: parte dall'ultimo token del
        # prompt, i cui logits predicono il primo token della risposta.
        tail = slice(prompt_len - 1, None)
        batch["full_input_ids"] = batch["input_ids"]
        batch["input_ids"] = batch["input_ids"][:, tail]
        batch["completion_mask"] = completion_mask[:, tail]
        batch["sample_index"] = torch.tensor([index])
        return batch


class VideoDPOTrainer(DPOTrainer):
    """DPOTrainer con le coppie video preparate da VideoPreferenceCollator."""

    def __init__(self, *args, cli_args, metrics_file, run_id, **kwargs):
        super().__init__(*args, **kwargs)
        self.cli_args, self.metrics_file, self.run_id = cli_args, metrics_file, run_id
        self._sample_index = None
        self.model.register_forward_pre_hook(full_sequence_forward, with_kwargs=True)

    def _prepare_dataset(self, dataset, *args, **kwargs):
        # Il dataset contiene solo sample_index: prompt, risposte e frame li tokenizza
        # il collator, come per i dataset di immagini di TRL.
        return dataset

    def _epoch(self):
        return math.floor(self.state.epoch or 0) + 1

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # sample_index serve solo ai log: il modello non lo accetta.
        self._sample_index = int(inputs.pop("sample_index")[0])
        chosen_tokens, rejected_tokens = inputs["completion_mask"].sum(dim=1).tolist()
        result = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        if self.model.training:
            # Valori appena registrati da DPOTrainer per questa coppia. Le log-prob
            # sono somme sui token anche con sigmoid_norm, che normalizza solo la loss.
            last = {key: values[-1] for key, values in self._metrics["train"].items() if values}
            policy_margin = last["logps/chosen"] - last["logps/rejected"]
            write_metric(self.metrics_file, run_id=self.run_id, record_type="sample", split="train",
                         epoch=self._epoch(), dataset_index=self._sample_index,
                         optimizer_step=self.state.global_step + 1,
                         chosen_completion_tokens=chosen_tokens, rejected_completion_tokens=rejected_tokens,
                         chosen_logps=last["logps/chosen"], rejected_logps=last["logps/rejected"],
                         policy_margin=policy_margin,
                         reference_margin=policy_margin - last["rewards/margins"] / self.beta,
                         reward_margin=last["rewards/margins"],
                         dpo_loss=(result[0] if return_outputs else result).detach().float().item(),
                         learning_rate=self._get_learning_rate())
        return result

    def training_step(self, model, inputs, num_items_in_batch=None):
        try:
            return super().training_step(model, inputs, num_items_in_batch)
        except torch.OutOfMemoryError as error:
            # Un rank che salta il backward bloccherebbe gli altri nell'all-reduce.
            if self.accelerator.num_processes > 1: raise
            message = str(error)
        # Come train_dpo.py: un video troppo lungo per la VRAM viene saltato e si
        # scartano i gradienti della finestra, che potrebbero essere parziali.
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        write_metric(self.metrics_file, run_id=self.run_id, record_type="sample", split="train",
                     epoch=self._epoch(), dataset_index=self._sample_index,
                     optimizer_step=self.state.global_step + 1, skipped_oom=True, error=message)
        print(f"[WARN] OOM sul sample {self._sample_index}: saltato.", file=sys.stderr)
        return torch.zeros((), device=self.accelerator.device)

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir, state_dict)
        # PEFT salverebbe anche l'adapter "ref" in ref/, ma al resume il Trainer, se
        # trova sottocartelle di adapter, carica solo quelle e non l'adapter addestrato
        # nella radice. La reference si ricostruisce comunque da dpo_init.json.
        shutil.rmtree(Path(output_dir or self.args.output_dir) / "ref", ignore_errors=True)

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        if self.args.should_save:
            checkpoint = Path(self._get_output_dir(trial=trial)) / f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            write_sidecars(checkpoint, self.cli_args)


class DPOSchedule(SFTSchedule):
    """SFTSchedule (validation, checkpoint-best/epoch, time budget) con la validation
    di train_dpo.py e il checkpoint di fine epoca salvato prima di validare."""

    def _validate(self, state, epoch, limit=None):
        trainer = self.trainer
        _, _, metrics = evaluate(trainer.model, trainer.processing_class, self.val_dataset, self.args,
                                 trainer.accelerator.device, self.metrics_file, self.run_id, epoch,
                                 self.model_type, limit=limit, optimizer_step=state.global_step)
        trainer.model.train()
        # Il best sta nel TrainerState: finisce nei checkpoint-N e torna con il resume.
        if state.best_metric is None or metrics["selection_score"] > state.best_metric:
            state.best_metric = metrics["selection_score"]
            save_named(trainer, self.save_dir / "checkpoint-best", self.args)
            print(f"[best] checkpoint-best aggiornato: {self.args.best_metric}={state.best_metric:.4f}")
        return metrics

    def _end_of_epoch(self, state, epoch):
        # Se SLURM termina il job durante la validation resta l'adapter dell'epoca.
        save_named(self.trainer, self.save_dir / f"checkpoint-epoch{epoch}-preval", self.args)
        metrics = self._validate(state, epoch)
        save_named(self.trainer, self.save_dir / f"checkpoint-epoch{epoch}", self.args)
        print(f"[epoch {epoch}] F1_MEAN={metrics['f1_mean']:.4f}")


def build_config(args, save_dir, **overrides):
    """DPOConfig con gli iperparametri di train_dpo.py."""
    return DPOConfig(**{
        "output_dir": str(save_dir),
        # Una coppia per micro-step, accum_steps coppie per update, come train_dpo.py.
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": args.accum_steps,
        "num_train_epochs": args.epochs,
        "learning_rate": args.lr, "lr_scheduler_type": "cosine", "warmup_steps": args.warmup_ratio,
        "weight_decay": args.weight_decay, "max_grad_norm": args.max_grad_norm, "optim": "adamw_torch",
        "bf16": True, "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "save_strategy": "steps", "save_steps": args.save_steps, "logging_steps": 1,
        "eval_strategy": "no", "report_to": "none",
        "beta": args.beta,
        # sigmoid: DPO standard sulle log-prob sommate; sigmoid_norm: media per token,
        # come --length-normalize di train_dpo.py.
        "loss_type": ["sigmoid_norm" if args.length_normalize else "sigmoid"],
        "disable_dropout": True,
        # Il collator costruisce input e maschere: TRL non deve troncare (max_length
        # taglierebbe i token video) ne' togliere la colonna sample_index.
        "max_length": None, "remove_unused_columns": False,
        "dataloader_num_workers": args.dataloader_num_workers, "seed": args.seed,
        **overrides,
    })


def load_policy(args):
    """Modello base, eventualmente con l'adapter SFT da cui partono policy e reference."""
    print(f"[model] caricamento offline di {args.model}")
    model, model_type = load_qwen_vl_model(args.model, torch_dtype=torch.bfloat16,
                                           attn_implementation=args.attn_impl)
    model = model.to(args.device)
    print(f"[model] architecture={model_type}")
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    if args.init_from is None:
        lora_targets = [name.strip() for name in args.lora_target_modules.split(",") if name.strip()]
        print(f"[lora] target_modules={lora_targets}; reference = modello base")
        return model, model_type, processor, LoraConfig(
            r=args.lora_r, lora_alpha=2*args.lora_r, lora_dropout=args.lora_dropout,
            target_modules=lora_targets, bias="none", task_type="CAUSAL_LM")
    # DPOTrainer copia questo adapter in "ref", congelato, e addestra l'originale.
    print(f"[model] policy e reference dall'adapter SFT {args.init_from}")
    model = PeftModel.from_pretrained(model, args.init_from, is_trainable=True, local_files_only=True)
    return model, model_type, processor, None


def train(args):
    if trl.__version__ != TRL_VERSION:
        raise SystemExit(f"[trl] richiesto trl=={TRL_VERSION}, installato {trl.__version__}")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if not args.model.is_dir():
        raise FileNotFoundError(f"Checkpoint locale non trovato: {args.model}")
    # Il Trainer usa la GPU visibile al processo: la scelta passa da CUDA_VISIBLE_DEVICES.
    if args.device not in ("cuda", "cuda:0"): raise SystemExit("--device deve essere cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA non disponibile: eseguire su un nodo GPU")
    if args.resume_from is not None:
        if not (args.resume_from / "trainer_state.json").is_file():
            raise SystemExit(f"[resume] {args.resume_from} non e' un checkpoint-N di questo script "
                             "(manca trainer_state.json)")
        # Il checkpoint contiene solo l'adapter addestrato: la reference va ricostruita
        # dallo stesso adapter SFT di partenza.
        found, args.init_from = read_dpo_init(args.resume_from)
        if not found:
            raise SystemExit(f"[resume] {args.resume_from} non contiene {DPO_INIT_NAME}")
    start_time = time.monotonic()
    # La policy eredita l'asse temporale da cui parte: dal resume o dall'SFT di init.
    resolve_video_time_mode(args, args.resume_from or args.init_from, default="seconds")
    resolve_best_metric(args)
    save_dir = args.saveroot / args.model.name
    results_dir = args.resultsroot / args.model.name
    if not args.resume_from:
        guard_checkpoint_dir(save_dir, args)
    save_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    metrics_path = results_dir / f"metrics-{run_id}.jsonl"
    metrics_file = metrics_path.open("w", encoding="utf-8", buffering=1)
    print(f"[metrics] log JSONL: {metrics_path}")
    train_dataset, val_dataset = load_datasets(args)
    # Stesso sottoinsieme che train_dpo.py usa nella prima epoca; l'ordine dei video
    # lo decide poi il sampler del Trainer, con lo stesso seed.
    indices = list(range(len(train_dataset)))
    random.Random(args.seed).shuffle(indices)
    n_samples = min(args.debug_samples, len(train_dataset)) if args.debug_samples else len(train_dataset)
    indices = indices[:n_samples]
    steps_per_epoch = math.ceil(n_samples / args.accum_steps)
    print(f"[train] samples={n_samples}, steps={steps_per_epoch * args.epochs}")

    model, model_type, processor, peft_config = load_policy(args)
    config = build_config(args, save_dir)
    write_metric(metrics_file, run_id=run_id, record_type="config", arguments=vars(args),
                 trl_version=trl.__version__, dpo_config=config.to_dict())
    trainer = VideoDPOTrainer(
        model=model, args=config,
        data_collator=VideoPreferenceCollator(train_dataset, processor, args, model_type),
        train_dataset=Dataset.from_dict({"sample_index": indices}),
        processing_class=processor, peft_config=peft_config,
        cli_args=args, metrics_file=metrics_file, run_id=run_id,
    )
    schedule = DPOSchedule(trainer, val_dataset, args, metrics_file, run_id, save_dir, model_type,
                           steps_per_epoch, start_time)
    trainer.add_callback(schedule)
    print(f"[dpo] loss_type={config.loss_type} beta={config.beta}")
    trainer.model.print_trainable_parameters()
    trainer.train(resume_from_checkpoint=str(args.resume_from) if args.resume_from else None)

    if schedule.out_of_time:
        # Il Trainer ha appena salvato checkpoint-N: lo script SLURM riparte da li'.
        checkpoint = save_dir / f"{PREFIX_CHECKPOINT_DIR}-{trainer.state.global_step}"
        (save_dir / RESUME_POINTER_NAME).write_text(f"{checkpoint}\n", encoding="utf-8")
        write_metric(metrics_file, run_id=run_id, record_type="time_budget_exit",
                     elapsed_hours=schedule.elapsed_hours(), optimizer_step=trainer.state.global_step,
                     epoch=trainer.state.epoch, resume_from=checkpoint)
        metrics_file.close()
        print(f"[time budget] {schedule.elapsed_hours():.2f}h >= {args.time_budget_hours}h: "
              f"riprendere da {checkpoint}")
        sys.exit(TIME_BUDGET_EXIT_CODE)

    final_path = save_dir / "final"
    save_named(trainer, final_path, args)
    metrics_file.close()
    print(f"[done] adapter finale salvato in {final_path}")
    print(f"[done] metriche salvate in {metrics_path}")


def parse_args():
    """Flag propri della versione TRL, poi la CLI di train_dpo.py sul resto."""
    parser = argparse.ArgumentParser(add_help=False, description="Opzioni TRL di train_dpo_trl.py "
                                     "(le altre sono quelle di train_dpo.py)")
    parser.add_argument("--dataloader_num_workers", type=int, default=2,
        help="processi che decodificano i video in parallelo al training (0: nel processo principale)")
    if {"-h", "--help"} & set(sys.argv[1:]): parser.print_help(); print()
    trl_args, rest = parser.parse_known_args()
    if trl_args.dataloader_num_workers < 0: parser.error("--dataloader_num_workers deve essere >= 0")
    sys.argv = [sys.argv[0], *rest]
    args = train_dpo.parse_args()
    vars(args).update(vars(trl_args))
    if args.eval_only or args.eval_checkpoint is not None:
        raise SystemExit("--eval_only resta in train_dpo.py, che valuta anche i checkpoint "
                         "salvati da questo script")
    if args.init_from and args.resume_from:
        raise SystemExit("--init_from e --resume_from sono mutuamente esclusivi: il resume "
                         f"rilegge l'adapter di partenza da {DPO_INIT_NAME}")
    if args.init_from and not (args.init_from / "adapter_config.json").is_file():
        raise SystemExit(f"--init_from non e' un adapter PEFT valido: {args.init_from}")
    if args.eval_steps and 0 < args.eval_samples < args.eval_steps_samples:
        raise SystemExit("--eval_samples deve essere >= --eval_steps_samples: checkpoint-best "
                         "confronta tutte le validation sui primi --eval_steps_samples video")
    return args


if __name__ == "__main__": train(parse_args())

#!/usr/bin/env python3
"""SFT VideoXum su TRL (trl.SFTTrainer) per Qwen2.5-VL/Qwen3-VL con input video.

Dati, prompt, target 'chosen', codifica video e validation generativa sono quelli
di sft_train_videoxum.py, che resta come implementazione di riferimento. Qui il
loop (loss, accumulo, scheduler, checkpoint e resume) e' quello di SFTTrainer: il
collator aggiunge l'input video, che TRL 1.13 non gestisce (prepara solo immagini).
"""
from __future__ import annotations
import argparse, math, os, random, sys, time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import trl
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoProcessor, TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from trl import SFTConfig, SFTTrainer

import sft_train_videoxum
from sft_train_videoxum import (RESUME_POINTER_NAME, TIME_BUDGET_EXIT_CODE, decode_video_frames,
                                evaluate, guard_checkpoint_dir, load_datasets,
                                measure_generation_diversity, measure_video_dependence,
                                prepare_completion, resolve_best_metric, resolve_video_time_mode,
                                write_metric, write_video_encoding)
from vl_model_loader import load_qwen_vl_model


# Versione su cui sono scritti gli override di SFTTrainer (vedi train_grpo_trl.py).
TRL_VERSION = "1.13.0"


def save_named(trainer, checkpoint, args):
    """Checkpoint con i nomi di sft_train_videoxum.py (best, epoch, final): adapter,
    processor e video_encoding.json, senza lo stato dell'optimizer."""
    trainer.save_model(str(checkpoint)); write_video_encoding(checkpoint, args)


class VideoSFTCollator:
    """Un video per batch: prompt+target tokenizzati con i frame, label -100 sul prompt.

    Stesso input di sft_train_videoxum.prepare_sft_example; lo spostamento sul
    device lo fa il Trainer."""

    def __init__(self, dataset, processor, args, model_type):
        self.dataset, self.processor, self.args, self.model_type = dataset, processor, args, model_type

    def __call__(self, rows):
        if len(rows) != 1:
            raise ValueError("serve per_device_train_batch_size=1: i video non vengono impaccati")
        index = rows[0]["sample_index"]
        sample = self.dataset[index]
        frames = decode_video_frames(sample, self.args)
        inputs, prompt_len = prepare_completion(
            self.processor, sample, sample["target"], self.args, self.model_type, frames)
        labels = inputs["input_ids"].clone()
        labels[:, :prompt_len] = -100
        return {**inputs, "labels": labels, "sample_index": torch.tensor([index])}


class VideoSFTTrainer(SFTTrainer):
    """SFTTrainer con i batch video preparati da VideoSFTCollator."""

    def __init__(self, *args, cli_args, metrics_file, run_id, **kwargs):
        super().__init__(*args, **kwargs)
        self.cli_args, self.metrics_file, self.run_id = cli_args, metrics_file, run_id
        self._sample_index = None
        if cli_args.loss_normalization == "sample":
            # Senza num_items_in_batch la loss del modello e' la media sui token del
            # video e il Trainer la divide per gli step di accumulo: ogni video pesa
            # uguale, come in sft_train_videoxum.py.
            self.model_accepts_loss_kwargs = False

    def _epoch(self):
        return math.floor(self.state.epoch or 0) + 1

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # sample_index serve solo ai log: il modello non lo accetta.
        self._sample_index = int(inputs.pop("sample_index")[0])
        tokens = int((inputs["labels"][:, 1:] != -100).sum())
        result = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        if self.model.training:
            loss = (result[0] if return_outputs else result).detach().float().item()
            if self.model_accepts_loss_kwargs and num_items_in_batch is not None:
                # Con la normalizzazione per token la loss e' somma/token della
                # finestra: la si riporta alla media sul video, come sft_loss di oggi.
                loss *= float(num_items_in_batch) / max(tokens, 1)
            write_metric(self.metrics_file, run_id=self.run_id, record_type="sample", split="train",
                         epoch=self._epoch(), dataset_index=self._sample_index,
                         optimizer_step=self.state.global_step + 1, completion_tokens=tokens,
                         sft_loss=loss, learning_rate=self._get_learning_rate())
        return result

    def training_step(self, model, inputs, num_items_in_batch=None):
        try:
            return super().training_step(model, inputs, num_items_in_batch)
        except torch.OutOfMemoryError as error:
            # Un rank che salta il backward bloccherebbe gli altri nell'all-reduce.
            if self.accelerator.num_processes > 1: raise
            message = str(error)
        # Come sft_train_videoxum.py: un video anomalo (troppi token) viene saltato
        # e si scartano i gradienti della finestra, che potrebbero essere parziali.
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        write_metric(self.metrics_file, run_id=self.run_id, record_type="sample", split="train",
                     epoch=self._epoch(), dataset_index=self._sample_index,
                     optimizer_step=self.state.global_step + 1, skipped_oom=True, error=message)
        print(f"[WARN] OOM sul sample {self._sample_index}: saltato.", file=sys.stderr)
        return torch.zeros((), device=self.accelerator.device)

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        if self.args.should_save:
            checkpoint = Path(self._get_output_dir(trial=trial)) / f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            write_video_encoding(checkpoint, self.cli_args)


class SFTSchedule(TrainerCallback):
    """Validation generativa, checkpoint-best/epoch e time budget di sft_train_videoxum.py.

    Un solo callback perche' conta l'ordine: il budget si controlla prima di ogni
    validation, cosi' un job vicino al limite SLURM salva ed esce invece di validare."""

    def __init__(self, trainer, val_dataset, args, metrics_file, run_id, save_dir, model_type,
                 steps_per_epoch, start_time):
        self.trainer, self.val_dataset, self.args = trainer, val_dataset, args
        self.metrics_file, self.run_id, self.save_dir = metrics_file, run_id, save_dir
        self.model_type, self.steps_per_epoch, self.start_time = model_type, steps_per_epoch, start_time
        self.out_of_time = False

    def elapsed_hours(self):
        return (time.monotonic() - self.start_time) / 3600

    def _check_budget(self, control):
        if self.args.time_budget_hours and self.elapsed_hours() >= self.args.time_budget_hours:
            # Il Trainer salva checkpoint-N (con optimizer e RNG) e interrompe il loop.
            self.out_of_time = True
            control.should_save = True
            control.should_training_stop = True
        return self.out_of_time

    def _validate(self, state, epoch, limit=None):
        trainer = self.trainer
        metrics = evaluate(trainer.model, trainer.processing_class, self.val_dataset, self.args,
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
        metrics = self._validate(state, epoch)
        save_named(self.trainer, self.save_dir / f"checkpoint-epoch{epoch}", self.args)
        print(f"[epoch {epoch}] F1_MEAN={metrics['f1_mean']:.4f}")

    def on_train_begin(self, args, state, control, **kwargs):
        # Resume da un checkpoint salvato a fine epoca (budget esaurito, o job
        # terminato durante la validation): quella validation manca ancora.
        epoch = state.global_step // self.steps_per_epoch
        if (state.global_step and state.global_step % self.steps_per_epoch == 0
                and not (self.save_dir / f"checkpoint-epoch{epoch}").exists()):
            self._end_of_epoch(state, epoch)

    def on_step_end(self, args, state, control, **kwargs):
        if self._check_budget(control): return
        # Validation ridotta ogni eval_steps, ma non all'ultimo step dell'epoca,
        # dove parte quella completa.
        if (self.args.eval_steps and state.global_step % self.args.eval_steps == 0
                and state.global_step % self.steps_per_epoch):
            self._validate(state, state.global_step // self.steps_per_epoch + 1,
                           limit=self.args.eval_steps_samples)
            self._check_budget(control)

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.out_of_time or self._check_budget(control): return
        self._end_of_epoch(state, round(state.epoch))

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # logs contiene gia' "epoch": le chiavi vanno unite, non ripetute.
            write_metric(self.metrics_file, run_id=self.run_id, record_type="train_log",
                         **{"global_step": state.global_step, "epoch": state.epoch, **logs})


def build_config(args, save_dir, **overrides):
    """SFTConfig con gli iperparametri di sft_train_videoxum.py."""
    return SFTConfig(**{
        "output_dir": str(save_dir),
        # Un video per micro-step, accum_steps video per update, come oggi.
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": args.accum_steps,
        "num_train_epochs": args.epochs,
        "learning_rate": args.lr, "lr_scheduler_type": "cosine", "warmup_steps": args.warmup_ratio,
        "weight_decay": args.weight_decay, "max_grad_norm": args.max_grad_norm, "optim": "adamw_torch",
        "bf16": True, "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "save_strategy": "steps", "save_steps": args.save_steps, "logging_steps": 1,
        "eval_strategy": "no", "report_to": "none",
        # Il collator costruisce input e label: TRL non deve tokenizzare ne' troncare
        # (max_length taglierebbe i token video), ne' togliere la colonna sample_index.
        "remove_unused_columns": False, "dataset_kwargs": {"skip_prepare_dataset": True},
        "max_length": None, "loss_type": "nll",
        "dataloader_num_workers": args.dataloader_num_workers, "seed": args.seed,
        **overrides,
    })


def train(args, datasets_loader=load_datasets, schedule_cls=SFTSchedule):
    """``datasets_loader`` e ``schedule_cls`` sono i due punti in cui la SFT V2VT
    (sft_train_videoxum_v2vt_trl.py) cambia: dati/prompt/target e validation. Il
    resto del loop (OOM, time budget, checkpoint, resume) resta uno solo."""
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
    if args.resume_from is not None and not (args.resume_from / "trainer_state.json").is_file():
        raise SystemExit(f"[resume] {args.resume_from} non e' un checkpoint-N di questo script "
                         "(manca trainer_state.json)")
    start_time = time.monotonic()
    resolve_video_time_mode(args, args.resume_from, default="seconds")
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
    train_dataset, val_dataset = datasets_loader(args)
    # Stesso sottoinsieme che sft_train_videoxum.py usa nella prima epoca; l'ordine
    # dei video lo decide poi il sampler del Trainer, con lo stesso seed.
    indices = list(range(len(train_dataset)))
    random.Random(args.seed).shuffle(indices)
    n_samples = min(args.debug_samples, len(train_dataset)) if args.debug_samples else len(train_dataset)
    indices = indices[:n_samples]
    steps_per_epoch = math.ceil(n_samples / args.accum_steps)
    print(f"[train] samples={n_samples}, steps={steps_per_epoch * args.epochs}")

    print(f"[model] caricamento offline di {args.model}")
    model, model_type = load_qwen_vl_model(args.model, torch_dtype=torch.bfloat16,
                                           attn_implementation=args.attn_impl)
    model = model.to(args.device)
    print(f"[model] architecture={model_type}")
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    lora_targets = [name.strip() for name in args.lora_target_modules.split(",") if name.strip()]
    config = build_config(args, save_dir)
    write_metric(metrics_file, run_id=run_id, record_type="config", arguments=vars(args),
                 trl_version=trl.__version__, sft_config=config.to_dict())
    trainer = VideoSFTTrainer(
        model=model, args=config,
        data_collator=VideoSFTCollator(train_dataset, processor, args, model_type),
        train_dataset=Dataset.from_dict({"sample_index": indices}),
        processing_class=processor,
        peft_config=LoraConfig(r=args.lora_r, lora_alpha=2*args.lora_r, lora_dropout=args.lora_dropout,
                               target_modules=lora_targets, bias="none", task_type="CAUSAL_LM"),
        cli_args=args, metrics_file=metrics_file, run_id=run_id,
    )
    schedule = schedule_cls(trainer, val_dataset, args, metrics_file, run_id, save_dir, model_type,
                            steps_per_epoch, start_time)
    trainer.add_callback(schedule)
    print(f"[lora] target_modules={lora_targets} loss_normalization={args.loss_normalization}")
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
    device = trainer.accelerator.device
    if args.diversity_check_samples > 0:
        measure_generation_diversity(trainer.model, processor, val_dataset, args, device,
                                     metrics_file, run_id, args.epochs, model_type)
    if args.video_dependence_samples > 0:
        measure_video_dependence(trainer.model, processor, val_dataset, args, device,
                                 metrics_file, run_id, args.epochs, model_type)
    metrics_file.close()
    print(f"[done] adapter finale salvato in {final_path}")
    print(f"[done] metriche salvate in {metrics_path}")


def parse_args():
    """Flag propri della versione TRL, poi la CLI di sft_train_videoxum.py sul resto."""
    parser = argparse.ArgumentParser(add_help=False, description="Opzioni TRL di sft_train_videoxum_trl.py "
                                     "(le altre sono quelle di sft_train_videoxum.py)")
    parser.add_argument("--loss_normalization", choices=("token", "sample"), default="token",
        help="token: ogni token del target pesa uguale su tutta la finestra di accumulo "
             "(standard TRL/HF); sample: media per video, come sft_train_videoxum.py")
    parser.add_argument("--dataloader_num_workers", type=int, default=2,
        help="processi che decodificano i video in parallelo al training (0: nel processo principale)")
    if {"-h", "--help"} & set(sys.argv[1:]): parser.print_help(); print()
    trl_args, rest = parser.parse_known_args()
    if trl_args.dataloader_num_workers < 0: parser.error("--dataloader_num_workers deve essere >= 0")
    sys.argv = [sys.argv[0], *rest]
    args = sft_train_videoxum.parse_args()
    vars(args).update(vars(trl_args))
    if args.eval_only or args.eval_checkpoint is not None:
        raise SystemExit("--eval_only resta in sft_train_videoxum.py, che valuta anche i checkpoint "
                         "salvati da questo script")
    if args.tune_visual_merger:
        raise SystemExit("--tune_visual_merger non e' supportato: con PEFT i pesi sbloccati fuori da "
                         "LoRA non vengono salvati nell'adapter")
    if args.eval_steps and 0 < args.eval_samples < args.eval_steps_samples:
        raise SystemExit("--eval_samples deve essere >= --eval_steps_samples: checkpoint-best "
                         "confronta tutte le validation sui primi --eval_steps_samples video")
    return args


if __name__ == "__main__": train(parse_args())

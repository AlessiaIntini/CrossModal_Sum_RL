"""Pieces shared by the trainers: TRL version, run logs, validation schedule and time budget."""

import math
import random
import sys
import time
from dataclasses import dataclass

import torch
import trl
from transformers import TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from ..cli import TIME_BUDGET_EXIT_CODE
from ..jsonl import new_run_id, write_record

TRL_VERSION = "1.13.0"
RESUME_POINTER_NAME = "resume_pointer.txt"


def check_environment(args):
    # The trainers override private TRL methods, so the TRL version is pinned.
    if trl.__version__ != TRL_VERSION:
        raise SystemExit(f"[trl] trl=={TRL_VERSION} is required, found {trl.__version__}")
    if not args.model.is_dir():
        raise FileNotFoundError(f"base model not found: {args.model}")
    # Training uses the GPU visible to the process (select it with CUDA_VISIBLE_DEVICES).
    if args.device not in ("cuda", "cuda:0"):
        raise SystemExit("--device must be cuda:0 (select the GPU with CUDA_VISIBLE_DEVICES)")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available: run on a GPU node")
    # Only Trainer checkpoints (checkpoint-N with trainer_state.json) can be resumed.
    if args.resume_from is not None and not (args.resume_from / "trainer_state.json").is_file():
        raise SystemExit(f"[resume] {args.resume_from} is not a resumable checkpoint-N "
                         "(trainer_state.json is missing)")


@dataclass
class RunLog:
    # Checkpoint directory and JSONL metrics log of a training run.
    save_dir: object
    metrics_path: object
    file: object
    run_id: str

    def write(self, **values):
        # Every record is tagged with the run id.
        write_record(self.file, run_id=self.run_id, **values)


def open_run_log(args):
    # Checkpoints go to <saveroot>/<model>, logs to <resultsroot>/<model>/metrics-<run_id>.jsonl.
    save_dir = args.saveroot / args.model.name
    results_dir = args.resultsroot / args.model.name
    save_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    run_id = new_run_id()
    metrics_path = results_dir / f"metrics-{run_id}.jsonl"
    print(f"[metrics] JSONL log: {metrics_path}")
    return RunLog(save_dir, metrics_path, metrics_path.open("w", encoding="utf-8", buffering=1), run_id)


def subset_indices(size, limit, seed, always_shuffle=False):
    # A fixed subset of `limit` indices (all if limit is 0), shuffled with `seed`; without
    # always_shuffle the full set keeps its natural order.
    indices = list(range(size))
    if always_shuffle or 0 < limit < size:
        random.Random(seed).shuffle(indices)
    return indices[:limit] if 0 < limit < size else indices


def trainer_arguments(args, save_dir):
    # TrainingArguments shared by SFT and DPO: one video per micro-step and a cosine schedule. A
    # warmup_steps value below 1 is read by transformers as a fraction of the training steps.
    return {
        "output_dir": str(save_dir),
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": args.accum_steps,
        "num_train_epochs": args.epochs,
        "learning_rate": args.lr, "lr_scheduler_type": "cosine", "warmup_steps": args.warmup_ratio,
        "weight_decay": args.weight_decay, "max_grad_norm": args.max_grad_norm, "optim": "adamw_torch",
        "bf16": True, "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "save_strategy": "steps", "save_steps": args.save_steps, "logging_steps": 1,
        "eval_strategy": "no", "report_to": "none", "remove_unused_columns": False,
        "dataloader_num_workers": args.dataloader_num_workers, "seed": args.seed,
    }


def epoch_of(state):
    # 1-based epoch of the current step.
    return math.floor(state.epoch or 0) + 1


class ValidationSchedule(TrainerCallback):
    def __init__(self, trainer, validate, save, args, log, steps_per_epoch, start_time,
                 epoch_metric=("f1_mean", "F1_MEAN"), save_before_validation=False):
        # Periodic and end-of-epoch validation, checkpoint-best/epoch saving and SLURM time budget.
        self.trainer, self.validate, self.save, self.args, self.log = trainer, validate, save, args, log
        self.steps_per_epoch, self.start_time = steps_per_epoch, start_time
        self.epoch_metric, self.save_before_validation = epoch_metric, save_before_validation
        self.out_of_time = False

    def elapsed_hours(self):
        # Hours since the start of the job.
        return (time.monotonic() - self.start_time) / 3600

    def _check_budget(self, control):
        # Out of time: save a resumable checkpoint-N and stop at the end of this step.
        if self.args.time_budget_hours and self.elapsed_hours() >= self.args.time_budget_hours:
            self.out_of_time = True
            control.should_save = True
            control.should_training_stop = True
        return self.out_of_time

    def _validate(self, state, epoch, limit=None):
        metrics = self.validate(limit=limit, epoch=epoch, step=state.global_step)
        self.trainer.model.train()
        # checkpoint-best is tracked in the TrainerState, so it survives resumes.
        if state.best_metric is None or metrics["selection_score"] > state.best_metric:
            state.best_metric = metrics["selection_score"]
            self.save(self.log.save_dir / "checkpoint-best")
            print(f"[best] checkpoint-best updated: {self.args.best_metric}={state.best_metric:.4f}")
        return metrics

    def _end_of_epoch(self, state, epoch):
        # DPO also saves the epoch adapter before validating, so it survives if SLURM ends the job
        # during validation.
        if self.save_before_validation:
            self.save(self.log.save_dir / f"checkpoint-epoch{epoch}-preval")
        metrics = self._validate(state, epoch)
        self.save(self.log.save_dir / f"checkpoint-epoch{epoch}")
        key, label = self.epoch_metric
        print(f"[epoch {epoch}] {label}={metrics[key]:.4f}")

    def on_train_begin(self, args, state, control, **kwargs):
        # A job resumed exactly at the end of an epoch runs the end-of-epoch validation it missed.
        epoch = state.global_step // self.steps_per_epoch
        if (state.global_step and state.global_step % self.steps_per_epoch == 0
                and not (self.log.save_dir / f"checkpoint-epoch{epoch}").exists()):
            self._end_of_epoch(state, epoch)

    def on_step_end(self, args, state, control, **kwargs):
        # The budget is checked before every validation, so a job close to its time limit saves and
        # stops instead of validating.
        if self._check_budget(control):
            return
        # Periodic validation on --eval_steps_samples videos, except at the end of an epoch, which
        # has its own.
        if (self.args.eval_steps and state.global_step % self.args.eval_steps == 0
                and state.global_step % self.steps_per_epoch):
            self._validate(state, state.global_step // self.steps_per_epoch + 1, limit=self.args.eval_steps_samples)
            self._check_budget(control)

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.out_of_time or self._check_budget(control):
            return
        self._end_of_epoch(state, round(state.epoch))

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Trainer logs (loss, learning rate, ...) go to the JSONL log as well.
        if logs:
            self.log.write(record_type="train_log", **{"global_step": state.global_step, "epoch": state.epoch, **logs})


def exit_if_out_of_time(trainer, schedule, log, args):
    # After a time-budget stop, point the SLURM chain to checkpoint-N and exit with code 85, so the
    # launcher requeues the job with --resume_from.
    if not schedule.out_of_time:
        return
    checkpoint = log.save_dir / f"{PREFIX_CHECKPOINT_DIR}-{trainer.state.global_step}"
    (log.save_dir / RESUME_POINTER_NAME).write_text(f"{checkpoint}\n", encoding="utf-8")
    log.write(record_type="time_budget_exit", elapsed_hours=schedule.elapsed_hours(),
              optimizer_step=trainer.state.global_step, epoch=trainer.state.epoch, resume_from=checkpoint)
    log.file.close()
    print(f"[time budget] {schedule.elapsed_hours():.2f}h >= {args.time_budget_hours}h: resume from {checkpoint}")
    sys.exit(TIME_BUDGET_EXIT_CODE)

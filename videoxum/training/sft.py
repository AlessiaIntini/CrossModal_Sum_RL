"""Supervised fine-tuning with trl.SFTTrainer on V2V, V2T and V2VT targets.

TRL 1.13 prepares image inputs only, so a collator builds the video inputs of one video per
micro-step; loss, accumulation, scheduler, checkpoints and resume are those of SFTTrainer.
"""

import math
import random
import sys
import time
from pathlib import Path

import datasets
import torch
import trl
from peft import LoraConfig
from transformers import AutoProcessor
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from trl import SFTConfig, SFTTrainer

from ..data import (JointDataset, SelectionDataset, SummaryDataset, load_joint_records, load_selection_records,
                    load_summary_records, split_records)
from ..inputs import prepare_completion
from ..models import load_model
from ..video import guard_checkpoint_dir, resolve_video_encoding, write_video_encoding
from .common import (ValidationSchedule, check_environment, epoch_of, exit_if_out_of_time, open_run_log,
                     subset_indices, trainer_arguments)
from .validation import (evaluate_joint, evaluate_selection, evaluate_summary, measure_generation_diversity,
                         measure_video_dependence, resolve_best_metric, sample_frames)

SFT_TASKS = ("v2v", "v2t", "v2vt")
VALIDATIONS = {"v2v": evaluate_selection, "v2t": evaluate_summary, "v2vt": evaluate_joint}


def task_records(args, path, with_target):
    # Records of the SFT task; validation records have no target and are scored on frames.
    if args.task == "v2t":
        return load_summary_records(path)
    if args.task == "v2vt":
        return load_joint_records(path, with_target, args.timeline_unit if with_target else "frame")
    return load_selection_records(path, with_target, args.timeline_unit if with_target else "frame")


def task_dataset(args, records, video_dir):
    # Dataset of the SFT task, with its prompt.
    if args.task == "v2t":
        return SummaryDataset(records, video_dir, args.timeline_unit, args.summary_style)
    if args.task == "v2vt":
        return JointDataset(records, video_dir, args.timeline_unit, args.summary_style, args.index_phrase)
    return SelectionDataset(records, video_dir, args.timeline_unit)


def load_sft_datasets(args):
    # (train, validation) datasets; without --val_json the validation set is split from the training
    # data.
    records = task_records(args, args.data_json, True)
    if args.val_json:
        val_records = task_records(args, args.val_json, False)
    else:
        records, val_records = split_records(records, args.val_ratio, args.seed)
    val_video_dir = args.val_video_dir or args.video_dir
    train, val = task_dataset(args, records, args.video_dir), task_dataset(args, val_records, val_video_dir)
    print(f"[data] train={len(train)}, val={len(val)}, val_video_dir={val_video_dir}")
    return train, val


def load_sft_validation_dataset(args):
    # Validation dataset alone, for --eval_only.
    if args.val_json:
        records = task_records(args, args.val_json, False)
    else:
        _, records = split_records(task_records(args, args.data_json, True), args.val_ratio, args.seed)
    val_video_dir = args.val_video_dir or args.video_dir
    dataset = task_dataset(args, records, val_video_dir)
    print(f"[data] val={len(dataset)}, val_video_dir={val_video_dir}")
    return dataset


def save_named(trainer, checkpoint, args):
    # Named checkpoint (best, epoch, final): adapter, processor and video_encoding.json.
    trainer.save_model(str(checkpoint))
    write_video_encoding(checkpoint, args)


class SFTCollator:
    def __init__(self, dataset, processor, args, model_type):
        self.dataset, self.processor, self.args, self.model_type = dataset, processor, args, model_type

    def __call__(self, rows):
        # One video per batch: prompt + target with the frames of the video.
        if len(rows) != 1:
            raise ValueError("per_device_train_batch_size must be 1: videos are not packed")
        index = rows[0]["sample_index"]
        sample = self.dataset[index]
        inputs, prompt_len = prepare_completion(self.processor, sample, sample["target"], self.args,
                                                self.model_type, sample_frames(sample, self.args))
        # Only the target tokens are trained: labels are -100 on the prompt.
        labels = inputs["input_ids"].clone()
        labels[:, :prompt_len] = -100
        return {**inputs, "labels": labels, "sample_index": torch.tensor([index])}


class VideoSFTTrainer(SFTTrainer):
    def __init__(self, *args, cli_args, log, **kwargs):
        super().__init__(*args, **kwargs)
        self.cli_args, self.run_log = cli_args, log
        self._sample_index = None
        # With loss_normalization "sample" the model loss is the mean over the tokens of the video
        # and the Trainer divides it by the accumulation steps, so every video weighs the same; with
        # "token" (TRL default) every target token of the accumulation window does.
        if cli_args.loss_normalization == "sample":
            self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # sample_index is used only by the logs: the model does not accept it.
        self._sample_index = int(inputs.pop("sample_index")[0])
        tokens = int((inputs["labels"][:, 1:] != -100).sum())
        result = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        if self.model.training:
            loss = (result[0] if return_outputs else result).detach().float().item()
            # With token normalization the loss is a sum over the tokens of the window: sft_loss
            # brings it back to the mean over the video, so the logs stay comparable between the two
            # normalizations.
            if self.model_accepts_loss_kwargs and num_items_in_batch is not None:
                loss *= float(num_items_in_batch) / max(tokens, 1)
            self.run_log.write(record_type="sample", split="train", epoch=epoch_of(self.state),
                               dataset_index=self._sample_index, optimizer_step=self.state.global_step + 1,
                               completion_tokens=tokens, sft_loss=loss, learning_rate=self._get_learning_rate())
        return result

    def training_step(self, model, inputs, num_items_in_batch=None):
        try:
            return super().training_step(model, inputs, num_items_in_batch)
        except torch.OutOfMemoryError as error:
            # With several processes a rank that skips the backward would block the others in the
            # all-reduce.
            if self.accelerator.num_processes > 1:
                raise
            message = str(error)
        # A video that runs out of memory is skipped, and the possibly partial gradients of the
        # window are discarded.
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        self.run_log.write(record_type="sample", split="train", epoch=epoch_of(self.state),
                           dataset_index=self._sample_index, optimizer_step=self.state.global_step + 1,
                           skipped_oom=True, error=message)
        print(f"[warning] out of memory on sample {self._sample_index}: skipped", file=sys.stderr)
        return torch.zeros((), device=self.accelerator.device)

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        # Resumable checkpoint-N also get video_encoding.json.
        if self.args.should_save:
            checkpoint = Path(self._get_output_dir(trial=trial)) / f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            write_video_encoding(checkpoint, self.cli_args)


def train_sft(args):
    # Train a LoRA adapter on the SFT task, validating and saving checkpoint-best, checkpoint-epochN
    # and final.
    check_environment(args)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    start_time = time.monotonic()
    resolve_video_encoding(args, args.resume_from, default="seconds")
    resolve_best_metric(args)
    # A new run must not mix its checkpoints with those of another encoding or task.
    if not args.resume_from:
        guard_checkpoint_dir(args.saveroot / args.model.name, args, None if args.task == "v2v" else args.task)
    log = open_run_log(args)
    train_dataset, val_dataset = load_sft_datasets(args)
    # A fixed subset of --debug_samples videos (all if 0); the order of the videos is then set by
    # the Trainer sampler, with the same seed.
    indices = subset_indices(len(train_dataset), args.debug_samples, args.seed, always_shuffle=True)
    steps_per_epoch = math.ceil(len(indices) / args.accum_steps)
    print(f"[train] task={args.task} samples={len(indices)}, steps={steps_per_epoch * args.epochs}")

    print(f"[model] loading {args.model}")
    model, model_type = load_model(args.model, dtype=torch.bfloat16, attn_implementation=args.attn_impl)
    model = model.to(args.device)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    lora_targets = [name.strip() for name in args.lora_target_modules.split(",") if name.strip()]
    # The collator builds inputs and labels: TRL must not tokenize or truncate them (max_length
    # would cut video tokens).
    config = SFTConfig(**trainer_arguments(args, log.save_dir), dataset_kwargs={"skip_prepare_dataset": True},
                       max_length=None, loss_type="nll")
    log.write(record_type="config", arguments=vars(args), trl_version=trl.__version__, sft_config=config.to_dict())
    trainer = VideoSFTTrainer(
        model=model, args=config,
        data_collator=SFTCollator(train_dataset, processor, args, model_type),
        train_dataset=datasets.Dataset.from_dict({"sample_index": indices}),
        processing_class=processor,
        peft_config=LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=args.lora_dropout,
                               target_modules=lora_targets, bias="none", task_type="CAUSAL_LM"),
        cli_args=args, log=log,
    )
    validate = VALIDATIONS[args.task]
    schedule = ValidationSchedule(
        trainer,
        validate=lambda limit, epoch, step: validate(trainer.model, trainer.processing_class, val_dataset, args,
                                                     trainer.accelerator.device, log, epoch, model_type,
                                                     limit=limit, step=step),
        save=lambda path: save_named(trainer, path, args),
        args=args, log=log, steps_per_epoch=steps_per_epoch, start_time=start_time,
        epoch_metric=("rougeL", "ROUGE-L") if args.task == "v2t" else ("f1_mean", "F1_MEAN"),
    )
    trainer.add_callback(schedule)
    print(f"[lora] target_modules={lora_targets} loss_normalization={args.loss_normalization}")
    trainer.model.print_trainable_parameters()
    trainer.train(resume_from_checkpoint=str(args.resume_from) if args.resume_from else None)
    # Out of time: the Trainer has just saved checkpoint-N and the SLURM launcher resumes from it.
    exit_if_out_of_time(trainer, schedule, log, args)

    final_path = log.save_dir / "final"
    save_named(trainer, final_path, args)
    device = trainer.accelerator.device
    # Optional diagnostics of the final adapter.
    if args.diversity_check_samples > 0:
        measure_generation_diversity(trainer.model, processor, val_dataset, args, device, log, args.epochs, model_type)
    if args.video_dependence_samples > 0:
        measure_video_dependence(trainer.model, processor, val_dataset, args, device, log, args.epochs, model_type)
    log.file.close()
    print(f"[done] final adapter: {final_path}")
    print(f"[done] metrics: {log.metrics_path}")

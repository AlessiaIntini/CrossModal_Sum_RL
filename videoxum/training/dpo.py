"""Direct Preference Optimization with trl.DPOTrainer on chosen/rejected interval pairs.

A collator builds one [chosen, rejected] pair of the same video per micro-step. With
--init_from the reference is a frozen copy ("ref" adapter) of the SFT adapter, without a
second model in memory; otherwise it is the base model with the adapter disabled. Dropout is
disabled (LoRA dropout included) because the reference forward runs in train mode.
"""

import json
import math
import os
import random
import sys
import time
from pathlib import Path

import datasets
import torch
import trl
from peft import LoraConfig, PeftModel
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoProcessor
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from trl import DPOConfig, DPOTrainer

from ..data import SelectionDataset, load_preference_records, split_records
from ..inputs import prepare_completion
from ..models import load_model
from ..video import guard_checkpoint_dir, resolve_video_encoding, write_video_encoding
from .common import (ValidationSchedule, check_environment, epoch_of, exit_if_out_of_time, open_run_log,
                     subset_indices, trainer_arguments)
from .validation import evaluate_selection, resolve_best_metric, sample_frames

DPO_INIT_NAME = "dpo_init.json"
TOKEN_KEYS = ("input_ids", "attention_mask", "mm_token_type_ids")


def load_dpo_datasets(args):
    # (train, validation) preference datasets with the DPO prompt wording ("indexed"); validation
    # pairs are optional.
    records = load_preference_records(args.data_json, True, args.timeline_unit)
    if args.val_json:
        val_records = load_preference_records(args.val_json, False)
    else:
        records, val_records = split_records(records, args.val_ratio, args.seed)
    val_video_dir = args.val_video_dir or args.video_dir
    train = SelectionDataset(records, args.video_dir, args.timeline_unit, "indexed")
    val = SelectionDataset(val_records, val_video_dir, args.timeline_unit, "indexed")
    print(f"[data] train={len(train)}, val={len(val)}, val_video_dir={val_video_dir}")
    return train, val


def load_dpo_validation_dataset(args):
    # Validation dataset alone, for --eval_only.
    if args.val_json:
        records = load_preference_records(args.val_json, False)
    else:
        _, records = split_records(load_preference_records(args.data_json, True, args.timeline_unit),
                                   args.val_ratio, args.seed)
    val_video_dir = args.val_video_dir or args.video_dir
    dataset = SelectionDataset(records, val_video_dir, args.timeline_unit, "indexed")
    print(f"[data] val={len(dataset)}, val_video_dir={val_video_dir}")
    return dataset


def write_sidecars(checkpoint, args):
    # video_encoding.json for inference and GRPO, dpo_init.json to rebuild the reference on resume.
    write_video_encoding(checkpoint, args)
    (Path(checkpoint) / DPO_INIT_NAME).write_text(
        json.dumps({"init_from": str(args.init_from) if args.init_from else None}, indent=2) + "\n",
        encoding="utf-8")


def read_dpo_init(checkpoint):
    path = Path(checkpoint) / DPO_INIT_NAME
    # (found, init_from) recorded by write_sidecars.
    if not path.is_file():
        return False, None
    init_from = json.loads(path.read_text(encoding="utf-8"))["init_from"]
    return True, Path(init_from) if init_from else None


def save_named(trainer, checkpoint, args):
    # Named checkpoint (best, epoch, final): adapter, processor and sidecars.
    trainer.save_model(str(checkpoint))
    write_sidecars(checkpoint, args)


def full_sequence_forward(module, args, kwargs):
    # Forward pre-hook. The collator puts only the completion tail (from the last prompt token) in
    # input_ids and completion_mask, which is all DPOTrainer needs, and the full sequence with the
    # video tokens in full_input_ids.
    full_input_ids = kwargs.pop("full_input_ids", None)
    if full_input_ids is None:
        return None
    # The model runs on the full sequence but returns the logits of the tail only: no
    # full-vocabulary logits on thousands of video tokens, and the log-probabilities are those of a
    # full forward, for both policy and reference.
    kwargs["logits_to_keep"] = kwargs["input_ids"].size(1)
    kwargs["input_ids"] = full_input_ids
    return args, kwargs


class PreferenceCollator:
    def __init__(self, dataset, processor, args, model_type):
        self.dataset, self.processor, self.args, self.model_type = dataset, processor, args, model_type

    def __call__(self, rows):
        # One pair per batch in the format of TRL's vision collator: rows [chosen, rejected] of the
        # same video.
        if len(rows) != 1:
            raise ValueError("per_device_train_batch_size must be 1: videos are not packed")
        index = rows[0]["sample_index"]
        sample = self.dataset[index]
        frames = sample_frames(sample, self.args)
        (chosen, prompt_len), (rejected, rejected_prompt_len) = (
            prepare_completion(self.processor, sample, sample[key], self.args, self.model_type, frames)
            for key in ("chosen", "rejected"))
        # Chosen and rejected must share the prompt, so they differ only in the completion.
        if rejected_prompt_len != prompt_len or not torch.equal(
                chosen["input_ids"][0, :prompt_len], rejected["input_ids"][0, :prompt_len]):
            raise RuntimeError(f"sample {index}: chosen and rejected have different prompts")
        # Token tensors are padded to the longer completion; the vision tensors of the two rows are
        # concatenated.
        batch = {}
        for key, value in chosen.items():
            if key in TOKEN_KEYS:
                pad = self.processor.tokenizer.pad_token_id if key == "input_ids" else 0
                batch[key] = pad_sequence([value[0], rejected[key][0]], batch_first=True, padding_value=pad)
            else:
                batch[key] = torch.cat([value, rejected[key]])
        completion_mask = batch["attention_mask"].clone()
        completion_mask[:, :prompt_len] = 0
        # Tail on which DPOTrainer computes the log-probabilities: it starts from the last prompt
        # token, whose logits predict the first completion token.
        tail = slice(prompt_len - 1, None)
        batch["full_input_ids"] = batch["input_ids"]
        batch["input_ids"] = batch["input_ids"][:, tail]
        batch["completion_mask"] = completion_mask[:, tail]
        batch["sample_index"] = torch.tensor([index])
        return batch


class VideoDPOTrainer(DPOTrainer):
    def __init__(self, *args, cli_args, log, **kwargs):
        super().__init__(*args, **kwargs)
        self.cli_args, self.run_log = cli_args, log
        self._sample_index = None
        # Every forward runs on the full sequence (see full_sequence_forward).
        self.model.register_forward_pre_hook(full_sequence_forward, with_kwargs=True)

    def _prepare_dataset(self, dataset, *args, **kwargs):
        # Overrides a private method of TRL 1.13, like _save below. The dataset holds only
        # sample_index: prompts, completions and frames are tokenized by the collator, as for TRL's
        # image datasets.
        return dataset

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # sample_index is used only by the logs: the model does not accept it.
        self._sample_index = int(inputs.pop("sample_index")[0])
        chosen_tokens, rejected_tokens = inputs["completion_mask"].sum(dim=1).tolist()
        result = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        if self.model.training:
            # Values just logged by DPOTrainer for this pair. The log-probabilities are sums over
            # tokens even with sigmoid_norm, which normalizes only the loss.
            last = {key: values[-1] for key, values in self._metrics["train"].items() if values}
            policy_margin = last["logps/chosen"] - last["logps/rejected"]
            self.run_log.write(record_type="sample", split="train", epoch=epoch_of(self.state),
                               dataset_index=self._sample_index, optimizer_step=self.state.global_step + 1,
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
            # With several processes a rank that skips the backward would block the others in the
            # all-reduce.
            if self.accelerator.num_processes > 1:
                raise
            message = str(error)
        # A video too long for the GPU memory is skipped, and the possibly partial gradients of the
        # window are discarded.
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        self.run_log.write(record_type="sample", split="train", epoch=epoch_of(self.state),
                           dataset_index=self._sample_index, optimizer_step=self.state.global_step + 1,
                           skipped_oom=True, error=message)
        print(f"[warning] out of memory on sample {self._sample_index}: skipped", file=sys.stderr)
        return torch.zeros((), device=self.accelerator.device)

    def _save(self, output_dir=None, state_dict=None):
        # Only the trained "default" adapter is saved. On resume the Trainer loads only adapter
        # subfolders when any exists, so saving the frozen "ref" adapter in ref/ would skip the
        # trained one; the reference is rebuilt from dpo_init.json instead.
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir, selected_adapters=["default"])
        self.processing_class.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        # Resumable checkpoint-N also get the sidecars.
        if self.args.should_save:
            checkpoint = Path(self._get_output_dir(trial=trial)) / f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            write_sidecars(checkpoint, self.cli_args)


def load_policy(args):
    # (model, model_type, processor, peft_config): a new LoRA, or the SFT adapter to continue.
    print(f"[model] loading {args.model}")
    model, model_type = load_model(args.model, dtype=torch.bfloat16, attn_implementation=args.attn_impl)
    model = model.to(args.device)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    if args.init_from is None:
        lora_targets = [name.strip() for name in args.lora_target_modules.split(",") if name.strip()]
        print(f"[lora] target_modules={lora_targets}; reference = base model")
        # Dropout 0: disable_dropout would zero it anyway, because the reference forward runs in
        # train mode.
        return model, model_type, processor, LoraConfig(
            r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
            target_modules=lora_targets, bias="none", task_type="CAUSAL_LM")
    print(f"[model] policy and reference from the SFT adapter {args.init_from}")
    # DPOTrainer copies this adapter into a frozen "ref" adapter and trains the original: the
    # reference needs no second model in memory.
    model = PeftModel.from_pretrained(model, args.init_from, is_trainable=True, local_files_only=True)
    return model, model_type, processor, None


def build_dpo_config(args, save_dir):
    # sigmoid: standard DPO on summed log-probabilities; sigmoid_norm: per-token mean
    # (--length-normalize). The collator builds inputs and masks, so TRL must not truncate them
    # (max_length would cut video tokens).
    return DPOConfig(**trainer_arguments(args, save_dir), beta=args.beta,
                     loss_type=["sigmoid_norm" if args.length_normalize else "sigmoid"],
                     disable_dropout=True, max_length=None)


def train_dpo(args):
    # Train the DPO adapter, validating and saving checkpoint-best, checkpoint-epochN and final.
    check_environment(args)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.resume_from is not None:
        # A resumable checkpoint holds only the trained adapter: the reference is rebuilt from the
        # same SFT adapter it started from.
        found, args.init_from = read_dpo_init(args.resume_from)
        if not found:
            raise SystemExit(f"[resume] {args.resume_from} does not contain {DPO_INIT_NAME}")
    start_time = time.monotonic()
    # The policy inherits the encoding of the checkpoint it resumes or starts from.
    resolve_video_encoding(args, args.resume_from or args.init_from, default="seconds")
    resolve_best_metric(args)
    if not args.resume_from:
        guard_checkpoint_dir(args.saveroot / args.model.name, args)
    log = open_run_log(args)
    train_dataset, val_dataset = load_dpo_datasets(args)
    # A fixed subset of --debug_samples pairs (all if 0); the order is then set by the Trainer
    # sampler, with the same seed.
    indices = subset_indices(len(train_dataset), args.debug_samples, args.seed, always_shuffle=True)
    steps_per_epoch = math.ceil(len(indices) / args.accum_steps)
    print(f"[train] samples={len(indices)}, steps={steps_per_epoch * args.epochs}")

    model, model_type, processor, peft_config = load_policy(args)
    config = build_dpo_config(args, log.save_dir)
    log.write(record_type="config", arguments=vars(args), trl_version=trl.__version__, dpo_config=config.to_dict())
    trainer = VideoDPOTrainer(
        model=model, args=config,
        data_collator=PreferenceCollator(train_dataset, processor, args, model_type),
        train_dataset=datasets.Dataset.from_dict({"sample_index": indices}),
        processing_class=processor, peft_config=peft_config, cli_args=args, log=log,
    )
    schedule = ValidationSchedule(
        trainer,
        validate=lambda limit, epoch, step: evaluate_selection(
            trainer.model, trainer.processing_class, val_dataset, args, trainer.accelerator.device, log, epoch,
            model_type, limit=limit, step=step, pairwise=True),
        save=lambda path: save_named(trainer, path, args),
        args=args, log=log, steps_per_epoch=steps_per_epoch, start_time=start_time, save_before_validation=True,
    )
    trainer.add_callback(schedule)
    print(f"[dpo] loss_type={config.loss_type} beta={config.beta}")
    trainer.model.print_trainable_parameters()
    trainer.train(resume_from_checkpoint=str(args.resume_from) if args.resume_from else None)
    # Out of time: the Trainer has just saved checkpoint-N and the SLURM launcher resumes from it.
    exit_if_out_of_time(trainer, schedule, log, args)

    final_path = log.save_dir / "final"
    save_named(trainer, final_path, args)
    log.file.close()
    print(f"[done] final adapter: {final_path}")
    print(f"[done] metrics: {log.metrics_path}")

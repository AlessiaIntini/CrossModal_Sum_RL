"""GRPO with trl.GRPOTrainer on the joint task (frame selection + summary).

Group advantages, KL (k3) to the model with the adapter disabled, loss, accumulation,
checkpoints and resume are those of GRPOTrainer. TRL 1.13 builds multimodal inputs from images
only, so the subclass adds the video of each generation group to prompts and forwards. With
--init_from the SFT/DPO adapter is merged into the base model, which is also the KL reference.
"""

import math
import traceback
from pathlib import Path

import datasets
import numpy as np
import torch
import trl
from peft import LoraConfig, PeftModel
from tqdm.auto import tqdm
from transformers import AutoProcessor, TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from trl import GRPOConfig, GRPOTrainer

from ..data import GRPODataset, load_grpo_records, split_records
from ..inputs import encode_prompt
from ..models import load_model, read_grpo_init, write_grpo_init
from ..prompts import SUMMARY_WORD_RANGE
from ..rewards import DEFAULT_REWARD_WEIGHTS, SentenceTransformerEncoder, VideoSummaryReward, ensure_video_prepared
from ..video import resolve_video_encoding, write_video_encoding
from .common import check_environment, open_run_log, subset_indices
from .validation import evaluate_grpo, sample_frames

LOSS_TYPES = ("grpo", "dapo", "dr_grpo", "bnpo")
SCALE_REWARDS = ("group", "batch", "none")
VIDEO_KEYS = ("pixel_values_videos", "video_grid_thw", "second_per_grid_ts")
LOGGED_COMPONENTS = ("total_reward_default", "valid", "format_reward", "f_beta_reward",
                     "representativeness_reward", "grounding_reward", "reference_reward",
                     "summary_length_reward", "length_budget_reward", "saliency_rank_reward",
                     "summary_words", "selected_ratio")


def load_grpo_datasets(args):
    # (train, validation) GRPO datasets; without --val_json the validation set is split from the
    # training data.
    train = load_grpo_records(args.data_json)
    if args.val_json:
        val = load_grpo_records(args.val_json)
    else:
        train, val = split_records(train, args.val_ratio, args.seed)
    return (GRPODataset(train, args.video_dir, args.feature_dir, "train", args.summary_style, args.timeline_unit),
            GRPODataset(val, args.val_video_dir or args.video_dir, args.val_feature_dir or args.feature_dir,
                        "val", args.summary_style, args.timeline_unit))


def write_sidecars(checkpoint, args):
    # video_encoding.json and grpo_init.json, read by inference next to the adapter.
    write_video_encoding(checkpoint, args)
    write_grpo_init(checkpoint, args.init_from)


def save_named(trainer, checkpoint, args):
    # Named checkpoint (epoch, best, final): adapter, processor and sidecars.
    trainer.save_model(str(checkpoint))
    write_sidecars(checkpoint, args)


class VideoForward:
    def __init__(self, model, video, token_types):
        # Model wrapper that adds the tensors of the current video to every forward of GRPOTrainer.
        self.model, self.video, self.token_types = model, video, token_types

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, input_ids, **kwargs):
        rows = input_ids.size(0)
        # The visual tensors of the prompt are repeated for every row of the batch.
        extra = {key: torch.cat([value] * rows) for key, value in self.video.items()}
        # mm_token_type_ids (image 1, video 2, as in processor.create_mm_token_type_ids) are rebuilt
        # on the full prompt + completion sequence: Qwen3-VL needs them for its M-RoPE positions.
        if self.token_types:
            types = torch.zeros_like(input_ids)
            for token_id, value in self.token_types.items():
                types[input_ids == token_id] = value
            extra["mm_token_type_ids"] = types
        return self.model(input_ids=input_ids, **kwargs, **extra)


class VideoGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, video_dataset, cli_args, model_type, **kwargs):
        super().__init__(*args, **kwargs)
        # One video (its G rollouts) per generation batch. The overrides use private methods and
        # attributes of TRL 1.13 (_tokenize_prompts, _step, _buffered_inputs, ...): re-check them
        # before upgrading TRL.
        self.video_dataset, self.cli_args, self.model_type = video_dataset, cli_args, model_type
        self._batch_indices = self._video = self._token_types = None

    def _generate_and_score_completions(self, inputs):
        # _tokenize_prompts receives only the prompts: the sample index is kept to find the video.
        self._batch_indices = [row["sample_index"] for row in inputs]
        return super()._generate_and_score_completions(inputs)

    def _tokenize_prompts(self, prompts):
        if len(set(self._batch_indices)) != 1:
            raise ValueError("every generation batch must contain one video: use "
                             "per_device_train_batch_size=1 and steps_per_generation=num_generations")
        # Same decoding and preprocessing as validation and inference (1 fps frames, temporal
        # markers, pixel budget), done once per group.
        sample = self.video_dataset[self._batch_indices[0]]
        inputs = encode_prompt(self.processing_class, sample, self.cli_args, self.model_type,
                               sample_frames(sample, self.cli_args), torch.device("cpu"))
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
        # TRL disables gradient checkpointing during generation and does not restore it after an
        # exception: without this the rest of training would run without checkpointing.
        model = self.accelerator.unwrap_model(self.model)
        if self.args.gradient_checkpointing and not model.is_gradient_checkpointing:
            model.gradient_checkpointing_enable(self.args.gradient_checkpointing_kwargs)

    def _generate_single_turn(self, prompt_ids, images, multimodal_fields, has_tool_images=False):
        try:
            return super()._generate_single_turn(prompt_ids, images, multimodal_fields, has_tool_images)
        except torch.OutOfMemoryError:
            # A group that runs out of memory is generated one rollout at a time.
            if len(prompt_ids) == 1:
                raise
        print("[warning] rollout batch out of memory; generating one rollout at a time")
        self._restore_gradient_checkpointing()
        torch.cuda.empty_cache()
        single = dict(self._video)
        if "mm_token_type_ids" in multimodal_fields:
            single["mm_token_type_ids"] = multimodal_fields["mm_token_type_ids"][:1]
        completion_ids = []
        for ids in prompt_ids:
            completion, _ = super()._generate_single_turn([ids], images, single, has_tool_images)
            completion_ids.extend(completion)
            torch.cuda.empty_cache()
        return completion_ids, None

    def _get_per_token_logps_and_entropies(self, model, *args, **kwargs):
        # Policy and reference forwards both see the video of the group.
        return super()._get_per_token_logps_and_entropies(
            VideoForward(model, self._video, self._token_types), *args, **kwargs)

    def training_step(self, model, inputs, num_items_in_batch):
        try:
            return super().training_step(model, inputs, num_items_in_batch)
        except Exception as error:
            # With several processes a rank that skips the backward would block the others in the
            # all-reduce.
            if self.accelerator.num_processes > 1:
                raise
            print(f"[warning] skip micro-step {self._step}: {type(error).__name__}: {error}")
            traceback.print_exc()
        # A failing micro-step is skipped: the gradients of the accumulation window may hold a
        # partial backward and are dropped, and TRL must generate again (_buffered_inputs=None) and
        # advance its counter, otherwise it would retry the same buffer slice forever.
        model.zero_grad(set_to_none=True)
        self._buffered_inputs = None
        self._step += 1
        self._restore_gradient_checkpointing()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return torch.zeros((), device=self.accelerator.device)

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        # Resumable checkpoint-N also get the sidecars.
        if self.args.should_save:
            checkpoint = Path(self._get_output_dir(trial=trial)) / f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            write_sidecars(checkpoint, self.cli_args)


def make_reward_function(reward, dataset, processor, log):
    # TRL reward function: scores every rollout with the video reward and logs every group.
    def video_summary(completion_ids, sample_index, log_metric, trainer_state, **kwargs):
        texts = [text.strip() for text in processor.batch_decode(
            completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)]
        samples = {index: dataset[index] for index in dict.fromkeys(sample_index)}
        rows = []
        for text, index in zip(texts, sample_index):
            ensure_video_prepared(reward, samples[index])
            rows.append(reward.score(text, samples[index]["cache_key"]))
        # Mean reward components in the TRL logs, next to rewards/video_summary.
        for name in LOGGED_COMPONENTS:
            log_metric(f"reward/{name}", float(np.mean([row[name] for row in rows])))
        memory = torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else None
        for index, sample in samples.items():
            group = [i for i, value in enumerate(sample_index) if value == index]
            log.write(record_type="sample", split="train", epoch=math.floor(trainer_state.epoch or 0) + 1,
                      global_step=trainer_state.global_step, video_id=sample["video_id"], count=sample["count"],
                      completions=[texts[i] for i in group], reward_components=[rows[i] for i in group],
                      total_reward=float(np.mean([rows[i]["total_reward"] for i in group])),
                      max_memory_allocated_gb=memory)
        return [row["total_reward"] for row in rows]
    return video_summary


class JsonlMetrics(TrainerCallback):
    def __init__(self, log):
        self.log = log

    def on_log(self, args, state, control, logs=None, **kwargs):
        # TRL logs (loss, kl, rewards, entropy, ...) go to the JSONL metrics file; logs already
        # contain "epoch", so the keys are merged.
        if logs:
            self.log.write(record_type="train_log", **{"global_step": state.global_step, "epoch": state.epoch, **logs})


class EpochValidation(TrainerCallback):
    def __init__(self, trainer, val_set, reward, args, log, model_type, save_dir):
        self.trainer, self.val_set, self.reward, self.args = trainer, val_set, reward, args
        self.log, self.model_type, self.save_dir, self.best = log, model_type, save_dir, -float("inf")

    def on_epoch_end(self, args, state, control, **kwargs):
        trainer, epoch = self.trainer, round(state.epoch)
        # Saved before validation, so the epoch model survives if SLURM ends the job during
        # validation.
        save_named(trainer, self.save_dir / f"checkpoint-epoch{epoch}-preval", self.args)
        metrics = evaluate_grpo(trainer.model, trainer.processing_class, self.val_set, self.reward, self.args,
                                trainer.accelerator.device, self.log, epoch, self.model_type)
        trainer.model.train()
        save_named(trainer, self.save_dir / f"checkpoint-epoch{epoch}", self.args)
        # checkpoint-best uses the reward with the default weights, comparable across ablations.
        if metrics.get("total_reward_default", -1) > self.best:
            self.best = metrics["total_reward_default"]
            save_named(trainer, self.save_dir / "checkpoint-best", self.args)


def build_reward(args):
    # VideoSummaryReward configured from the command line.
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
    print(f"[reward] weights={reward.weights} length_penalty={reward.use_length_penalty} "
          f"f_beta={args.reward_f_beta} f_beta_aggregation={reward.f_beta_aggregation} "
          f"summary_style={args.summary_style} summary_word_range={SUMMARY_WORD_RANGE} "
          f"summary_length_sigma={reward.summary_length_sigma} "
          f"summary_target_words={reward.summary_target_words or 'per-video'} "
          f"length_budget_sigma={reward.length_budget_sigma} "
          f"reference_cosine=clamp((cos-{reward.reference_cosine_floor})/{reward.reference_cosine_scale}) "
          f"timeline_unit={reward.timeline_unit}")
    return reward


def cache_train_rewards(reward, dataset, indices):
    # Prepare the reward caches once and drop videos whose features cannot be scored.
    valid = []
    for index in tqdm(indices, desc="reward cache"):
        sample = dataset[index]
        try:
            ensure_video_prepared(reward, sample)
        except Exception as error:
            print(f"[warning] excluded {sample['video_id']}: {type(error).__name__}: {error}")
            continue
        valid.append(index)
    print(f"[data] training videos with a valid reward: {len(valid)}/{len(indices)}")
    return valid


def train_grpo(args):
    # Train the GRPO adapter on top of the base model, or of a merged SFT/DPO adapter.
    check_environment(args)
    if args.init_from is not None and not args.init_from.is_dir():
        raise FileNotFoundError(f"starting PEFT checkpoint not found: {args.init_from}")
    # The LoRA to resume is a delta on the model with the starting adapter merged, so the resume
    # needs the same starting adapter.
    if args.resume_from is not None:
        found, saved_init = read_grpo_init(args.resume_from)
        if found and args.init_from is None:
            args.init_from = saved_init
        elif found and saved_init is not None and Path(args.init_from) != saved_init:
            raise SystemExit(f"[model] {args.resume_from} starts from {saved_init}, not from {args.init_from}")
    # A resume inherits the encoding of its checkpoint, a warm start that of the SFT/DPO checkpoint.
    resolve_video_encoding(args, args.resume_from or args.init_from, default="seconds")
    train_set, val_set = load_grpo_datasets(args)
    reward = build_reward(args)
    log = open_run_log(args)
    save_dir = log.save_dir
    train_indices = cache_train_rewards(
        reward, train_set, subset_indices(len(train_set), args.train_samples or args.debug_samples, args.seed))
    print(f"[data] training={len(train_indices)}/{len(train_set)}, validation="
          f"{min(args.val_samples, len(val_set)) if args.val_samples else len(val_set)}/{len(val_set)}")

    model, model_type = load_model(args.model, dtype=torch.bfloat16, attn_implementation=args.attn_impl)
    model = model.to(args.device)
    print(f"[model] architecture={model_type}")
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    # The starting adapter is merged into the base weights: the merged model is both the warm start
    # and the KL reference (adapter disabled).
    if args.init_from is not None:
        model = PeftModel.from_pretrained(model, args.init_from, is_trainable=False,
                                          local_files_only=True).merge_and_unload()
        print(f"[model] warm start and KL reference from the PEFT checkpoint {args.init_from}")
    else:
        print("[model] no PEFT checkpoint: warm start and KL reference from the base model")

    # One micro-step per rollout and one generation batch per video, with accum_steps videos per
    # update.
    config = GRPOConfig(
        output_dir=str(save_dir),
        per_device_train_batch_size=1,
        num_generations=args.num_generations,
        steps_per_generation=args.num_generations,
        gradient_accumulation_steps=args.num_generations * args.accum_steps,
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
    log.write(record_type="config", arguments=vars(args), trl_version=trl.__version__, grpo_config=config.to_dict())
    dataset = datasets.Dataset.from_dict({"prompt": [train_set[i]["prompt"] for i in train_indices],
                                          "sample_index": train_indices})
    trainer = VideoGRPOTrainer(
        model=model,
        reward_funcs=make_reward_function(reward, train_set, processor, log),
        args=config, train_dataset=dataset, processing_class=processor,
        peft_config=LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0,
                               target_modules=["q_proj", "v_proj"], bias="none", task_type="CAUSAL_LM"),
        callbacks=[JsonlMetrics(log)],
        video_dataset=train_set, cli_args=args, model_type=model_type,
    )
    trainer.add_callback(EpochValidation(trainer, val_set, reward, args, log, model_type, save_dir))
    trainer.model.print_trainable_parameters()
    trainer.train(resume_from_checkpoint=str(args.resume_from) if args.resume_from else None)
    save_named(trainer, save_dir / "final", args)
    log.file.close()
    print(f"[done] {save_dir / 'final'}")

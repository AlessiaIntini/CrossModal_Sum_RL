#!/usr/bin/env python3
"""Direct Preference Optimization of a LoRA adapter on VideoXum chosen/rejected frame selections."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from videoxum import paths
from videoxum.cli import add_schedule_arguments, add_video_arguments, check_schedule_arguments, positive_int
from videoxum.training.dpo import DPO_INIT_NAME, load_dpo_validation_dataset, train_dpo
from videoxum.training.validation import validate_checkpoint


def parse_args():
    # Command line of the DPO stage.
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, default=paths.DEFAULT_MODEL)
    parser.add_argument("--data_json", type=Path, default=paths.PAIRS_JSON,
                        help="training metadata with vsum_onehot_chosen and vsum_onehot_rejected")
    parser.add_argument("--video_dir", type=Path, default=paths.TRAIN_VIDEO_DIR)
    parser.add_argument("--val_json", type=Path, default=paths.VAL_JSON)
    parser.add_argument("--val_video_dir", type=Path, default=paths.VAL_VIDEO_DIR)
    parser.add_argument("--val_ratio", type=float, default=0.05, help="validation split without --val_json")
    parser.add_argument("--saveroot", type=Path, default=paths.OUTPUTS_DIR / "dpo" / "checkpoints")
    parser.add_argument("--resultsroot", type=Path, default=paths.OUTPUTS_DIR / "dpo" / "results")
    parser.add_argument("--resume_from", type=Path, help="checkpoint-N saved by this script")
    parser.add_argument("--init_from", type=Path, help="SFT adapter that initializes policy and reference")
    parser.add_argument("--eval_only", action="store_true", help="only validate --eval_checkpoint")
    parser.add_argument("--eval_checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn_impl", choices=("sdpa", "flash_attention_2"), default="sdpa")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=positive_int, default=3)
    parser.add_argument("--accum_steps", type=positive_int, default=8)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--length-normalize", action=argparse.BooleanOptionalAction, default=True,
                        help="mean token log-probability (TRL sigmoid_norm); --no-length-normalize uses the "
                             "standard summed DPO loss (sigmoid)")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--fps", type=float, choices=(1.0,), default=1.0, help="VideoXum is sampled at 1 fps")
    parser.add_argument("--max_frames", type=int, default=0, help="safety limit; 0 keeps the whole timeline")
    add_video_arguments(parser)
    parser.add_argument("--lora_r", type=positive_int, default=16)
    parser.add_argument("--lora_target_modules", default="q_proj,k_proj,v_proj,o_proj",
                        help="LoRA target modules when starting from the base model")
    parser.add_argument("--save_steps", type=positive_int, default=100)
    parser.add_argument("--eval_samples", type=int, default=200, help="validation videos; <= 0 uses all")
    parser.add_argument("--eval_max_new_tokens", type=positive_int, default=256)
    add_schedule_arguments(parser)
    parser.add_argument("--debug_samples", type=int, default=0, help="if > 0, train on N pairs")
    parser.add_argument("--dataloader_num_workers", type=int, default=2,
                        help="processes decoding videos in parallel to training")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # A resume reads its starting SFT adapter from dpo_init.json.
    if args.init_from and args.resume_from:
        parser.error(f"--init_from and --resume_from are mutually exclusive: resuming reads {DPO_INIT_NAME}")
    if args.init_from and not (args.init_from / "adapter_config.json").is_file():
        parser.error(f"--init_from is not a PEFT adapter: {args.init_from}")
    if args.dataloader_num_workers < 0:
        parser.error("--dataloader_num_workers must be >= 0")
    if args.eval_only != (args.eval_checkpoint is not None):
        parser.error("--eval_only and --eval_checkpoint must be given together")
    check_schedule_arguments(args)
    return args


if __name__ == "__main__":
    cli_args = parse_args()
    # --eval_only validates a saved adapter without training.
    if cli_args.eval_only:
        validate_checkpoint(cli_args, load_dpo_validation_dataset, pairwise=True)
    else:
        train_dpo(cli_args)

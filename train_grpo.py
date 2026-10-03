#!/usr/bin/env python3
"""GRPO of a LoRA adapter on VideoXum: joint frame selection and summary with a composite reward."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from videoxum import paths
from videoxum.cli import add_video_arguments, positive_int
from videoxum.prompts import SUMMARY_STYLES, SUMMARY_WORD_RANGE
from videoxum.rewards import (DEFAULT_REWARD_WEIGHTS, F_BETA_AGGREGATIONS, LENGTH_BUDGET_SIGMA,
                              REFERENCE_COSINE_FLOOR, REFERENCE_COSINE_SCALE, SUMMARY_LENGTH_SIGMA)
from videoxum.training.grpo import LOSS_TYPES, SCALE_REWARDS, train_grpo


def parse_args():
    # Command line of the GRPO stage.
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, default=paths.DEFAULT_MODEL)
    parser.add_argument("--init_from", type=Path,
                        help="SFT/DPO adapter merged into the base model; also the KL reference")
    parser.add_argument("--resume_from", type=Path, help="checkpoint-N saved by this script")
    parser.add_argument("--data_json", type=Path, default=paths.TRAIN_JSON)
    parser.add_argument("--video_dir", type=Path, default=paths.TRAIN_VIDEO_DIR)
    parser.add_argument("--feature_dir", type=Path, default=paths.BLIP_FEATURE_DIR,
                        help="BLIP frame features (<video_id>.npz) used by the reward")
    parser.add_argument("--val_json", type=Path, default=paths.VAL_JSON)
    parser.add_argument("--val_video_dir", type=Path, default=paths.VAL_VIDEO_DIR)
    parser.add_argument("--val_feature_dir", type=Path)
    parser.add_argument("--val_ratio", type=float, default=0.05, help="validation split without --val_json")
    parser.add_argument("--saveroot", type=Path, default=paths.OUTPUTS_DIR / "grpo" / "checkpoints")
    parser.add_argument("--resultsroot", type=Path, default=paths.OUTPUTS_DIR / "grpo" / "results")
    parser.add_argument("--semantic_model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--semantic_device", default="cpu")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn_impl", choices=("sdpa", "flash_attention_2"), default="sdpa")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--epochs", type=positive_int, default=2)
    parser.add_argument("--accum_steps", type=positive_int, default=8, help="videos per optimizer step")
    parser.add_argument("--beta", type=float, default=0.04, help="KL coefficient")
    parser.add_argument("--num_generations", type=positive_int, default=4, help="rollouts per video")
    parser.add_argument("--loss_type", choices=LOSS_TYPES, default="grpo",
                        help="TRL loss aggregation: grpo (mean per sequence); dapo and dr_grpo remove the "
                             "length bias")
    parser.add_argument("--scale_rewards", choices=SCALE_REWARDS, default="group",
                        help="advantage normalization: group standard deviation, or none as in Dr. GRPO")
    parser.add_argument("--reward_f_beta", type=float, default=0.5,
                        help="beta of the F-beta reward (1 = F1; < 1 favors precision)")
    parser.add_argument("--f_beta_aggregation", choices=F_BETA_AGGREGATIONS, default="max",
                        help="aggregation of the F-beta over the 10 annotators")
    # One --w_<component> option per reward weight.
    for name, weight in DEFAULT_REWARD_WEIGHTS.items():
        parser.add_argument(f"--w_{name}", type=float, default=weight,
                            help=f"weight of the {name} reward term (default {weight}; 0 disables it)")
    parser.add_argument("--length_penalty", action=argparse.BooleanOptionalAction, default=False,
                        help="legacy multiplicative frame-budget penalty (use with --w_length_budget 0)")
    parser.add_argument("--length_budget_sigma", type=float, default=LENGTH_BUDGET_SIGMA)
    parser.add_argument("--summary_style", choices=SUMMARY_STYLES, default="reference",
                        help=f"reference: 3-4 sentences ({SUMMARY_WORD_RANGE[0]}-{SUMMARY_WORD_RANGE[1]} words) "
                             "as the VideoXum references; concise: one sentence")
    parser.add_argument("--summary_length_sigma", type=float, default=SUMMARY_LENGTH_SIGMA)
    parser.add_argument("--summary_target_words", type=positive_int, default=None,
                        help="fixed word target of the summary_length term instead of the per-video reference")
    parser.add_argument("--reference_cosine_floor", type=float, default=REFERENCE_COSINE_FLOOR,
                        help="the reference term is clamp((cos - floor) / scale)")
    parser.add_argument("--reference_cosine_scale", type=float, default=REFERENCE_COSINE_SCALE)
    parser.add_argument("--max_completion_length", type=positive_int, default=384)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20,
                        help="sampling top-k; 20 is the generation_config value of Qwen3-VL")
    parser.add_argument("--max_frames", type=int, default=0)
    add_video_arguments(parser)
    parser.add_argument("--lora_r", type=positive_int, default=16)
    parser.add_argument("--save_steps", type=positive_int, default=100)
    parser.add_argument("--train_samples", type=int, default=0, help="training videos; 0 uses all")
    parser.add_argument("--val_samples", "--eval_samples", dest="val_samples", type=int, default=200,
                        help="validation videos; 0 uses all")
    parser.add_argument("--debug_samples", type=int, default=0, help="deprecated alias of --train_samples")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    # Group advantages need at least two rollouts per video.
    if args.num_generations < 2:
        parser.error("--num_generations must be >= 2")
    if args.top_k < 0:
        parser.error("--top_k must be >= 0")
    if args.train_samples < 0 or args.val_samples < 0 or args.debug_samples < 0:
        parser.error("--train_samples and --val_samples must be >= 0")
    if args.train_samples and args.debug_samples:
        parser.error("use either --train_samples or --debug_samples")
    return args


if __name__ == "__main__":
    train_grpo(parse_args())

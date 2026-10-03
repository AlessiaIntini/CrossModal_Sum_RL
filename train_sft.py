#!/usr/bin/env python3
"""Supervised fine-tuning of a LoRA adapter on VideoXum.

Tasks: v2v (frame selection as JSON intervals), v2t (summary only), v2vt (selection and summary
in one JSON object, the output format of GRPO).
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from videoxum import paths
from videoxum.cli import add_schedule_arguments, add_video_arguments, check_schedule_arguments, positive_int
from videoxum.metrics import METRIC_NAMES
from videoxum.prompts import INDEX_PHRASES, SUMMARY_STYLES
from videoxum.training.sft import SFT_TASKS, load_sft_validation_dataset, train_sft
from videoxum.training.validation import validate_checkpoint

TEXT_BEST_METRICS = ("rouge1", "rougeL", "bleu4_sentence")
BEST_METRICS = {"v2v": METRIC_NAMES, "v2t": TEXT_BEST_METRICS, "v2vt": METRIC_NAMES + TEXT_BEST_METRICS + ("v2vt",)}
MAX_NEW_TOKENS = {"v2v": 256, "v2t": 192, "v2vt": 384}


def parse_args():
    # Command line of the SFT stage.
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=SFT_TASKS, default="v2v")
    parser.add_argument("--model", type=Path, default=paths.DEFAULT_MODEL)
    parser.add_argument("--data_json", type=Path, default=paths.PAIRS_JSON,
                        help="training metadata with vsum_onehot_chosen (consensus pairs)")
    parser.add_argument("--video_dir", type=Path, default=paths.TRAIN_VIDEO_DIR)
    parser.add_argument("--val_json", type=Path, default=paths.VAL_JSON)
    parser.add_argument("--val_video_dir", type=Path, default=paths.VAL_VIDEO_DIR)
    parser.add_argument("--val_ratio", type=float, default=0.05, help="validation split without --val_json")
    parser.add_argument("--saveroot", type=Path, default=paths.OUTPUTS_DIR / "sft" / "checkpoints")
    parser.add_argument("--resultsroot", type=Path, default=paths.OUTPUTS_DIR / "sft" / "results")
    parser.add_argument("--resume_from", type=Path, help="checkpoint-N saved by this script")
    parser.add_argument("--eval_only", action="store_true", help="only validate --eval_checkpoint (task v2v)")
    parser.add_argument("--eval_checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn_impl", choices=("sdpa", "flash_attention_2"), default="sdpa")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=positive_int, default=3)
    parser.add_argument("--accum_steps", type=positive_int, default=8)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--fps", type=float, choices=(1.0,), default=1.0, help="VideoXum is sampled at 1 fps")
    parser.add_argument("--max_frames", type=int, default=0, help="safety limit; 0 keeps the whole timeline")
    add_video_arguments(parser)
    parser.add_argument("--lora_r", type=positive_int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", default="q_proj,k_proj,v_proj,o_proj",
                        help="comma-separated LoRA target modules")
    parser.add_argument("--loss_normalization", choices=("token", "sample"), default="token",
                        help="token: every target token of the accumulation window weighs the same (TRL); "
                             "sample: mean per video")
    parser.add_argument("--save_steps", type=positive_int, default=100)
    parser.add_argument("--eval_samples", type=int, default=200, help="validation videos; <= 0 uses all")
    parser.add_argument("--eval_max_new_tokens", type=positive_int, default=None,
                        help="generated tokens in validation; default 256 (v2v), 192 (v2t), 384 (v2vt)")
    add_schedule_arguments(parser, best_metrics=sorted(set(BEST_METRICS["v2vt"])))
    parser.add_argument("--summary_style", choices=SUMMARY_STYLES, default="reference",
                        help="v2t/v2vt: summary style of the prompt, the same of GRPO and inference")
    parser.add_argument("--index_phrase", choices=tuple(INDEX_PHRASES), default="indexed",
                        help="v2vt: wording of the timeline intro (indexed as GRPO, numbered as SFT V2V); "
                             "ignored with --timeline_unit pair")
    parser.add_argument("--video_dependence_samples", type=int, default=0,
                        help="v2v: regenerate N validation videos with replaced frames after training")
    parser.add_argument("--video_dependence_mode", choices=("swap", "blank"), default="swap")
    parser.add_argument("--diversity_check_samples", type=int, default=0,
                        help="v2v: sample N validation videos after training to detect mode collapse")
    parser.add_argument("--diversity_check_k", type=positive_int, default=4)
    parser.add_argument("--diversity_temperature", type=float, default=0.7)
    parser.add_argument("--debug_samples", type=int, default=0, help="if > 0, train on N videos")
    parser.add_argument("--dataloader_num_workers", type=int, default=2,
                        help="processes decoding videos in parallel to training")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # V2T has no selection: checkpoint-best uses ROUGE-L.
    if args.best_metric is None and args.task == "v2t":
        args.best_metric = "rougeL"
    if args.best_metric is not None and args.best_metric not in BEST_METRICS[args.task]:
        parser.error(f"--best_metric {args.best_metric} is not available for task {args.task}")
    if args.eval_max_new_tokens is None:
        args.eval_max_new_tokens = MAX_NEW_TOKENS[args.task]
    # The summary style concerns only the text tasks; --eval_only and the diagnostics of the final
    # adapter exist only for V2V.
    if args.task == "v2v":
        args.summary_style = None
    elif args.eval_only or args.diversity_check_samples or args.video_dependence_samples:
        parser.error("--eval_only, --diversity_check_samples and --video_dependence_samples need task v2v")
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
        validate_checkpoint(cli_args, load_sft_validation_dataset)
    else:
        train_sft(cli_args)

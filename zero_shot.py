#!/usr/bin/env python3
"""Zero-shot baselines on the VideoXum test set with local base models or OpenRouter API models."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from videoxum import paths
from videoxum.cli import nonnegative_int, positive_int
from videoxum.prompts import SUMMARY_STYLES
from videoxum.zero_shot.common import TASKS
from videoxum.zero_shot.openrouter import DEFAULT_API_BASE, IMAGE_DETAILS, INPUT_MODES, REASONING_EFFORTS


def parse_providers(value):
    # Comma-separated OpenRouter providers.
    return [name.strip() for name in value.split(",") if name.strip()]


def parse_args():
    # Command line of the zero-shot baselines; each backend reads only its own option group.
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=("local", "openrouter"), default="local")
    parser.add_argument("--model", default=str(paths.DEFAULT_MODEL),
                        help="local model directory, or OpenRouter id (e.g. google/gemini-3.5-flash)")
    parser.add_argument("--task", choices=TASKS, default="v2v",
                        help="v2v: frame selection; v2t: summary only; v2vt: selection and summary")
    parser.add_argument("--summary_style", choices=SUMMARY_STYLES, default="reference",
                        help="v2t/v2vt: summary style, as in the SFT and GRPO prompts")
    parser.add_argument("--test_json", type=Path, default=paths.TEST_JSON)
    parser.add_argument("--test_video_dir", type=Path, default=paths.TEST_VIDEO_DIR)
    parser.add_argument("--output_dir", type=Path, default=paths.OUTPUTS_DIR / "zero_shot")
    parser.add_argument("--run_name", help="results directory (and API cache) of the run")
    parser.add_argument("--start_index", type=nonnegative_int, default=0)
    parser.add_argument("--num_samples", type=nonnegative_int, default=0, help="test videos; 0 uses all remaining")
    parser.add_argument("--max_frames", type=nonnegative_int, default=0)
    parser.add_argument("--selection_ratio", type=float, default=0.20,
                        help="v2v/v2vt: share of frames suggested in the prompt")
    parser.add_argument("--max_pixels", type=positive_int, default=128 * 32 * 32, help="pixels per frame")
    parser.add_argument("--max_model_frames", type=nonnegative_int, default=None,
                        help="frames shown to InternVL (local, default 64) or as images (openrouter frames, "
                             "default 256, 0 = all); longer videos are sampled uniformly")
    parser.add_argument("--seed", type=int, default=42)

    local = parser.add_argument_group("local backend")
    local.add_argument("--device", default="cuda:0")
    local.add_argument("--attn_impl", choices=("sdpa", "flash_attention_2"), default="sdpa")
    local.add_argument("--min_pixels", type=positive_int, default=4 * 32 * 32)
    local.add_argument("--max_video_pixels", type=positive_int, default=8 * 1024 * 1024,
                       help="pixel budget of the whole video, so that long videos fit in memory")
    local.add_argument("--eval_max_new_tokens", type=positive_int, default=None,
                       help="generated tokens; default 256, 384 for v2vt")

    api = parser.add_argument_group("openrouter backend")
    api.add_argument("--api_base", default=DEFAULT_API_BASE)
    api.add_argument("--provider", type=parse_providers, default=[],
                     help="allowed providers (comma-separated); google-vertex by default with --input video")
    api.add_argument("--input", choices=INPUT_MODES, default="frames",
                     help="frames: JPEG images labeled with their second; video: MP4 at 1 fps")
    api.add_argument("--jpeg_quality", type=positive_int, default=90)
    api.add_argument("--image_detail", choices=IMAGE_DETAILS, default=None,
                     help="frames: image detail (Gemini 3: low ~290 tokens per frame instead of ~1120)")
    api.add_argument("--max_tokens", type=positive_int, default=16384, help="output tokens, reasoning included")
    api.add_argument("--temperature", type=float, default=0.0)
    reasoning = api.add_mutually_exclusive_group()
    reasoning.add_argument("--reasoning_effort", choices=REASONING_EFFORTS, help="thinking level (Gemini 3)")
    reasoning.add_argument("--reasoning_max_tokens", type=positive_int, help="thinking budget (Gemini 2.5)")
    api.add_argument("--num_workers", type=positive_int, default=8, help="parallel API requests")
    api.add_argument("--timeout", type=positive_int, default=600, help="seconds per request")
    api.add_argument("--max_retries", type=nonnegative_int, default=6)
    args = parser.parse_args()
    if not 0 < args.selection_ratio <= 1:
        parser.error("--selection_ratio must be in (0, 1]")
    return args


if __name__ == "__main__":
    cli_args = parse_args()
    # The local backend (transformers and qwen_vl_utils) is imported only when it is used.
    if cli_args.backend == "openrouter":
        from videoxum.zero_shot.openrouter import run
    else:
        from videoxum.zero_shot.local import run
    run(cli_args)

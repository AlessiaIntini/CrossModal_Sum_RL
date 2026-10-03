"""Command-line argument types and option groups shared by the entry points."""

import argparse

from .metrics import METRIC_NAMES
from .timeline import TIMELINE_UNITS
from .video import DEFAULT_MAX_PIXELS, DEFAULT_MIN_PIXELS

TIME_BUDGET_EXIT_CODE = 85


def positive_int(value):
    # argparse type: integers > 0.
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_int(value):
    # argparse type: integers >= 0.
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be an integer >= 0")
    return parsed


def add_video_arguments(parser):
    # Video encoding: temporal markers, timeline unit and per-frame pixel budget. Every option
    # left unset is inherited from the starting checkpoint (see video.resolve_video_encoding).
    parser.add_argument(
        "--video_time_mode", choices=("auto", "legacy", "seconds"), default="auto",
        help="Qwen3-VL <t seconds> markers of the decoded frames. seconds: real seconds; "
             "legacy: half the real time, as in the first checkpoints. auto: inherit from the "
             "starting checkpoint (legacy without video_encoding.json), seconds from the base "
             "model; always legacy with --timeline_unit pair.",
    )
    parser.add_argument(
        "--timeline_unit", choices=("auto",) + TIMELINE_UNITS, default="auto",
        help="indices predicted by the model. frame: one per frame at 1 fps; pair: one per pair "
             "of frames, as the temporal positions of Qwen3-VL. auto: inherit from the starting "
             "checkpoint, frame from the base model.",
    )
    parser.add_argument(
        "--max_pixels", type=positive_int, default=None,
        help=f"maximum pixels per frame; inherited from the starting checkpoint, "
             f"otherwise {DEFAULT_MAX_PIXELS}",
    )
    parser.add_argument(
        "--min_pixels", type=positive_int, default=None,
        help=f"minimum pixels per frame (<= --max_pixels); inherited from the starting "
             f"checkpoint, otherwise {DEFAULT_MIN_PIXELS}",
    )


def add_schedule_arguments(parser, best_metrics=METRIC_NAMES):
    # Periodic validation and checkpoint-best selection.
    parser.add_argument(
        "--eval_steps", type=int, default=0,
        help="if > 0, validate every N optimizer steps on --eval_steps_samples videos and "
             "update checkpoint-best; 0 validates only at the end of each epoch",
    )
    parser.add_argument("--eval_steps_samples", type=positive_int, default=200)
    parser.add_argument(
        "--best_metric", choices=best_metrics, default=None,
        help="validation metric for checkpoint-best; default kappa with --timeline_unit pair, "
             "f1_mean otherwise",
    )
    # SLURM time budget: the launchers requeue the job when it exits with TIME_BUDGET_EXIT_CODE.
    parser.add_argument(
        "--time_budget_hours", type=float, default=0.0,
        help=f"if > 0, save a resumable checkpoint after this time and exit with code "
             f"{TIME_BUDGET_EXIT_CODE}",
    )


def check_schedule_arguments(args):
    # checkpoint-best compares validations on the first --eval_steps_samples videos, so the
    # end-of-epoch validation must cover at least those videos.
    if args.eval_steps and 0 < args.eval_samples < args.eval_steps_samples:
        raise SystemExit("--eval_samples must be >= --eval_steps_samples")

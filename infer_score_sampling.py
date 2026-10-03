#!/usr/bin/env python3
"""V2VT test inference with per-frame scores estimated from sampled responses.

The official Kendall/Spearman rank frames by a continuous score, while the model outputs a
binary selection. Next to the greedy response, K responses are sampled as GRPO rollouts and the
score of a frame is the share of valid responses that select it.
"""

import argparse
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from videoxum.cli import positive_int
from videoxum.inference import add_inference_arguments, run_score_sampling


def parse_args():
    # Command line of the score-sampling inference, which exists only for the joint task.
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_inference_arguments(parser)
    parser.set_defaults(run_name="grpo-score-sampling-test")
    parser.add_argument("--score_samples", type=positive_int, default=8, help="sampled responses per video")
    parser.add_argument("--score_temperature", type=float, default=1.0)
    parser.add_argument("--score_top_p", type=float, default=0.95)
    args = parser.parse_args()
    args.task = "v2vt"
    if args.index_phrase is None:
        args.index_phrase = "indexed"
    return args


if __name__ == "__main__":
    run_score_sampling(parse_args())

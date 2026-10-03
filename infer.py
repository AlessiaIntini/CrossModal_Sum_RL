#!/usr/bin/env python3
"""Test-set inference of a trained adapter on VideoXum (v2v, v2t or v2vt)."""

import argparse
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from videoxum.inference import add_inference_arguments, run_joint_inference, run_selection_inference

TASKS = ("v2v", "v2t", "v2vt")


def parse_args():
    # Command line of the test-set inference.
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=TASKS, default="v2v",
                        help="v2v: JSON intervals; v2t: summary only; v2vt: selection and summary")
    add_inference_arguments(parser)
    parser.add_argument("--postprocess", choices=("off", "normalize", "recover"), default="normalize",
                        help="v2v: off uses the strict parser; normalize fixes overlapping or out-of-range "
                             "intervals of valid JSON; recover also salvages complete pairs of truncated JSON")
    args = parser.parse_args()
    # Default timeline wording: the one of the SFT V2V prompt for v2v, the one of the joint prompt
    # otherwise.
    if args.index_phrase is None:
        args.index_phrase = "numbered" if args.task == "v2v" else "indexed"
    return args


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.task == "v2v":
        run_selection_inference(cli_args)
    else:
        run_joint_inference(cli_args)

#!/usr/bin/env python3
"""Inferenza di un adapter PEFT (SFT, DPO o GRPO) su VideoXum."""

import argparse
import json
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoProcessor

from sft_train_videoxum import (
    PROJECT_ROOT,
    VIDEOXUM_ROOT,
    VideoXumSFTDataset,
    decode_video_frames,
    format_response,
    generate_interval_response,
    load_records,
    parse_interval_response,
    videoxum_metrics,
    write_metric,
)
from vl_model_loader import load_qwen_vl_model


DEFAULT_MODEL = PROJECT_ROOT / "models" / "Qwen2.5-VL-7B-Instruct"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs" / "sft" / "checkpoints"
    / "Qwen2.5-VL-7B-Instruct" / "checkpoint-best"
)
DEFAULT_TEST_JSON = VIDEOXUM_ROOT / "metadata" / "test_videoxum.json"
DEFAULT_TEST_VIDEO_DIR = VIDEOXUM_ROOT / "test"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "inference"


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("deve essere un intero positivo")
    return parsed


def nonnegative_int(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("deve essere un intero >= 0")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inferenza di un checkpoint PEFT SFT/DPO/GRPO sul test set VideoXum"
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--test_json", type=Path, default=DEFAULT_TEST_JSON)
    parser.add_argument("--test_video_dir", type=Path, default=DEFAULT_TEST_VIDEO_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--run_name",
        help="etichetta opzionale della run; usata nella directory dei risultati",
    )
    parser.add_argument(
        "--start_index", type=nonnegative_int, default=0,
        help="indice iniziale incluso nel file di test",
    )
    parser.add_argument(
        "--num_samples", type=nonnegative_int, default=0,
        help="numero di video da elaborare; 0 usa tutti quelli rimanenti",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--attn_impl", choices=("sdpa", "flash_attention_2"), default="sdpa"
    )
    parser.add_argument("--fps", type=float, choices=(1.0,), default=1.0)
    parser.add_argument("--max_frames", type=nonnegative_int, default=0)
    parser.add_argument("--max_pixels", type=positive_int, default=128 * 28 * 28)
    parser.add_argument("--min_pixels", type=positive_int, default=4 * 28 * 28)
    parser.add_argument("--eval_max_new_tokens", type=positive_int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_args(args):
    for path, description in (
        (args.model, "modello base"),
        (args.checkpoint, "checkpoint PEFT"),
        (args.test_video_dir, "directory video test"),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"{description} non trovato: {path}")
    if not args.test_json.is_file():
        raise FileNotFoundError(f"metadata di test non trovati: {args.test_json}")
    if not (args.checkpoint / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"adapter_config.json non trovato nel checkpoint: {args.checkpoint}"
        )
    if args.min_pixels > args.max_pixels:
        raise ValueError(
            f"--min_pixels ({args.min_pixels}) supera --max_pixels "
            f"({args.max_pixels})"
        )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA non disponibile: eseguire su un nodo GPU")


def select_records(records, start_index, num_samples):
    if start_index >= len(records):
        raise ValueError(
            f"--start_index={start_index} fuori dal test set ({len(records)} video)"
        )
    stop_index = len(records) if num_samples == 0 else min(
        len(records), start_index + num_samples
    )
    return records[start_index:stop_index], stop_index


def default_run_name(checkpoint):
    """Nome leggibile e non ambiguo per SFT/DPO/GRPO senza imporre convenzioni."""
    try:
        training_stage = checkpoint.parents[2].name
    except IndexError:
        training_stage = "adapter"
    raw_name = f"{training_stage}-{checkpoint.name}"
    return re.sub(r"[^A-Za-z0-9._-]+", "-", raw_name).strip("-")


@torch.inference_mode()
def run_inference(args):
    validate_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    all_records = load_records(args.test_json, require_chosen=False)
    records, stop_index = select_records(
        all_records, args.start_index, args.num_samples
    )
    dataset = VideoXumSFTDataset(records, args.test_video_dir)

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"[data] test totale={len(all_records)}, subset=[{args.start_index}, {stop_index})")
    print(f"[model] modello base={args.model}")
    print(f"[model] checkpoint PEFT={args.checkpoint}")
    print(f"[model] attention={args.attn_impl}, dtype={dtype}")

    base_model, model_type = load_qwen_vl_model(
        args.model, torch_dtype=dtype, attn_implementation=args.attn_impl
    )
    base_model = base_model.to(device)
    print(f"[model] architecture={model_type}")
    model = PeftModel.from_pretrained(
        base_model,
        args.checkpoint,
        is_trainable=False,
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(
        args.checkpoint, local_files_only=True
    )
    model.eval()
    model.config.use_cache = True

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_name = args.run_name or default_run_name(args.checkpoint)
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / (
        f"predictions-test-{args.start_index}-{stop_index}-{run_id}.jsonl"
    )
    metrics_path = output_dir / (
        f"metrics-test-{args.start_index}-{stop_index}-{run_id}.jsonl"
    )
    metric_values = {
        name: [] for name in ("f1_max", "f1_mean", "kendall", "spearman")
    }
    invalid_predictions = 0
    processing_errors = 0

    with (
        predictions_path.open("w", encoding="utf-8", buffering=1) as predictions_file,
        metrics_path.open("w", encoding="utf-8", buffering=1) as metrics_file,
    ):
        write_metric(
            predictions_file,
            run_id=run_id,
            record_type="config",
            split="test",
            run_name=run_name,
            model=args.model,
            checkpoint=args.checkpoint,
            test_json=args.test_json,
            test_video_dir=args.test_video_dir,
            start_index=args.start_index,
            stop_index=stop_index,
            samples=len(dataset),
            arguments=vars(args),
        )
        write_metric(
            metrics_file,
            run_id=run_id,
            record_type="config",
            split="test",
            run_name=run_name,
            checkpoint=args.checkpoint,
            test_json=args.test_json,
            start_index=args.start_index,
            stop_index=stop_index,
            samples=len(dataset),
            metric_protocol="VideoXum aggregation; binary generated saliency",
        )

        for local_index in tqdm(range(len(dataset)), desc="test", unit="video"):
            dataset_index = args.start_index + local_index
            record = records[local_index]
            sample = dataset[local_index]
            response = None
            try:
                frames = decode_video_frames(sample, args)
                response = generate_interval_response(
                    model, processor, sample, args, device, frames
                )
                parse_error = None
                try:
                    prediction = parse_interval_response(
                        response, sample["sampled_frames"]
                    )
                except (json.JSONDecodeError, ValueError) as error:
                    prediction = [0] * sample["sampled_frames"]
                    parse_error = str(error)
                    invalid_predictions += 1

                sample_metrics = videoxum_metrics(
                    prediction, sample["references"]
                )
                for name, value in sample_metrics.items():
                    metric_values[name].append(value)
                write_metric(
                    predictions_file,
                    run_id=run_id,
                    record_type="sample",
                    split="test",
                    sample_position=local_index + 1,
                    dataset_index=dataset_index,
                    video_id=record["video_id"],
                    sampled_frames=sample["sampled_frames"],
                    generated_response=response,
                    prediction_valid=parse_error is None,
                    parse_error=parse_error,
                    predicted_intervals=json.loads(format_response(prediction)),
                    selected_frame_indices=[
                        index for index, selected in enumerate(prediction) if selected
                    ],
                    selected_frames=sum(prediction),
                    selection_ratio=sum(prediction) / len(prediction),
                )
                write_metric(
                    metrics_file,
                    run_id=run_id,
                    record_type="sample",
                    split="test",
                    sample_position=local_index + 1,
                    dataset_index=dataset_index,
                    video_id=record["video_id"],
                    prediction_valid=parse_error is None,
                    **sample_metrics,
                )
            except Exception as error:
                processing_errors += 1
                write_metric(
                    predictions_file,
                    run_id=run_id,
                    record_type="sample_error",
                    split="test",
                    sample_position=local_index + 1,
                    dataset_index=dataset_index,
                    video_id=record["video_id"],
                    sampled_frames=sample["sampled_frames"],
                    generated_response=response,
                    error_type=type(error).__name__,
                    error=str(error),
                )
                write_metric(
                    metrics_file,
                    run_id=run_id,
                    record_type="sample_error",
                    split="test",
                    sample_position=local_index + 1,
                    dataset_index=dataset_index,
                    video_id=record["video_id"],
                    error_type=type(error).__name__,
                    error=str(error),
                )
                print(
                    f"[warning] test[{dataset_index}] video={record['video_id']}: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
            finally:
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        summary_metrics = {
            name: (sum(values) / len(values) if values else None)
            for name, values in metric_values.items()
        }
        write_metric(
            metrics_file,
            run_id=run_id,
            record_type="summary",
            split="test",
            samples_requested=len(dataset),
            samples_evaluated=len(metric_values["f1_mean"]),
            invalid_predictions=invalid_predictions,
            processing_errors=processing_errors,
            metric_protocol="VideoXum aggregation; binary generated saliency",
            **summary_metrics,
        )

    print(
        "[test] "
        + " ".join(
            f"{name.upper()}={value:.4f}"
            for name, value in summary_metrics.items()
            if value is not None
        )
        + f" valid={len(dataset) - invalid_predictions - processing_errors}/{len(dataset)}"
    )
    print(f"[done] predizioni: {predictions_path}")
    print(f"[done] metriche: {metrics_path}")


if __name__ == "__main__":
    run_inference(parse_args())

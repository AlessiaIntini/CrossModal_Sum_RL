#!/usr/bin/env python3
"""Inferenza di un adapter PEFT SFT o DPO su VideoXum.

La modalità selezionata riusa prompt e preprocessing del relativo training,
evitando di valutare un adapter DPO con il protocollo SFT.
"""

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

from evaluation.chance_baselines import load_positional_prior, matched_baselines
from sft_train_videoxum import (
    METRIC_NAMES,
    PROJECT_ROOT,
    VIDEOXUM_ROOT,
    VideoXumSFTDataset,
    add_pixel_arguments,
    decode_video_frames as decode_sft_video_frames,
    add_video_time_argument,
    format_response,
    generate_interval_response as generate_sft_interval_response,
    load_records,
    parse_interval_response,
    resolve_video_time_mode,
    segments_to_frames,
    videoxum_metrics,
    write_metric,
)
from train_dpo import (
    VideoXumDPODataset,
    decode_video_frames as decode_dpo_video_frames,
    generate_interval_response as generate_dpo_interval_response,
)
from vl_model_loader import load_qwen_vl_model

BASELINE_NAMES = ("random", "positional")


DEFAULT_MODEL = PROJECT_ROOT / "models" / "Qwen2.5-VL-7B-Instruct"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs" / "sft" / "checkpoints"
    / "Qwen2.5-VL-7B-Instruct" / "checkpoint-best"
)
DEFAULT_TEST_JSON = VIDEOXUM_ROOT / "metadata" / "test_videoxum.json"
DEFAULT_TEST_VIDEO_DIR = VIDEOXUM_ROOT / "test"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "inference"

_INTERVAL_PATTERN = re.compile(
    r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]"
)


def extract_intervals(response, allow_truncated=False):
    """Estrae gli intervalli dalla risposta.

    Con allow_truncated=False richiede un array JSON valido.
    Con allow_truncated=True recupera tutte le coppie [start,end] complete
    presenti anche in una risposta troncata.
    """
    array_start = response.find("[")
    if array_start < 0:
        raise ValueError("nessun array trovato nella risposta")

    try:
        intervals, _ = json.JSONDecoder().raw_decode(response[array_start:])
    except json.JSONDecodeError:
        if not allow_truncated:
            raise

        intervals = [
            [int(start), int(end)]
            for start, end in _INTERVAL_PATTERN.findall(response[array_start:])
        ]
        if not intervals:
            raise ValueError(
                "nessun intervallo completo recuperabile dalla risposta troncata"
            )

    if not isinstance(intervals, list):
        raise ValueError("la risposta non contiene un array di intervalli")

    return intervals


def normalize_intervals(intervals, frame_count):
    """Valida, ordina, limita alla timeline e fonde gli intervalli."""
    if frame_count <= 0:
        raise ValueError("frame_count deve essere positivo")

    cleaned = []
    actions = set()

    for interval in intervals:
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            actions.add("dropped_malformed_interval")
            continue

        start, end = interval

        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
        ):
            actions.add("dropped_non_integer_interval")
            continue

        if start > end:
            start, end = end, start
            actions.add("swapped_bounds")

        clipped_start = max(0, min(frame_count - 1, start))
        clipped_end = max(0, min(frame_count - 1, end))

        if (clipped_start, clipped_end) != (start, end):
            actions.add("clipped_to_timeline")

        cleaned.append([clipped_start, clipped_end])

    if not cleaned:
        raise ValueError("nessun intervallo valido dopo la normalizzazione")

    if cleaned != sorted(cleaned):
        actions.add("sorted_intervals")
    cleaned.sort(key=lambda interval: (interval[0], interval[1]))

    merged = []
    for start, end in cleaned:
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
            actions.add("merged_overlapping_or_adjacent")

    prediction = [0] * frame_count
    for start, end in merged:
        for index in range(start, end + 1):
            prediction[index] = 1

    return prediction, merged, sorted(actions)


def postprocess_interval_response(response, frame_count, mode):
    """Restituisce predizione normalizzata e informazioni sul recovery."""
    if mode == "off":
        prediction = parse_interval_response(response, frame_count)
        intervals = json.loads(format_response(prediction))
        return prediction, intervals, []

    allow_truncated = mode == "recover"
    intervals = extract_intervals(
        response,
        allow_truncated=allow_truncated,
    )
    prediction, normalized_intervals, actions = normalize_intervals(
        intervals,
        frame_count,
    )

    if allow_truncated:
        try:
            extract_intervals(response, allow_truncated=False)
        except (json.JSONDecodeError, ValueError):
            actions = sorted(set(actions) | {"recovered_truncated_json"})

    return prediction, normalized_intervals, actions
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
        description="Inferenza coerente di un checkpoint PEFT SFT o DPO su VideoXum"
    )
    parser.add_argument(
        "--mode",
        choices=("sft", "dpo"),
        default="sft",
        help=(
            "seleziona prompt e preprocessing del training corrispondente "
            "(default: sft)"
        ),
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
    add_video_time_argument(parser)
    parser.add_argument("--max_frames", type=nonnegative_int, default=0)
    add_pixel_arguments(parser)
    parser.add_argument(
        "--prior_json", type=Path,
        help="metadata da cui stimare la baseline posizionale (default: "
             "train_videoxum.json accanto a --test_json, se esiste)",
    )
    parser.add_argument("--eval_max_new_tokens", type=positive_int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
    "--postprocess",
    choices=("off", "normalize", "recover"),
    default="normalize",
    help=(
        "off usa il parser stretto; normalize corregge JSON validi con "
        "intervalli sovrapposti/fuori timeline; recover tenta anche di "
        "recuperare coppie complete da JSON troncati (default: normalize)"
    ),
)
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


def default_run_name(checkpoint, mode):
    """Nome leggibile che include esplicitamente la modalita' di inferenza."""
    try:
        training_stage = checkpoint.parents[2].name
    except IndexError:
        training_stage = "adapter"
    raw_name = f"{mode}-{training_stage}-{checkpoint.name}"
    return re.sub(r"[^A-Za-z0-9._-]+", "-", raw_name).strip("-")


def select_inference_backend(mode, records, video_dir, timeline_unit="frame"):
    """Restituisce dataset e funzioni dello stesso protocollo usato in training."""
    if mode == "dpo":
        return (
            VideoXumDPODataset(records, video_dir, timeline_unit),
            decode_dpo_video_frames,
            generate_dpo_interval_response,
        )
    return (
        VideoXumSFTDataset(records, video_dir, timeline_unit),
        decode_sft_video_frames,
        generate_sft_interval_response,
    )


@torch.inference_mode()
def run_inference(args):
    validate_args(args)
    # Un checkpoint si valuta con l'asse temporale con cui e' stato addestrato;
    # quelli senza video_encoding.json restano legacy, quindi identici a prima.
    resolve_video_time_mode(args, args.checkpoint, strict=False)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    all_records = load_records(args.test_json, require_chosen=False)
    records, stop_index = select_records(
        all_records, args.start_index, args.num_samples
    )
    dataset, decode_frames, generate_response = select_inference_backend(
        args.mode, records, args.test_video_dir, args.timeline_unit
    )
    prior_json = args.prior_json or args.test_json.parent / "train_videoxum.json"
    positional_prior = None
    if prior_json.is_file():
        positional_prior = load_positional_prior(prior_json)
        print(f"[baseline] prior posizionale da {prior_json}")
    else:
        print(f"[baseline] {prior_json} assente: baseline posizionale disattivata")
    baseline_rng = random.Random(args.seed)

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"[data] test totale={len(all_records)}, subset=[{args.start_index}, {stop_index})")
    print(f"[model] modello base={args.model}")
    print(f"[model] checkpoint PEFT={args.checkpoint}")
    print(f"[model] max pixels={args.max_pixels}")
    print(f"[model] min pixels={args.min_pixels}")
    print(f"[inference] mode={args.mode}, prompt/preprocessing={args.mode}")
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
    run_name = args.run_name or default_run_name(args.checkpoint, args.mode)
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / (
        f"predictions-test-{args.start_index}-{stop_index}-{run_id}.jsonl"
    )
    metrics_path = output_dir / (
        f"metrics-test-{args.start_index}-{stop_index}-{run_id}.jsonl"
    )
    metric_names = METRIC_NAMES

    raw_metric_values = {
    name: [] for name in metric_names
    }
    normalized_metric_values = {
    name: [] for name in metric_names
    }
    # Baseline con lo stesso numero di frame della predizione normalizzata.
    baseline_metric_values = {
        baseline: {name: [] for name in metric_names} for baseline in BASELINE_NAMES
    }

    invalid_predictions = 0
    recovered_predictions = 0
    unrecoverable_predictions = 0
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
            mode=args.mode,
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
            mode=args.mode,
            run_name=run_name,
            checkpoint=args.checkpoint,
            test_json=args.test_json,
            start_index=args.start_index,
            stop_index=stop_index,
            samples=len(dataset),
            video_time_mode=args.video_time_mode,
            timeline_unit=args.timeline_unit,
            max_pixels=args.max_pixels,
            min_pixels=args.min_pixels,
            metric_protocol="VideoXum aggregation; binary generated saliency",
        )

        for local_index in tqdm(range(len(dataset)), desc="test", unit="video"):
            dataset_index = args.start_index + local_index
            record = records[local_index]
            sample = dataset[local_index]
            response = None
            try:
                frames = decode_frames(sample, args)
                response = generate_response(
                    model, processor, sample, args, device, frames, model_type
                )
                frame_count = sample["sampled_frames"]
                # Il modello risponde sulla propria timeline (frame o coppie);
                # le metriche si calcolano sempre sugli N frame originali.
                timeline_count = sample["timeline_count"]

                # Valutazione raw: comportamento originale.
                raw_parse_error = None
                try:
                    raw_prediction = segments_to_frames(
                        parse_interval_response(response, timeline_count),
                        frame_count, args.timeline_unit,
                    )
                    raw_valid = True
                except (json.JSONDecodeError, ValueError) as error:
                    raw_prediction = [0] * frame_count
                    raw_valid = False
                    raw_parse_error = str(error)
                    invalid_predictions += 1

                # Post-processing separato.
                postprocess_error = None
                repair_actions = []
                try:
                    timeline_prediction, normalized_intervals, repair_actions = (
                        postprocess_interval_response(
                            response,
                            timeline_count,
                            args.postprocess,
                        )
                    )
                    normalized_prediction = segments_to_frames(
                        timeline_prediction, frame_count, args.timeline_unit
                    )
                    normalized_valid = True
                except (json.JSONDecodeError, ValueError) as error:
                    normalized_prediction = [0] * frame_count
                    normalized_intervals = []
                    normalized_valid = False
                    postprocess_error = str(error)
                    unrecoverable_predictions += 1

                prediction_recovered = not raw_valid and normalized_valid
                if prediction_recovered:
                    recovered_predictions += 1

                raw_metrics = videoxum_metrics(
                    raw_prediction,
                    sample["references"],
                )
                normalized_metrics = videoxum_metrics(
                    normalized_prediction,
                    sample["references"],
                )

                baseline_metrics = matched_baselines(
                    normalized_prediction, sample["references"], videoxum_metrics,
                    prior=positional_prior, rng=baseline_rng,
                )

                for name in metric_names:
                    raw_metric_values[name].append(raw_metrics[name])
                    normalized_metric_values[name].append(normalized_metrics[name])
                    for baseline, values in baseline_metrics.items():
                        baseline_metric_values[baseline][name].append(values[name])
                write_metric(
                    predictions_file,
                    run_id=run_id,
                    record_type="sample",
                    split="test",
                    sample_position=local_index + 1,
                    dataset_index=dataset_index,
                    video_id=record["video_id"],
                    sampled_frames=sample["sampled_frames"],
                    timeline_unit=args.timeline_unit,
                    timeline_count=timeline_count,
                    generated_response=response,
                    prediction_valid=raw_valid,
                    parse_error=raw_parse_error,
                    normalized_valid=normalized_valid,
                    postprocess_error=postprocess_error,
                    postprocess_mode=args.postprocess,
                    prediction_recovered=prediction_recovered,
                    repair_actions=repair_actions,
                    predicted_intervals=normalized_intervals,
                    selected_frame_indices=[
                        index
                        for index, selected in enumerate(normalized_prediction)
                    if selected
                    ],
                    selected_frames=sum(normalized_prediction),
                    selection_ratio=sum(normalized_prediction) / frame_count,
                    raw_selected_frames=sum(raw_prediction),
                    raw_selection_ratio=sum(raw_prediction) / frame_count,
                )
                write_metric(
                    metrics_file,
                    run_id=run_id,
                    record_type="sample",
                    split="test",
                    sample_position=local_index + 1,
                    dataset_index=dataset_index,
                    video_id=record["video_id"],
                    postprocess_mode=args.postprocess,
                    prediction_valid=raw_valid,
                    normalized_valid=normalized_valid,
                    prediction_recovered=prediction_recovered,
                    selection_ratio=sum(normalized_prediction) / frame_count,
                    **raw_metrics,
                    **{
                        f"normalized_{name}": value
                        for name, value in normalized_metrics.items()
                    },
                    **{
                        f"baseline_{baseline}_{name}": value
                        for baseline, values in baseline_metrics.items()
                        for name, value in values.items()
                    },
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

        raw_summary_metrics = {
            name: sum(values) / len(values) if values else None
            for name, values in raw_metric_values.items()
            }
        normalized_summary_metrics = {
            name: sum(values) / len(values) if values else None
            for name, values in normalized_metric_values.items()
        }
        baseline_summary_metrics = {
            f"baseline_{baseline}_{name}": sum(values) / len(values)
            for baseline, metrics in baseline_metric_values.items()
            for name, values in metrics.items()
            if values
        }
        write_metric(
    metrics_file,
    run_id=run_id,
    record_type="summary",
    split="test",
    samples_requested=len(dataset),
    samples_evaluated=len(raw_metric_values["f1_mean"]),
    invalid_predictions=invalid_predictions,
    recovered_predictions=recovered_predictions,
    unrecoverable_predictions=unrecoverable_predictions,
    processing_errors=processing_errors,
    postprocess_mode=args.postprocess,
    metric_protocol="VideoXum aggregation; binary generated saliency",
    timeline_unit=args.timeline_unit,
    **raw_summary_metrics,
    **{
        f"normalized_{name}": value
        for name, value in normalized_summary_metrics.items()
    },
    **baseline_summary_metrics,
)
    print(
        "[test] "
        + " ".join(
            f"{name.upper()}={value:.4f}"
            for name, value in raw_summary_metrics.items()
            if value is not None
        )
        + f" valid={len(dataset) - invalid_predictions - processing_errors}/{len(dataset)}"
    )
    for baseline in BASELINE_NAMES:
        kappa = baseline_summary_metrics.get(f"baseline_{baseline}_kappa")
        lift = baseline_summary_metrics.get(f"baseline_{baseline}_lift")
        if kappa is not None:
            print(
                f"[baseline {baseline}, stessi frame] KAPPA={kappa:.4f} LIFT={lift:.4f} "
                f"F1_MEAN={baseline_summary_metrics[f'baseline_{baseline}_f1_mean']:.4f}"
            )
    print(f"[done] predizioni: {predictions_path}")
    print(f"[done] metriche: {metrics_path}")


if __name__ == "__main__":
    run_inference(parse_args())

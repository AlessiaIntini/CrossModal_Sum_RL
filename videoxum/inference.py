"""Test-set inference of trained adapters: V2V selections, V2VT selections + summaries, V2T summaries.

Every video gets a greedy response with the prompt and video encoding of the checkpoint.
Selection metrics are reported next to random and positional baselines with the same number
of frames.
"""

import json
import random
import re
from pathlib import Path

import torch
from tqdm.auto import tqdm

from . import paths
from .baselines import BASELINE_NAMES, load_positional_prior, matched_baselines
from .cli import add_video_arguments, nonnegative_int, positive_int
from .data import SelectionDataset, load_joint_records, load_selection_records
from .inputs import encode_prompt, generate_completions, generate_response
from .jsonl import mean_or_none, new_run_id, write_record
from .metrics import METRIC_NAMES, compute_f1, compute_kendall, compute_spearman, videoxum_metrics
from .models import load_checkpoint_model
from .parsing import (CompletionValidationError, parse_interval_response, parse_joint_response,
                      parse_summary_response, postprocess_interval_response)
from .prompts import INDEX_PHRASES, SUMMARY_STYLES, SYSTEM_PROMPTS, build_joint_prompt, build_summary_prompt
from .timeline import segments_to_frames, timeline_length
from .video import decode_frames, resolve_video, resolve_video_encoding

SELECTION_PROTOCOL = "VideoXum aggregation; binary generated saliency"
PARSE_ERRORS = (CompletionValidationError, ValueError)


def add_inference_arguments(parser):
    # Checkpoint, test data, output and generation options shared by the test-set scripts.
    parser.add_argument("--model", type=Path, default=paths.DEFAULT_MODEL, help="base model of the adapter")
    parser.add_argument("--checkpoint", type=Path, required=True, help="trained PEFT adapter")
    parser.add_argument("--init_from", type=Path,
                        help="adapter merged before a GRPO adapter; read from grpo_init.json if omitted")
    parser.add_argument("--test_json", type=Path, default=paths.TEST_JSON)
    parser.add_argument("--test_video_dir", type=Path, default=paths.TEST_VIDEO_DIR)
    parser.add_argument("--output_dir", type=Path, default=paths.OUTPUTS_DIR / "inference")
    parser.add_argument("--run_name", help="results subdirectory; default <task>-<training run>-<checkpoint>")
    parser.add_argument("--start_index", type=nonnegative_int, default=0, help="first test video (inclusive)")
    parser.add_argument("--num_samples", type=nonnegative_int, default=0, help="test videos; 0 uses all remaining")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn_impl", choices=("sdpa", "flash_attention_2"), default="sdpa")
    parser.add_argument("--fps", type=float, choices=(1.0,), default=1.0)
    parser.add_argument("--max_frames", type=nonnegative_int, default=0)
    # The video encoding is inherited from the checkpoint unless given explicitly.
    add_video_arguments(parser)
    parser.add_argument("--index_phrase", choices=tuple(INDEX_PHRASES), default=None,
                        help="timeline wording used in training: numbered (SFT V2V, default for v2v) or indexed "
                             "(DPO and GRPO, default for v2vt); identical with --timeline_unit pair")
    parser.add_argument("--summary_style", choices=SUMMARY_STYLES, default="reference",
                        help="v2t/v2vt: summary style used in training (concise for GRPO runs before 2026-09-17)")
    parser.add_argument("--summary_word_range", type=positive_int, nargs=2, metavar=("LOW", "HIGH"),
                        help="v2t/v2vt: word range asked in the prompt instead of the training one")
    parser.add_argument("--prior_json", type=Path,
                        help="metadata of the positional baseline (default: train_videoxum.json next to --test_json)")
    parser.add_argument("--eval_max_new_tokens", type=positive_int, default=256, help="v2v: generated tokens")
    parser.add_argument("--max_completion_length", type=positive_int, default=384,
                        help="v2t/v2vt: generated tokens")
    parser.add_argument("--seed", type=int, default=42)


def check_paths(args):
    # Base model, adapter checkpoint, test videos and metadata must exist before loading anything.
    for path, description in ((args.model, "base model"), (args.checkpoint, "PEFT checkpoint"),
                              (args.test_video_dir, "test video directory")):
        if not path.is_dir():
            raise FileNotFoundError(f"{description} not found: {path}")
    if not args.test_json.is_file():
        raise FileNotFoundError(f"test metadata not found: {args.test_json}")
    if not (args.checkpoint / "adapter_config.json").is_file():
        raise FileNotFoundError(f"adapter_config.json not found in {args.checkpoint}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available: run on a GPU node")


def select_range(records, start_index, num_samples):
    # records[start_index:stop]; num_samples=0 means all the remaining videos.
    if start_index >= len(records):
        raise ValueError(f"--start_index={start_index} is outside the test set ({len(records)} videos)")
    stop = len(records) if num_samples == 0 else min(len(records), start_index + num_samples)
    return records[start_index:stop], stop


def default_run_name(checkpoint, task):
    # <task>-<training run>-<checkpoint>, e.g. v2v-sft_consensus_pairs_8k-checkpoint-epoch1, where
    # the training run is the directory above checkpoints/<model>/.
    try:
        stage = checkpoint.parents[2].name
    except IndexError:
        stage = "adapter"
    return re.sub(r"[^A-Za-z0-9._-]+", "-", f"{task}-{stage}-{checkpoint.name}").strip("-")


def open_outputs(args, run_name, stop):
    # Predictions and metrics of a run go to two JSONL files named after the test range and the run
    # id.
    run_id = new_run_id()
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"test-{args.start_index}-{stop}-{run_id}.jsonl"
    return run_id, output_dir / f"predictions-{suffix}", output_dir / f"metrics-{suffix}"


def load_prior(args, verbose=False):
    # Positional prior from --prior_json, or from train_videoxum.json next to the test metadata;
    # without it the positional baseline is disabled.
    prior_json = args.prior_json or args.test_json.parent / "train_videoxum.json"
    if prior_json.is_file():
        if verbose:
            print(f"[baseline] positional prior from {prior_json}")
        return load_positional_prior(prior_json)
    if verbose:
        print(f"[baseline] {prior_json} not found: positional baseline disabled")
    return None


def mean_values(values):
    # Mean of every list of a dict of lists (None for empty lists).
    return {name: mean_or_none(items) for name, items in values.items()}


@torch.inference_mode()
def run_selection_inference(args):
    # V2V inference. The checkpoint encoding is inherited, with only a warning if an explicit option
    # differs (strict=False).
    check_paths(args)
    resolve_video_encoding(args, args.checkpoint, strict=False)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    all_records = load_selection_records(args.test_json, with_target=False)
    records, stop = select_range(all_records, args.start_index, args.num_samples)
    dataset = SelectionDataset(records, args.test_video_dir, args.timeline_unit, args.index_phrase)
    prior = load_prior(args, verbose=True)
    baseline_rng = random.Random(args.seed)
    device = torch.device(args.device)
    print(f"[data] test={len(all_records)}, subset=[{args.start_index}, {stop})")
    print(f"[inference] task=v2v index_phrase={args.index_phrase} max_pixels={args.max_pixels} "
          f"min_pixels={args.min_pixels} attention={args.attn_impl}")
    model, processor, model_type, _ = load_checkpoint_model(args.model, args.checkpoint, device, args.attn_impl)

    run_name = args.run_name or default_run_name(args.checkpoint, "v2v")
    run_id, predictions_path, metrics_path = open_outputs(args, run_name, stop)
    raw_values = {name: [] for name in METRIC_NAMES}
    normalized_values = {name: [] for name in METRIC_NAMES}
    baseline_values = {baseline: {name: [] for name in METRIC_NAMES} for baseline in BASELINE_NAMES}
    invalid = recovered = unrecoverable = errors = 0
    with predictions_path.open("w", encoding="utf-8", buffering=1) as predictions_file, \
            metrics_path.open("w", encoding="utf-8", buffering=1) as metrics_file:
        common = dict(run_id=run_id, record_type="config", split="test", task="v2v", index_phrase=args.index_phrase,
                      run_name=run_name, checkpoint=args.checkpoint, test_json=args.test_json,
                      start_index=args.start_index, stop_index=stop, samples=len(dataset))
        write_record(predictions_file, **common, model=args.model, test_video_dir=args.test_video_dir,
                     arguments=vars(args))
        write_record(metrics_file, **common, video_time_mode=args.video_time_mode,
                     timeline_unit=args.timeline_unit, max_pixels=args.max_pixels, min_pixels=args.min_pixels,
                     metric_protocol=SELECTION_PROTOCOL)
        for local_index in tqdm(range(len(dataset)), desc="test", unit="video"):
            dataset_index = args.start_index + local_index
            record, sample = records[local_index], dataset[local_index]
            position = dict(run_id=run_id, split="test", sample_position=local_index + 1,
                            dataset_index=dataset_index, video_id=record["video_id"])
            response = None
            try:
                frames = decode_frames(sample["video_path"], sample["sampled_frames"], args.max_frames)
                response = generate_response(model, processor, sample, args, device, frames, model_type)
                frame_count, timeline_count = sample["sampled_frames"], sample["timeline_count"]
                # Strict parse: the official validity of a prediction. Invalid responses count as an
                # empty selection.
                raw_error = None
                try:
                    raw_prediction = segments_to_frames(parse_interval_response(response, timeline_count),
                                                        frame_count, args.timeline_unit)
                except (json.JSONDecodeError, ValueError) as error:
                    raw_prediction, raw_error = [0] * frame_count, str(error)
                    invalid += 1
                # Post-processed parse (--postprocess), reported as normalized_* next to the strict
                # metrics.
                postprocess_error, actions = None, []
                try:
                    timeline_prediction, intervals, actions = postprocess_interval_response(
                        response, timeline_count, args.postprocess)
                    prediction = segments_to_frames(timeline_prediction, frame_count, args.timeline_unit)
                except (json.JSONDecodeError, ValueError) as error:
                    prediction, intervals, postprocess_error = [0] * frame_count, [], str(error)
                    unrecoverable += 1
                raw_valid, normalized_valid = raw_error is None, postprocess_error is None
                prediction_recovered = not raw_valid and normalized_valid
                recovered += prediction_recovered
                raw_metrics = videoxum_metrics(raw_prediction, sample["references"])
                normalized_metrics = videoxum_metrics(prediction, sample["references"])
                # Random and positional baselines with as many frames as the post-processed
                # prediction.
                baselines = matched_baselines(prediction, sample["references"], videoxum_metrics,
                                              prior=prior, rng=baseline_rng)
                for name in METRIC_NAMES:
                    raw_values[name].append(raw_metrics[name])
                    normalized_values[name].append(normalized_metrics[name])
                    for baseline, values in baselines.items():
                        baseline_values[baseline][name].append(values[name])
                write_record(predictions_file, record_type="sample", **position, sampled_frames=frame_count,
                             timeline_unit=args.timeline_unit, timeline_count=timeline_count,
                             generated_response=response, prediction_valid=raw_valid, parse_error=raw_error,
                             normalized_valid=normalized_valid, postprocess_error=postprocess_error,
                             postprocess_mode=args.postprocess, prediction_recovered=prediction_recovered,
                             repair_actions=actions, predicted_intervals=intervals,
                             selected_frame_indices=[i for i, selected in enumerate(prediction) if selected],
                             selected_frames=sum(prediction), selection_ratio=sum(prediction) / frame_count,
                             raw_selected_frames=sum(raw_prediction),
                             raw_selection_ratio=sum(raw_prediction) / frame_count)
                write_record(metrics_file, record_type="sample", **position, postprocess_mode=args.postprocess,
                             prediction_valid=raw_valid, normalized_valid=normalized_valid,
                             prediction_recovered=prediction_recovered,
                             selection_ratio=sum(prediction) / frame_count, **raw_metrics,
                             **{f"normalized_{name}": value for name, value in normalized_metrics.items()},
                             **{f"baseline_{baseline}_{name}": value for baseline, values in baselines.items()
                                for name, value in values.items()})
            except Exception as error:
                # A failing video (decoding, out of memory, ...) is logged and skipped.
                errors += 1
                write_record(predictions_file, record_type="sample_error", **position,
                             sampled_frames=sample["sampled_frames"], generated_response=response,
                             error_type=type(error).__name__, error=str(error))
                write_record(metrics_file, record_type="sample_error", **position,
                             error_type=type(error).__name__, error=str(error))
                print(f"[warning] test[{dataset_index}] video={record['video_id']}: "
                      f"{type(error).__name__}: {error}", flush=True)
            finally:
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        # Summary: mean strict metrics, post-processed metrics and baselines.
        raw_summary = mean_values(raw_values)
        baseline_summary = {f"baseline_{baseline}_{name}": sum(values) / len(values)
                            for baseline, metrics in baseline_values.items()
                            for name, values in metrics.items() if values}
        write_record(metrics_file, run_id=run_id, record_type="summary", split="test",
                     samples_requested=len(dataset), samples_evaluated=len(raw_values["f1_mean"]),
                     invalid_predictions=invalid, recovered_predictions=recovered,
                     unrecoverable_predictions=unrecoverable, processing_errors=errors,
                     postprocess_mode=args.postprocess, metric_protocol=SELECTION_PROTOCOL,
                     timeline_unit=args.timeline_unit, **raw_summary,
                     **{f"normalized_{name}": value for name, value in mean_values(normalized_values).items()},
                     **baseline_summary)
    print("[test] " + " ".join(f"{name.upper()}={value:.4f}" for name, value in raw_summary.items()
                               if value is not None)
          + f" valid={len(dataset) - invalid - errors}/{len(dataset)}")
    for baseline in BASELINE_NAMES:
        kappa = baseline_summary.get(f"baseline_{baseline}_kappa")
        if kappa is not None:
            print(f"[baseline {baseline}, same frames] KAPPA={kappa:.4f} "
                  f"LIFT={baseline_summary[f'baseline_{baseline}_lift']:.4f} "
                  f"F1_MEAN={baseline_summary[f'baseline_{baseline}_f1_mean']:.4f}")
    print(f"[done] predictions: {predictions_path}")
    print(f"[done] metrics: {metrics_path}")


def joint_sample(record, args):
    # Test sample with the prompt of the checkpoint: V2VT, or V2T for task v2t.
    count = record["sampled_frames"]
    timeline_count = timeline_length(count, args.timeline_unit)
    if args.task == "v2t":
        prompt = build_summary_prompt(timeline_count, args.summary_style, args.timeline_unit,
                                      word_range=args.summary_word_range)
    else:
        prompt = build_joint_prompt(timeline_count, index_phrase=INDEX_PHRASES[args.index_phrase],
                                    summary_style=args.summary_style, unit=args.timeline_unit,
                                    word_range=args.summary_word_range)
    return {**record, "video_path": str(resolve_video(args.test_video_dir, record["video_id"])),
            "timeline_count": timeline_count, "system_prompt": SYSTEM_PROMPTS[args.summary_style],
            "prompt": prompt}


def parse_prediction(response, sample, args):
    # (mask on the original frames, parsed joint response or None, summary). V2T responses select no
    # frame.
    count = sample["sampled_frames"]
    if args.task == "v2t":
        return [0] * count, None, parse_summary_response(response)
    parsed = parse_joint_response(response, count, sample["timeline_count"], args.timeline_unit)
    return parsed.pred_mask.int().tolist(), parsed, parsed.summary


def load_joint_model(args, device):
    # Checked paths, inherited video encoding and the model with the merged starting adapter of GRPO
    # checkpoints.
    check_paths(args)
    resolve_video_encoding(args, args.checkpoint, strict=False)
    model, processor, model_type, args.init_from = load_checkpoint_model(
        args.model, args.checkpoint, device, args.attn_impl, args.init_from)
    return model, processor, model_type


@torch.inference_mode()
def run_joint_inference(args):
    # V2VT (selection + summary) or V2T (summary only): one greedy response per video, parsed
    # strictly.
    records, stop = select_range(load_joint_records(args.test_json, with_target=False),
                                 args.start_index, args.num_samples)
    device = torch.device(args.device)
    model, processor, model_type = load_joint_model(args, device)
    torch.manual_seed(args.seed)
    prior = load_prior(args)
    baseline_rng = random.Random(args.seed)
    run_name = args.run_name or default_run_name(args.checkpoint, args.task)
    run_id, predictions_path, metrics_path = open_outputs(args, run_name, stop)
    totals = {name: [] for name in METRIC_NAMES}
    baseline_totals = {baseline: {name: [] for name in METRIC_NAMES} for baseline in BASELINE_NAMES}
    invalid = errors = 0
    with predictions_path.open("w", encoding="utf-8", buffering=1) as predictions_file, \
            metrics_path.open("w", encoding="utf-8", buffering=1) as metrics_file:
        for handle in (predictions_file, metrics_file):
            write_record(handle, record_type="config", run_id=run_id, arguments=vars(args))
        for position, record in enumerate(tqdm(records, desc="test", unit="video")):
            sample = joint_sample(record, args)
            count, response = sample["sampled_frames"], None
            try:
                frames = decode_frames(sample["video_path"], count, args.max_frames)
                inputs = encode_prompt(processor, sample, args, model_type, frames, device)
                response = generate_completions(model, inputs, processor, args.max_completion_length)[1][0]
                try:
                    prediction, parsed, summary = parse_prediction(response, sample, args)
                    parse_error = None
                except PARSE_ERRORS as error:
                    # Invalid responses count as an empty selection without summary.
                    prediction, parsed, summary, parse_error = [0] * count, None, None, str(error)
                    invalid += 1
                sample_metrics = videoxum_metrics(prediction, record["references"])
                for name, value in sample_metrics.items():
                    totals[name].append(value)
                # V2T selects no frame, so it has no baselines.
                baselines = {} if args.task == "v2t" else matched_baselines(
                    prediction, record["references"], videoxum_metrics, prior=prior, rng=baseline_rng)
                for baseline, values in baselines.items():
                    for name, value in values.items():
                        baseline_totals[baseline][name].append(value)
                write_record(predictions_file, record_type="sample", run_id=run_id,
                             dataset_index=args.start_index + position, video_id=record["video_id"],
                             sampled_frames=count, timeline_unit=args.timeline_unit,
                             timeline_count=sample["timeline_count"], generated_response=response,
                             prediction_valid=parse_error is None, parse_error=parse_error,
                             predicted_intervals=parsed.selected_frames if parsed else [], summary=summary,
                             selected_frame_indices=parsed.selected_indices if parsed else [],
                             selected_frames=sum(prediction), selection_ratio=sum(prediction) / count)
                write_record(metrics_file, record_type="sample", run_id=run_id, video_id=record["video_id"],
                             prediction_valid=parse_error is None, selection_ratio=sum(prediction) / count,
                             **sample_metrics, **{f"baseline_{baseline}_{name}": value
                                                  for baseline, values in baselines.items()
                                                  for name, value in values.items()})
            except Exception as error:
                # A failing video (decoding, out of memory, ...) is logged and skipped.
                errors += 1
                write_record(predictions_file, record_type="sample_error", run_id=run_id, video_id=record["video_id"],
                             generated_response=response, error_type=type(error).__name__, error=str(error))
                print(f"[warning] {record['video_id']}: {type(error).__name__}: {error}", flush=True)
            finally:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        baseline_summary = {f"baseline_{baseline}_{name}": sum(values) / len(values)
                            for baseline, metrics in baseline_totals.items()
                            for name, values in metrics.items() if values}
        write_record(metrics_file, record_type="summary", run_id=run_id, samples_requested=len(records),
                     samples_evaluated=len(totals["f1_mean"]), invalid_predictions=invalid,
                     processing_errors=errors, timeline_unit=args.timeline_unit, init_from=args.init_from,
                     **mean_values(totals), **baseline_summary)
    print(f"[done] predictions: {predictions_path}")
    print(f"[done] metrics: {metrics_path}")


def rank_metrics(scores, references):
    # Official Kendall/Spearman against the mean saliency and, averaged, against each annotator.
    pred = torch.tensor(scores, dtype=torch.float32).unsqueeze(0)
    refs = torch.tensor(references, dtype=torch.float32)
    mask = torch.ones_like(pred)
    repeated = pred.expand(len(refs), -1)
    repeated_mask = torch.ones_like(repeated)
    mean_saliency = refs.mean(dim=0, keepdim=True)
    return {
        "kendall": compute_kendall(pred, mean_saliency, mask, num_repeat=1)[2].item(),
        "spearman": compute_spearman(pred, mean_saliency, mask, num_repeat=1)[2].item(),
        "kendall_per_annotator": compute_kendall(repeated, refs, repeated_mask, num_repeat=len(refs))[2].item(),
        "spearman_per_annotator": compute_spearman(repeated, refs, repeated_mask, num_repeat=len(refs))[2].item(),
    }


def score_metrics(scores, references):
    # Official top-15% F1 and rank metrics of continuous frame scores.
    pred = torch.tensor(scores, dtype=torch.float32).unsqueeze(0).expand(len(references), -1)
    refs = torch.tensor(references, dtype=torch.float32)
    _, f1_max, f1_mean = compute_f1(pred, refs, torch.ones_like(pred), num_repeat=len(references))
    return {"f1_top15_max": f1_max.item(), "f1_top15_mean": f1_mean.item(), **rank_metrics(scores, references)}


def sampled_scores(responses, sample, args):
    # Score of a frame: the share of valid sampled responses that select it; also returns the number
    # of valid responses.
    masks = []
    for response in responses:
        try:
            masks.append(parse_prediction(response, sample, args)[0])
        except PARSE_ERRORS:
            continue
    if not masks:
        return None, 0
    return [sum(column) / len(masks) for column in zip(*masks)], len(masks)


@torch.inference_mode()
def run_score_sampling(args):
    # V2VT greedy prediction plus frame scores from K sampled responses: the official rank metrics
    # need continuous scores, and the share of sampled responses that select a frame estimates
    # P(frame selected).
    records, stop = select_range(load_joint_records(args.test_json, with_target=False),
                                 args.start_index, args.num_samples)
    device = torch.device(args.device)
    model, processor, model_type = load_joint_model(args, device)
    torch.manual_seed(args.seed)
    run_id, predictions_path, metrics_path = open_outputs(args, args.run_name, stop)
    sampling = dict(num_samples=args.score_samples, temperature=args.score_temperature, top_p=args.score_top_p)
    totals, valid_counts = {}, []
    invalid_greedy = no_valid_samples = errors = 0
    with predictions_path.open("w", encoding="utf-8", buffering=1) as predictions_file, \
            metrics_path.open("w", encoding="utf-8", buffering=1) as metrics_file:
        for handle in (predictions_file, metrics_file):
            write_record(handle, record_type="config", run_id=run_id, mode="score_sampling", arguments=vars(args))
        for position, record in enumerate(tqdm(records, desc="test", unit="video")):
            sample = joint_sample(record, args)
            count, response = sample["sampled_frames"], None
            try:
                frames = decode_frames(sample["video_path"], count, args.max_frames)
                inputs = encode_prompt(processor, sample, args, model_type, frames, device)
                response = generate_completions(model, inputs, processor, args.max_completion_length)[1][0]
                try:
                    prediction, parsed, summary = parse_prediction(response, sample, args)
                    parse_error = None
                except PARSE_ERRORS as error:
                    prediction, parsed, summary, parse_error = [0] * count, None, None, str(error)
                    invalid_greedy += 1
                # K sampled responses for the same prompt.
                sampled = generate_completions(model, inputs, processor, args.max_completion_length, sampling)[1]
                scores, valid = sampled_scores(sampled, sample, args)
                valid_counts.append(valid)
                # Without any valid sampled response the scores fall back to the greedy selection.
                if scores is None:
                    no_valid_samples += 1
                    scores = [float(value) for value in prediction]
                # greedy_*: metrics of the greedy selection; score_*: metrics of the sampled scores.
                metrics = {
                    **{f"greedy_{k}": v for k, v in {**videoxum_metrics(prediction, record["references"]),
                                                    **rank_metrics(prediction, record["references"])}.items()},
                    **{f"score_{k}": v for k, v in score_metrics(scores, record["references"]).items()},
                }
                for name, value in metrics.items():
                    totals.setdefault(name, []).append(value)
                write_record(predictions_file, record_type="sample", run_id=run_id,
                             dataset_index=args.start_index + position, video_id=record["video_id"],
                             sampled_frames=count, timeline_unit=args.timeline_unit,
                             timeline_count=sample["timeline_count"], generated_response=response,
                             prediction_valid=parse_error is None, parse_error=parse_error,
                             predicted_intervals=parsed.selected_frames if parsed else [], summary=summary,
                             selected_frame_indices=parsed.selected_indices if parsed else [],
                             selected_frames=sum(prediction), selection_ratio=sum(prediction) / count,
                             frame_scores=[round(value, 3) for value in scores], score_valid_samples=valid,
                             score_selection_ratio_mean=sum(scores) / count)
                write_record(metrics_file, record_type="sample", run_id=run_id, video_id=record["video_id"],
                             prediction_valid=parse_error is None, score_valid_samples=valid, **metrics)
            except Exception as error:
                # A failing video (decoding, out of memory, ...) is logged and skipped.
                errors += 1
                for handle in (predictions_file, metrics_file):
                    write_record(handle, record_type="sample_error", run_id=run_id, video_id=record["video_id"],
                                 generated_response=response, error_type=type(error).__name__, error=str(error))
                print(f"[warning] {record['video_id']}: {type(error).__name__}: {error}", flush=True)
            finally:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        summary = mean_values(totals)
        write_record(metrics_file, record_type="summary", run_id=run_id, samples_requested=len(records),
                     samples_evaluated=len(valid_counts), invalid_greedy_predictions=invalid_greedy,
                     videos_without_valid_samples=no_valid_samples, valid_samples_mean=mean_or_none(valid_counts),
                     score_samples=args.score_samples, processing_errors=errors, **summary)
    for variant in ("greedy", "score"):
        print(f"[test] {variant}: " + " ".join(
            f"{name[len(variant) + 1:]}={value:.4f}" for name, value in summary.items()
            if name.startswith(f"{variant}_") and value is not None))
    print(f"[done] predictions: {predictions_path}")
    print(f"[done] metrics: {metrics_path}")

#!/usr/bin/env python3
"""Inferenza test di un adapter GRPO VideoXum."""
import argparse, json, math, os
from datetime import datetime, timezone
from pathlib import Path
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import random
import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoProcessor
from evaluation.chance_baselines import load_positional_prior, matched_baselines
from grpo_rewards import (
    CompletionValidationError,
    expand_pair_completion,
    parse_and_validate_completion,
)
from sft_train_videoxum import (
    METRIC_NAMES,
    add_pixel_arguments,
    add_video_time_argument,
    resolve_video_time_mode,
    timeline_length,
    videoxum_metrics,
    write_metric,
)
from train_grpo import (
    build_grpo_prompt,
    build_summary_only_prompt,
    SUMMARY_STYLES,
    decode_frames,
    generate,
    load_records,
    prompt_inputs,
    read_grpo_init,
    resolve_video,
)
from vl_model_loader import load_qwen_vl_model

BASELINE_NAMES = ("random", "positional")

def nonnegative_int(value):
    value = int(value)
    if value < 0: raise argparse.ArgumentTypeError("deve essere >= 0")
    return value

def positive_int(value):
    value = int(value)
    if value <= 0: raise argparse.ArgumentTypeError("deve essere > 0")
    return value

def parse_args():
    p = argparse.ArgumentParser(description="Inferenza GRPO sul test VideoXum")
    p.add_argument("--model", type=Path, required=True); p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--test_json", type=Path, required=True); p.add_argument("--test_video_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True); p.add_argument("--run_name", default="grpo-final-test")
    p.add_argument("--start_index", type=nonnegative_int, default=0); p.add_argument("--num_samples", type=nonnegative_int, default=0)
    p.add_argument("--device", default="cuda:0"); p.add_argument("--attn_impl", choices=("sdpa", "flash_attention_2"), default="sdpa")
    p.add_argument("--max_frames", type=nonnegative_int, default=0); add_pixel_arguments(p)
    p.add_argument("--init_from", type=Path,
        help="adapter SFT/DPO fuso prima del LoRA GRPO; se omesso si legge grpo_init.json "
             "del checkpoint (obbligatorio per i checkpoint GRPO precedenti al sidecar "
             "partiti da SFT/DPO)")
    p.add_argument("--prior_json", type=Path,
        help="metadata per la baseline posizionale (default: train_videoxum.json accanto a --test_json)")
    p.add_argument("--max_completion_length", type=positive_int, default=384); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--summary_style", choices=SUMMARY_STYLES, default="reference",
        help="deve coincidere con lo stile di training del checkpoint: concise per i checkpoint GRPO precedenti al 2026-09-17")
    add_video_time_argument(p)
    p.add_argument(
        "--summary_only", action="store_true",
        help="richiede solo {\"summary\":\"...\"}; senza questa opzione mantiene anche selected_frames",
    )
    return p.parse_args()


def parse_summary_only_response(response):
    payload = json.loads(response)
    if not isinstance(payload, dict) or set(payload) != {"summary"}:
        raise ValueError('la risposta deve contenere solo il campo "summary"')
    if not isinstance(payload["summary"], str) or not payload["summary"].strip():
        raise ValueError("summary deve essere una stringa non vuota")
    return payload["summary"].strip()

@torch.inference_mode()
def main(args):
    for path in (args.model, args.checkpoint, args.test_video_dir):
        if not path.is_dir(): raise FileNotFoundError(path)
    if not (args.checkpoint/"adapter_config.json").is_file(): raise FileNotFoundError(f"adapter_config.json assente in {args.checkpoint}")
    found, saved_init = read_grpo_init(args.checkpoint)
    if args.init_from is None:
        args.init_from = saved_init
        if not found:
            print(f"[WARN] {args.checkpoint} non registra l'adapter di partenza (grpo_init.json): "
                  "il LoRA GRPO viene applicato al modello base. Se GRPO era partito da "
                  "SFT/DPO passare --init_from, altrimenti il modello valutato non e' quello addestrato.",
                  flush=True)
    elif found and saved_init is not None and Path(args.init_from) != saved_init:
        raise SystemExit(f"[model] {args.checkpoint} parte da {saved_init}, non da {args.init_from}")
    resolve_video_time_mode(args, args.checkpoint, strict=False)
    records = load_records(args.test_json)
    if args.start_index >= len(records): raise ValueError(f"start_index={args.start_index}, test size={len(records)}")
    stop = len(records) if args.num_samples == 0 else min(len(records), args.start_index+args.num_samples)
    records = records[args.start_index:stop]
    device = torch.device(args.device); dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    base, _ = load_qwen_vl_model(args.model, torch_dtype=dtype, attn_implementation=args.attn_impl)
    base = base.to(device)
    if args.init_from is not None:
        # Stesso modello su cui e' stato addestrato il LoRA GRPO (train_grpo.train).
        base = PeftModel.from_pretrained(base, args.init_from, is_trainable=False, local_files_only=True).merge_and_unload()
        print(f"[model] adapter di partenza fuso: {args.init_from}")
    model = PeftModel.from_pretrained(base, args.checkpoint, is_trainable=False, local_files_only=True)
    processor = AutoProcessor.from_pretrained(args.checkpoint, local_files_only=True); model.eval(); torch.manual_seed(args.seed)
    prior_json = args.prior_json or args.test_json.parent / "train_videoxum.json"
    positional_prior = load_positional_prior(prior_json) if prior_json.is_file() else None
    baseline_rng = random.Random(args.seed)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"); output_dir = args.output_dir/args.run_name; output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir/f"predictions-test-{args.start_index}-{stop}-{run_id}.jsonl"; metrics_path = output_dir/f"metrics-test-{args.start_index}-{stop}-{run_id}.jsonl"
    totals = {name: [] for name in METRIC_NAMES}; invalid = errors = 0
    baseline_totals = {baseline: {name: [] for name in METRIC_NAMES} for baseline in BASELINE_NAMES}
    with predictions_path.open("w", encoding="utf-8", buffering=1) as pred_file, metrics_path.open("w", encoding="utf-8", buffering=1) as metric_file:
        write_metric(pred_file, record_type="config", run_id=run_id, arguments=vars(args)); write_metric(metric_file, record_type="config", run_id=run_id, arguments=vars(args))
        for position, record in enumerate(tqdm(records, desc="test", unit="video")):
            count = record["count"]; response = None
            timeline_count = timeline_length(count, args.timeline_unit)
            prompt = (
                build_summary_only_prompt(timeline_count, args.summary_style, args.timeline_unit)
                if args.summary_only else
                build_grpo_prompt(timeline_count, summary_style=args.summary_style, unit=args.timeline_unit)
            )
            sample = {**record, "video_path": str(resolve_video(args.test_video_dir, record["video_id"])), "prompt": prompt}
            try:
                frames = decode_frames(sample, args.max_frames); inputs = prompt_inputs(processor, sample, args, device, frames)
                _, responses = generate(model, inputs, processor, args, training=False); response = responses[0]
                try:
                    if args.summary_only:
                        summary = parse_summary_only_response(response)
                        parsed = None
                        prediction = [0] * count
                    else:
                        parsed = parse_and_validate_completion(
                            response, timeline_count, max(1, math.floor(.50*timeline_count))
                        )
                        if args.timeline_unit == "pair":
                            parsed = expand_pair_completion(parsed, count)
                        prediction = parsed.pred_mask.int().tolist()
                        summary = parsed.summary
                    parse_error = None
                except CompletionValidationError as error:
                    prediction = [0]*count; parsed = None; parse_error = str(error); invalid += 1
                    summary = None
                except (json.JSONDecodeError, ValueError) as error:
                    prediction = [0]*count; parsed = None; parse_error = str(error); invalid += 1
                    summary = None
                sample_metrics = videoxum_metrics(prediction, record["vsum_onehot"])
                for name, value in sample_metrics.items(): totals[name].append(value)
                baseline_metrics = {} if args.summary_only else matched_baselines(
                    prediction, record["vsum_onehot"], videoxum_metrics,
                    prior=positional_prior, rng=baseline_rng,
                )
                for baseline, values in baseline_metrics.items():
                    for name, value in values.items(): baseline_totals[baseline][name].append(value)
                # predicted_intervals e selected_frame_indices sono sempre sui frame.
                write_metric(pred_file, record_type="sample", run_id=run_id, dataset_index=args.start_index+position, video_id=record["video_id"], sampled_frames=count, timeline_unit=args.timeline_unit, timeline_count=timeline_count, generated_response=response, prediction_valid=parse_error is None, parse_error=parse_error, predicted_intervals=parsed.selected_frames if parsed else [], summary=summary, selected_frame_indices=parsed.selected_indices if parsed else [], selected_frames=sum(prediction), selection_ratio=sum(prediction)/count)
                write_metric(metric_file, record_type="sample", run_id=run_id, video_id=record["video_id"], prediction_valid=parse_error is None, selection_ratio=sum(prediction)/count, **sample_metrics,
                             **{f"baseline_{baseline}_{name}": value for baseline, values in baseline_metrics.items() for name, value in values.items()})
            except Exception as error:
                errors += 1; write_metric(pred_file, record_type="sample_error", run_id=run_id, video_id=record["video_id"], generated_response=response, error_type=type(error).__name__, error=str(error)); print(f"[warning] {record['video_id']}: {type(error).__name__}: {error}", flush=True)
            finally:
                if device.type == "cuda": torch.cuda.empty_cache()
        summary = {name: sum(values)/len(values) if values else None for name, values in totals.items()}
        baseline_summary = {f"baseline_{baseline}_{name}": sum(values)/len(values)
                            for baseline, metrics in baseline_totals.items() for name, values in metrics.items() if values}
        write_metric(metric_file, record_type="summary", run_id=run_id, samples_requested=len(records), samples_evaluated=len(totals["f1_mean"]), invalid_predictions=invalid, processing_errors=errors, timeline_unit=args.timeline_unit, init_from=args.init_from, **summary, **baseline_summary)
    print(f"[done] predizioni: {predictions_path}"); print(f"[done] metriche: {metrics_path}")

if __name__ == "__main__": main(parse_args())

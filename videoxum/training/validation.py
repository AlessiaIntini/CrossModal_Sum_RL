"""Generative validation of every task, collapse diagnostics and checkpoint-only evaluation."""

import json
import random

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

from ..inputs import encode_prompt, generate_completions, generate_response, move_to_device, prepare_completion
from ..jsonl import new_run_id
from ..metrics import (
    METRIC_NAMES, TEXT_METRIC_NAMES, binary_f1, collapse_metrics, compute_cider_score, compute_corpus_bleu,
    distinct_summary_ratio, finite, format_collapse_summary, format_selection_summary, summary_metrics,
    videoxum_metrics,
)
from ..models import load_checkpoint_model
from ..parsing import (CompletionValidationError, parse_joint_response, parse_summary_response,
                       parse_timeline_response)
from ..rewards import ensure_video_prepared
from ..video import decode_frames, resolve_video_encoding
from .common import RunLog, subset_indices

SELECTION_PROTOCOL = "VideoXum aggregation; binary generated saliency"


def resolve_best_metric(args):
    # Default checkpoint-best metric: kappa on the pair timeline, where f1_mean rewards selecting
    # more frames; f1_mean otherwise.
    if args.best_metric is None:
        args.best_metric = "kappa" if args.timeline_unit == "pair" else "f1_mean"
    print(f"[best] checkpoint-best selected by {args.best_metric}")


def validation_limit(dataset, requested):
    # Number of validation videos; a request <= 0 means all.
    return len(dataset) if requested <= 0 else min(requested, len(dataset))


def checkpoint_score(metric_values, args):
    # best_metric "v2vt" averages the selection metric (kappa or f1_mean) and ROUGE-L.
    if args.best_metric == "v2vt":
        selection = "kappa" if args.timeline_unit == "pair" else "f1_mean"
        values = [(a + b) / 2 for a, b in zip(metric_values[selection], metric_values["rougeL"])]
    else:
        values = metric_values[args.best_metric]
    # With periodic validation every score uses the first --eval_steps_samples videos, so periodic
    # and end-of-epoch validations are comparable.
    if args.eval_steps:
        values = values[:args.eval_steps_samples]
    return sum(values) / len(values)


def sample_frames(sample, args):
    # Decoded 1 fps frames of a dataset sample.
    return decode_frames(sample["video_path"], sample["sampled_frames"], args.max_frames)


def completion_log_prob(model, inputs, prompt_len, normalize_by_length=True):
    # Log-probability of the completion tokens, mean or sum over tokens.
    input_ids = inputs["input_ids"][0]
    with torch.autocast(device_type=input_ids.device.type, dtype=torch.bfloat16,
                        enabled=input_ids.device.type == "cuda"):
        outputs = model(**inputs)
    # The logits at position t predict token t + 1.
    logits = outputs.logits[0, prompt_len - 1:input_ids.numel() - 1].float()
    labels = input_ids[prompt_len:]
    del outputs
    token_log_probs = F.log_softmax(logits, dim=-1).gather(1, labels[:, None]).squeeze(1)
    return token_log_probs.mean() if normalize_by_length else token_log_probs.sum()


def pairwise_margin(model, processor, sample, args, device, model_type, frames):
    # (log p(chosen) - log p(rejected), chosen tokens, rejected tokens) of a preference pair.
    scores, tokens = [], []
    for key in ("chosen", "rejected"):
        inputs, start = prepare_completion(processor, sample, sample[key], args, model_type, frames)
        inputs = move_to_device(inputs, device)
        scores.append(completion_log_prob(model, inputs, start, args.length_normalize))
        tokens.append(inputs["input_ids"].shape[-1] - start)
        del inputs
    return (scores[0] - scores[1]).item(), tokens[0], tokens[1]


@torch.no_grad()
def evaluate_selection(model, processor, dataset, args, device, log, epoch, model_type, limit=None,
                       step=None, pairwise=False):
    # V2V validation: greedy intervals scored with the VideoXum metrics.
    model.eval()
    limit = validation_limit(dataset, args.eval_samples if limit is None else limit)
    metric_values = {name: [] for name in METRIC_NAMES}
    responses, predictions, references_list, margins = [], [], [], []
    invalid_predictions = 0
    for index in tqdm(range(limit), desc="validation", unit="video", leave=False):
        sample = dataset[index]
        frames = sample_frames(sample, args)
        pair_fields = {}
        # With pairwise=True, samples with chosen/rejected labels also report the DPO
        # log-probability margin of the policy.
        if pairwise:
            margin = chosen_tokens = rejected_tokens = None
            if "chosen" in sample:
                margin, chosen_tokens, rejected_tokens = pairwise_margin(
                    model, processor, sample, args, device, model_type, frames)
                margins.append(margin)
            pair_fields = dict(chosen_completion_tokens=chosen_tokens, rejected_completion_tokens=rejected_tokens,
                               policy_margin=margin)
        response = generate_response(model, processor, sample, args, device, frames, model_type)
        parse_error = None
        try:
            prediction = parse_timeline_response(response, sample)
        except (json.JSONDecodeError, ValueError) as error:
            # Invalid responses count as an empty selection.
            prediction, parse_error = [0] * sample["sampled_frames"], str(error)
            invalid_predictions += 1
        sample_metrics = videoxum_metrics(prediction, sample["references"])
        for name, value in sample_metrics.items():
            metric_values[name].append(value)
        responses.append(response)
        predictions.append(prediction)
        references_list.append(sample["references"])
        log.write(record_type="sample", split="validation", epoch=epoch, optimizer_step=step,
                  sample_position=index + 1, dataset_index=index, **pair_fields, generated_response=response,
                  model_type=model_type, prediction_valid=parse_error is None, parse_error=parse_error,
                  selected_frames=sum(prediction), selection_ratio=sum(prediction) / len(prediction),
                  **sample_metrics)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = {name: sum(values) / len(values) for name, values in metric_values.items()}
    summary["selection_score"] = checkpoint_score(metric_values, args)
    # Collapse diagnostics: how much the outputs change from video to video.
    collapse = collapse_metrics(responses, predictions, references_list)
    pairwise_summary = {}
    if pairwise:
        accuracy = sum(margin > 0 for margin in margins) / len(margins) if margins else None
        mean_margin = sum(margins) / len(margins) if margins else None
        pairwise_summary = dict(pairwise_samples=len(margins), pairwise_accuracy=accuracy,
                                mean_policy_margin=mean_margin)
        print(f"[val] pairwise_accuracy={accuracy:.4f} mean_logprob_margin={mean_margin:.4f}" if margins
              else "[val] no chosen/rejected pairs in the validation set")
    print(format_selection_summary(summary, limit, invalid_predictions))
    print(f"[val] collapse: {format_collapse_summary(collapse)}")
    log.write(record_type="summary", split="validation", epoch=epoch, optimizer_step=step, samples=limit,
              **pairwise_summary, invalid_predictions=invalid_predictions, metric_protocol=SELECTION_PROTOCOL,
              video_time_mode=args.video_time_mode, timeline_unit=args.timeline_unit,
              best_metric=args.best_metric, collapse=collapse, **summary)
    return summary


def corpus_summary_values(candidates, references):
    # Corpus BLEU@1-4 and CIDEr of the validation summaries (None for NaN).
    values = {name: finite(value) for name, value in compute_corpus_bleu(candidates, references).items()}
    values["cider"] = finite(compute_cider_score(candidates, references)[0])
    return values


@torch.no_grad()
def evaluate_summary(model, processor, dataset, args, device, log, epoch, model_type, limit=None, step=None):
    # V2T validation: the generated summary against the joined VideoXum captions.
    model.eval()
    limit = validation_limit(dataset, args.eval_samples if limit is None else limit)
    metric_values = {name: [] for name in TEXT_METRIC_NAMES}
    candidates, references = [], []
    invalid_predictions = 0
    for index in tqdm(range(limit), desc="validation", unit="video", leave=False):
        sample = dataset[index]
        response = generate_response(model, processor, sample, args, device, sample_frames(sample, args), model_type)
        parse_error = None
        try:
            summary = parse_summary_response(response)
        except (json.JSONDecodeError, ValueError) as error:
            # Invalid responses count as an empty summary.
            summary, parse_error = "", str(error)
            invalid_predictions += 1
        sample_metrics = summary_metrics(summary, sample["reference_summary"])
        for name, value in sample_metrics.items():
            metric_values[name].append(value)
        candidates.append(summary)
        references.append(sample["reference_summary"])
        log.write(record_type="sample", split="validation", epoch=epoch, optimizer_step=step,
                  sample_position=index + 1, dataset_index=index, video_id=sample["video_id"],
                  generated_response=response, model_type=model_type, prediction_valid=parse_error is None,
                  parse_error=parse_error, summary=summary, **sample_metrics)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    values = {name: sum(items) / len(items) for name, items in metric_values.items()}
    values["selection_score"] = checkpoint_score(metric_values, args)
    values.update(corpus_summary_values(candidates, references))
    values["distinct_summary_ratio"] = distinct_summary_ratio(candidates)
    cider = "n/a" if values["cider"] is None else f"{values['cider']:.4f}"
    print(f"[val] ROUGE-1={values['rouge1']:.4f} ROUGE-L={values['rougeL']:.4f} "
          f"BLEU@4={values['bleu4_sentence']:.4f} CIDEr={cider} words={values['summary_words']:.1f} "
          f"distinct_summary_ratio={values['distinct_summary_ratio']:.3f} "
          f"valid_predictions={limit - invalid_predictions}/{limit}")
    log.write(record_type="summary", split="validation", epoch=epoch, optimizer_step=step, samples=limit,
              invalid_predictions=invalid_predictions,
              metric_protocol="summary vs joined tsum (split() tokenization)",
              video_time_mode=args.video_time_mode, timeline_unit=args.timeline_unit,
              summary_style=args.summary_style, best_metric=args.best_metric, **values)
    return values


@torch.no_grad()
def evaluate_joint(model, processor, dataset, args, device, log, epoch, model_type, limit=None, step=None):
    # V2VT validation: the selection metrics of V2V and the text metrics of V2T on one response.
    model.eval()
    limit = validation_limit(dataset, args.eval_samples if limit is None else limit)
    metric_values = {name: [] for name in METRIC_NAMES + TEXT_METRIC_NAMES}
    responses, predictions, references_list, candidates, references = [], [], [], [], []
    invalid_predictions = 0
    for index in tqdm(range(limit), desc="validation", unit="video", leave=False):
        sample = dataset[index]
        response = generate_response(model, processor, sample, args, device, sample_frames(sample, args), model_type)
        parse_error = None
        try:
            parsed = parse_joint_response(response, sample["sampled_frames"], sample["timeline_count"],
                                          sample["timeline_unit"])
            prediction, summary = parsed.pred_mask.int().tolist(), parsed.summary
        except (CompletionValidationError, json.JSONDecodeError, ValueError) as error:
            # Invalid responses count as an empty selection and an empty summary.
            prediction, summary, parse_error = [0] * sample["sampled_frames"], "", str(error)
            invalid_predictions += 1
        sample_metrics = videoxum_metrics(prediction, sample["references"])
        sample_metrics.update(summary_metrics(summary, sample["reference_summary"]))
        for name, value in sample_metrics.items():
            metric_values[name].append(value)
        responses.append(response)
        predictions.append(prediction)
        references_list.append(sample["references"])
        candidates.append(summary)
        references.append(sample["reference_summary"])
        log.write(record_type="sample", split="validation", epoch=epoch, optimizer_step=step,
                  sample_position=index + 1, dataset_index=index, video_id=sample["video_id"],
                  generated_response=response, model_type=model_type, prediction_valid=parse_error is None,
                  parse_error=parse_error, summary=summary, selected_frames=sum(prediction),
                  selection_ratio=sum(prediction) / len(prediction), **sample_metrics)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    values = {name: sum(items) / len(items) for name, items in metric_values.items()}
    values["selection_score"] = checkpoint_score(metric_values, args)
    values.update(corpus_summary_values(candidates, references))
    collapse = collapse_metrics(responses, predictions, references_list)
    collapse["distinct_summary_ratio"] = distinct_summary_ratio(candidates)
    cider = "n/a" if values["cider"] is None else f"{values['cider']:.4f}"
    print(f"[val] VideoXum F1_MEAN={values['f1_mean']:.4f} Kappa={values['kappa']:.4f} "
          f"Kendall={values['kendall']:.4f} | ROUGE-L={values['rougeL']:.4f} "
          f"BLEU@4={values['bleu4_sentence']:.4f} CIDEr={cider} words={values['summary_words']:.1f} "
          f"valid_predictions={limit - invalid_predictions}/{limit}")
    print(f"[val] collapse: {format_collapse_summary(collapse)} "
          f"distinct_summary_ratio={collapse['distinct_summary_ratio']:.3f}")
    log.write(record_type="summary", split="validation", epoch=epoch, optimizer_step=step, samples=limit,
              invalid_predictions=invalid_predictions,
              metric_protocol="VideoXum aggregation; binary generated saliency; "
                              "summary vs joined tsum (split() tokenization)",
              video_time_mode=args.video_time_mode, timeline_unit=args.timeline_unit,
              summary_style=args.summary_style, best_metric=args.best_metric, collapse=collapse, **values)
    return values


@torch.no_grad()
def evaluate_grpo(model, processor, dataset, reward, args, device, log, epoch, model_type):
    # GRPO validation: mean reward components of greedy responses on a fixed subset of --val_samples
    # videos.
    model.eval()
    totals, evaluated, skipped = {}, 0, 0
    indices = subset_indices(len(dataset), args.val_samples, args.seed)
    for position, index in enumerate(tqdm(indices, desc="validation", leave=False), 1):
        sample = dataset[index]
        try:
            ensure_video_prepared(reward, sample)
            inputs = encode_prompt(processor, sample, args, model_type, sample_frames(sample, args), device)
            _, texts = generate_completions(model, inputs, processor, args.max_completion_length)
            row = reward.score(texts[0], sample["cache_key"])
        except Exception as error:
            # A failing video is skipped and counted.
            print(f"[warning] skip validation {sample['video_id']}: {type(error).__name__}: {error}")
            skipped += 1
            continue
        evaluated += 1
        for key, value in row.items():
            # Only scalar components are averaged.
            if isinstance(value, (int, float, bool)):
                totals[key] = totals.get(key, 0.0) + float(value)
        log.write(record_type="sample", split="validation", epoch=epoch, sample_position=position,
                  dataset_index=index, video_id=sample["video_id"], completion=texts[0], **row)
    means = {key: value / max(evaluated, 1) for key, value in totals.items()}
    log.write(record_type="summary", split="validation", epoch=epoch, samples=evaluated, skipped=skipped, **means)
    print(f"[val] reward={means.get('total_reward', 0):.4f} reward_default={means.get('total_reward_default', 0):.4f} "
          f"valid={means.get('valid', 0) * 100:.1f}% F_beta={means.get('f_beta_reward', 0):.4f} "
          f"summary_words={means.get('summary_words', 0):.1f} "
          f"summary_length={means.get('summary_length_reward', 0):.4f} samples={evaluated} skipped={skipped}")
    return means


@torch.no_grad()
def measure_generation_diversity(model, processor, dataset, args, device, log, epoch, model_type):
    # Distinct sampled responses per video: values near 0 signal a collapsed policy, which would
    # leave GRPO without reward variance.
    model.eval()
    count = min(args.diversity_check_samples, len(dataset))
    unique_ratios = []
    for index in range(count):
        sample = dataset[index]
        inputs = encode_prompt(processor, sample, args, model_type, sample_frames(sample, args), device)
        prompt_length = inputs["input_ids"].shape[-1]
        responses = []
        for _ in range(args.diversity_check_k):
            generated = model.generate(**inputs, max_new_tokens=args.eval_max_new_tokens, do_sample=True,
                                       temperature=args.diversity_temperature, top_p=0.95, use_cache=True)
            responses.append(processor.batch_decode(generated[:, prompt_length:], skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=False)[0].strip())
        # Agreement between the valid sampled selections (mean pairwise F1).
        predictions = []
        for response in responses:
            try:
                predictions.append(parse_timeline_response(response, sample))
            except (json.JSONDecodeError, ValueError):
                predictions.append(None)
        valid = [prediction for prediction in predictions if prediction is not None]
        pairwise_f1 = [binary_f1(valid[i], valid[j]) for i in range(len(valid)) for j in range(i + 1, len(valid))]
        unique_ratio = len(set(responses)) / len(responses)
        unique_ratios.append(unique_ratio)
        log.write(record_type="diversity", split="validation", epoch=epoch, dataset_index=index,
                  k=args.diversity_check_k, temperature=args.diversity_temperature, unique_ratio=unique_ratio,
                  valid_responses=len(valid),
                  mean_pairwise_f1_among_samples=sum(pairwise_f1) / len(pairwise_f1) if pairwise_f1 else None,
                  responses=responses)
    if unique_ratios:
        mean_unique_ratio = sum(unique_ratios) / len(unique_ratios)
        log.write(record_type="diversity_summary", split="validation", epoch=epoch, samples=count,
                  mean_unique_ratio=mean_unique_ratio)
        print(f"[diversity] mean_unique_ratio={mean_unique_ratio:.2f} on {count} videos "
              f"(k={args.diversity_check_k}, T={args.diversity_temperature})")


@torch.no_grad()
def measure_video_dependence(model, processor, dataset, args, device, log, epoch, model_type):
    # Regenerate with the frames of another video (swap) or with gray frames (blank): if the
    # response does not change, the model is not using the video.
    model.eval()
    count = min(args.video_dependence_samples, len(dataset))
    if count < 2:
        return None
    frames_cache = [sample_frames(dataset[index], args) for index in range(count)]
    identical, ious = 0, []
    for index in tqdm(range(count), desc="video dependence", unit="video", leave=False):
        sample = dataset[index]
        original_frames = frames_cache[index]
        if args.video_dependence_mode == "blank":
            substituted = [Image.new("RGB", original_frames[0].size, (128, 128, 128))] * len(original_frames)
        else:
            # The donor video is brought to exactly the same number of frames, so the prompt is
            # unchanged.
            donor = frames_cache[(index + 1) % count]
            substituted = (donor + [donor[-1]] * len(original_frames))[:len(original_frames)]
        original = generate_response(model, processor, sample, args, device, original_frames, model_type)
        replaced = generate_response(model, processor, sample, args, device, substituted, model_type)
        same = original == replaced
        identical += same
        iou = None
        # IoU between the original and the substituted selections, when both are valid.
        try:
            first = parse_timeline_response(original, sample)
            second = parse_timeline_response(replaced, sample)
            union = sum(a | b for a, b in zip(first, second))
            iou = sum(a & b for a, b in zip(first, second)) / union if union else None
        except (json.JSONDecodeError, ValueError):
            pass
        if iou is not None:
            ious.append(iou)
        log.write(record_type="video_dependence", split="validation", epoch=epoch, dataset_index=index,
                  mode=args.video_dependence_mode, original=original, substituted=replaced, identical=same, iou=iou)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summary = {
        "samples": count,
        "mode": args.video_dependence_mode,
        "identical_output_rate": identical / count,
        "mean_iou_original_vs_substituted": sum(ious) / len(ious) if ious else None,
    }
    log.write(record_type="video_dependence_summary", split="validation", epoch=epoch,
              video_time_mode=args.video_time_mode, **summary)
    print(f"[video dependence] {args.video_dependence_mode}: identical_output_rate="
          f"{summary['identical_output_rate']:.3f} on {count} videos, "
          f"mean IoU={summary['mean_iou_original_vs_substituted']}")
    return summary


def validate_checkpoint(args, load_dataset, pairwise=False):
    # --eval_only: V2V validation of a saved adapter, written to
    # validation-<checkpoint>-<run>.jsonl.
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if not args.model.is_dir():
        raise FileNotFoundError(f"base model not found: {args.model}")
    if not args.eval_checkpoint.is_dir():
        raise FileNotFoundError(f"adapter checkpoint not found: {args.eval_checkpoint}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available: run on a GPU node")
    # The video encoding is inherited from the checkpoint before load_dataset(args) builds the
    # validation set.
    resolve_video_encoding(args, args.eval_checkpoint, strict=False)
    resolve_best_metric(args)
    device = torch.device(args.device)
    results_dir = args.resultsroot / args.model.name
    results_dir.mkdir(parents=True, exist_ok=True)
    run_id = new_run_id()
    metrics_path = results_dir / f"validation-{args.eval_checkpoint.name}-{run_id}.jsonl"
    dataset = load_dataset(args)
    model, processor, model_type, _ = load_checkpoint_model(args.model, args.eval_checkpoint, device, args.attn_impl)
    with metrics_path.open("w", encoding="utf-8", buffering=1) as handle:
        log = RunLog(None, metrics_path, handle, run_id)
        log.write(record_type="config", mode="validation_only", checkpoint=args.eval_checkpoint,
                  arguments=vars(args))
        evaluate_selection(model, processor, dataset, args, device, log, 0, model_type, pairwise=pairwise)
        if getattr(args, "video_dependence_samples", 0) > 0:
            measure_video_dependence(model, processor, dataset, args, device, log, 0, model_type)
    print(f"[done] validation of {args.eval_checkpoint}: {metrics_path}")

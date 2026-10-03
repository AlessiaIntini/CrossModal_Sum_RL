"""Selection metrics (VideoXum protocol), output-collapse diagnostics and summary text metrics."""

import math
import random
import re
import warnings
from collections import Counter, defaultdict

import numpy as np
import torch

from .timeline import selected_runs

METRIC_NAMES = ("f1_max", "f1_mean", "kendall", "spearman", "kappa", "lift")
TEXT_METRIC_NAMES = ("rouge1", "rougeL", "bleu4_sentence", "summary_words")
BLEU_WEIGHTS = {
    "bleu1": (1.0, 0.0, 0.0, 0.0),
    "bleu2": (0.5, 0.5, 0.0, 0.0),
    "bleu3": (1 / 3, 1 / 3, 1 / 3, 0.0),
    "bleu4": (0.25, 0.25, 0.25, 0.25),
}
_cider_warned = False
_rouge_scorer = None


def get_frame_scores(pred, gt, mask):
    # Ground truth and prediction restricted to the valid frames (official evalVtV).
    return gt[mask == 1], pred[mask == 1]


def compute_kendall(pred, gt, mask, num_repeat=10):
    from scipy.stats import kendalltau, rankdata

    # Kendall tau of the official VideoXum evaluation, as (min, max, mean) over the repeats.
    pred_np = pred.sigmoid().cpu().numpy()
    gt_np = gt.cpu().numpy()
    mask_np = mask.cpu().numpy()
    batch = pred_np.shape[0]
    scores = np.zeros(batch)
    for i in range(batch):
        gt_i, pred_i = get_frame_scores(pred_np[i], gt_np[i], mask_np[i])
        # Undefined correlations (constant inputs) count as 0.
        tau = kendalltau(rankdata(-gt_i), rankdata(-pred_i))[0]
        scores[i] = 0.0 if math.isnan(tau) else tau
    scores = scores.reshape(-1, num_repeat)
    return (torch.from_numpy(np.min(scores, axis=1)), torch.from_numpy(np.max(scores, axis=1)),
            torch.from_numpy(np.mean(scores, axis=1)))


def compute_spearman(pred, gt, mask, num_repeat=10):
    from scipy.stats import spearmanr

    # Spearman rho of the official VideoXum evaluation, as (min, max, mean) over the repeats.
    pred_np = pred.sigmoid().cpu().numpy()
    gt_np = gt.cpu().numpy()
    mask_np = mask.cpu().numpy()
    batch = pred_np.shape[0]
    scores = np.zeros(batch)
    for i in range(batch):
        gt_i, pred_i = get_frame_scores(pred_np[i], gt_np[i], mask_np[i])
        # Undefined correlations (constant inputs) count as 0.
        rho = spearmanr(gt_i, pred_i)[0]
        scores[i] = 0.0 if math.isnan(rho) else rho
    scores = scores.reshape(-1, num_repeat)
    return (torch.from_numpy(np.min(scores, axis=1)), torch.from_numpy(np.max(scores, axis=1)),
            torch.from_numpy(np.mean(scores, axis=1)))


def get_top15_frames(pred, gt, mask):
    # Binary prediction of the top 15% scores (official evalVtV).
    valid = pred[mask == 1]
    top15 = int(len(valid) * 0.15)
    threshold = np.sort(valid)[::-1][int(top15)]
    return gt[mask == 1], (valid >= threshold).astype(np.float32)


def compute_f1(pred, gt, mask, num_repeat=10):
    from sklearn.metrics import f1_score

    # F1 of the top-15% frames of the official VideoXum evaluation, as (min, max, mean) over the
    # repeats.
    pred_np = pred.cpu().numpy()
    gt_np = gt.cpu().numpy()
    mask_np = mask.cpu().numpy()
    batch = pred_np.shape[0]
    f1 = np.zeros(batch)
    for i in range(batch):
        gt_i, pred_i = get_top15_frames(pred_np[i], gt_np[i], mask_np[i])
        f1[i] = f1_score(gt_i, pred_i)
    f1 = f1.reshape(-1, num_repeat)
    return (torch.from_numpy(np.min(f1, axis=1)), torch.from_numpy(np.max(f1, axis=1)),
            torch.from_numpy(np.mean(f1, axis=1)))


def binary_f1(prediction, reference):
    # Frame-level F1 of two binary vectors, equal to sklearn's (0 when both are empty).
    true_positive = sum(p == 1 and r == 1 for p, r in zip(prediction, reference))
    denominator = sum(prediction) + sum(reference)
    return 2.0 * true_positive / denominator if denominator else 0.0


def chance_corrected_metrics(prediction, references):
    # Metrics that do not improve by selecting more frames at random.
    frame_count = len(prediction)
    selected = sum(prediction)
    kappas = []
    # kappa: mean Cohen's kappa against each annotator (0 = chance).
    for reference in references:
        positives = sum(reference)
        true_positive = sum(p and r for p, r in zip(prediction, reference))
        true_negative = frame_count - selected - positives + true_positive
        observed = (true_positive + true_negative) / frame_count
        expected = (
            selected * positives + (frame_count - selected) * (frame_count - positives)
        ) / frame_count ** 2
        kappas.append((observed - expected) / (1 - expected) if expected < 1 else 0.0)
    # lift: mean saliency of the selected frames over the mean saliency of the video (1 = chance, 0
    # for empty selections).
    saliency = [sum(column) / len(references) for column in zip(*references)]
    mean_saliency = sum(saliency) / frame_count
    if selected and mean_saliency:
        selected_saliency = sum(s for s, p in zip(saliency, prediction) if p) / selected
        lift = selected_saliency / mean_saliency
    else:
        lift = 0.0
    return {"kappa": sum(kappas) / len(kappas), "lift": lift}


def videoxum_metrics(prediction, references):
    # F1 against every annotator (max and mean).
    f1_scores = [binary_f1(prediction, reference) for reference in references]
    prediction_tensor = torch.tensor(prediction, dtype=torch.float32).unsqueeze(0)
    references_tensor = torch.tensor(references, dtype=torch.float32)
    # Kendall and Spearman against the mean saliency of the annotators.
    mean_saliency = references_tensor.mean(dim=0, keepdim=True)
    mask = torch.ones_like(prediction_tensor)
    _, _, kendall_mean = compute_kendall(prediction_tensor, mean_saliency, mask, num_repeat=1)
    _, _, spearman_mean = compute_spearman(prediction_tensor, mean_saliency, mask, num_repeat=1)
    return {
        "f1_max": max(f1_scores),
        "f1_mean": sum(f1_scores) / len(f1_scores),
        "kendall": kendall_mean.item(),
        "spearman": spearman_mean.item(),
        **chance_corrected_metrics(prediction, references),
    }


def format_selection_summary(metrics, samples, invalid_predictions):
    # One-line log of the mean selection metrics.
    return (
        f"[val] VideoXum F1_MAX={metrics['f1_max']:.4f} "
        f"F1_MEAN={metrics['f1_mean']:.4f} "
        f"Kendall={metrics['kendall']:.4f} "
        f"Spearman={metrics['spearman']:.4f} "
        f"Kappa={metrics['kappa']:.4f} "
        f"Lift={metrics['lift']:.4f} "
        f"valid_predictions={samples - invalid_predictions}/{samples}"
    )


def decile_profile(vectors, deciles=10):
    # Mean selected density in every decile of the timeline.
    totals = [0.0] * deciles
    for vector in vectors:
        length = len(vector)
        for decile in range(deciles):
            low = decile * length // deciles
            high = max((decile + 1) * length // deciles, low + 1)
            totals[decile] += sum(vector[low:high]) / (high - low)
    return [total / len(vectors) for total in totals]


def collapse_metrics(responses, predictions, references_list, seed=0):
    # How much the output depends on the video rather than on the number of frames alone.
    total = len(predictions)
    if total == 0:
        return {}
    # Identical responses up to whitespace count as the same output.
    compact = [re.sub(r"\s+", "", response) for response in responses]
    nonempty = [prediction for prediction in predictions if any(prediction)]
    metrics = {
        "distinct_output_ratio": len(set(compact)) / total,
        "top4_output_share": sum(c for _, c in Counter(compact).most_common(4)) / total,
        "empty_prediction_rate": 1.0 - len(nonempty) / total,
    }
    if not nonempty:
        return metrics

    # Structural statistics use non-empty predictions only.
    runs = [selected_runs(prediction) for prediction in nonempty]
    prefixes = Counter(tuple(run[:2]) for run in runs)
    starts = Counter(run[0][0] for run in runs)
    lengths = Counter(end - start + 1 for run in runs for start, end in run)
    interval_count = sum(lengths.values())

    # Cross-video IoU between predictions of videos with the same length; cross_video_iou_null is
    # the expected IoU of two independent selections with the same densities.
    by_length = defaultdict(list)
    for position, prediction in enumerate(nonempty):
        by_length[len(prediction)].append(position)
    rng = random.Random(seed)
    ious, nulls = [], []
    for positions in by_length.values():
        if len(positions) < 2:
            continue
        for _ in range(min(50, len(positions) * (len(positions) - 1) // 2)):
            first, second = (nonempty[p] for p in rng.sample(positions, 2))
            union = sum(a | b for a, b in zip(first, second))
            ious.append(sum(a & b for a, b in zip(first, second)) / union)
            density_a = sum(first) / len(first)
            density_b = sum(second) / len(second)
            nulls.append(density_a * density_b / (density_a + density_b - density_a * density_b))

    # Decile profiles of the predictions and of the annotator saliency.
    saliency = [
        [sum(column) / len(references) for column in zip(*references)]
        for references in references_list
    ]
    predicted_profile = decile_profile(predictions)
    reference_profile = decile_profile(saliency)
    metrics.update({
        "prefix2_share": prefixes.most_common(1)[0][1] / len(nonempty),
        "first_start_mode_share": starts.most_common(1)[0][1] / len(nonempty),
        "first_start_top5": [[start, count / len(nonempty)] for start, count in starts.most_common(5)],
        "mean_interval_length": sum(k * c for k, c in lengths.items()) / interval_count,
        "modal_interval_length_share": lengths.most_common(1)[0][1] / interval_count,
        "mean_density": sum(sum(p) / len(p) for p in nonempty) / len(nonempty),
        "cross_video_iou": sum(ious) / len(ious) if ious else None,
        "cross_video_iou_null": sum(nulls) / len(nulls) if nulls else None,
        "decile_profile_pred": predicted_profile,
        "decile_profile_gt": reference_profile,
        "decile_profile_l1": sum(abs(a - b) for a, b in zip(predicted_profile, reference_profile)),
    })
    return metrics


def format_collapse_summary(metrics):
    # One-line log of the main collapse statistics.
    keys = (
        "distinct_output_ratio", "prefix2_share", "first_start_mode_share",
        "modal_interval_length_share", "cross_video_iou", "cross_video_iou_null",
    )
    return " ".join(f"{key}={metrics[key]:.3f}" for key in keys if metrics.get(key) is not None)


def finite(value):
    # None instead of NaN/inf, so that JSONL logs stay valid JSON.
    value = float(value)
    return value if math.isfinite(value) else None


def compute_meteor_scores(candidates, references):
    from nltk.translate.meteor_score import meteor_score

    # NLTK METEOR (stemming + WordNet synonyms) of each candidate/reference pair.
    return [meteor_score([reference.split()], candidate.split())
            for candidate, reference in zip(candidates, references)]


def compute_corpus_bleu(candidates, references):
    from nltk.translate.bleu_score import corpus_bleu

    # Corpus-level BLEU@1-4 with split() tokenization, comparable across our runs only.
    tokenized_candidates = [candidate.split() for candidate in candidates]
    tokenized_references = [[reference.split()] for reference in references]
    scores = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for name, weights in BLEU_WEIGHTS.items():
            score = float(corpus_bleu(tokenized_references, tokenized_candidates, weights=weights))
            scores[name] = 0.0 if score < 1e-12 else score
    return scores


def compute_sentence_bleu_scores(candidates, references, weights=BLEU_WEIGHTS["bleu4"]):
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    # Per-sample BLEU with smoothing method1: without it short texts score 0.
    smoothing = SmoothingFunction().method1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        scores = [
            float(sentence_bleu([reference.split()], candidate.split(),
                                weights=weights, smoothing_function=smoothing))
            for candidate, reference in zip(candidates, references)
        ]
    return [0.0 if score < 1e-12 else score for score in scores]


def compute_cider_score(candidates, references):
    global _cider_warned
    # Corpus-level CIDEr-D: (mean, per-sample scores). The TF-IDF weights depend on log(N), so
    # values are not comparable across sets of different size, and fewer than 2 samples give NaN.
    if len(candidates) < 2:
        if not _cider_warned:
            print("[warning] CIDEr needs at least 2 samples (corpus-level): value -> NaN")
            _cider_warned = True
        return float("nan"), [float("nan")] * len(candidates)
    from pycocoevalcap.cider.cider import Cider

    gts = {i: [reference] for i, reference in enumerate(references)}
    res = {i: [candidate] for i, candidate in enumerate(candidates)}
    mean_score, per_sample = Cider().compute_score(gts, res)
    return float(mean_score), [float(score) for score in per_sample]


def rouge_scorer():
    global _rouge_scorer
    # ROUGE-1/ROUGE-L scorer with stemming, built once and shared.
    if _rouge_scorer is None:
        from rouge_score import rouge_scorer as rouge
        _rouge_scorer = rouge.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    return _rouge_scorer


def summary_metrics(candidate, reference):
    # ROUGE-1/L F1, sentence BLEU@4 and word count of one summary; an empty summary scores 0.
    scores = rouge_scorer().score(reference, candidate)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
        "bleu4_sentence": compute_sentence_bleu_scores([candidate], [reference])[0],
        "summary_words": float(len(candidate.split())),
    }


def distinct_summary_ratio(candidates):
    # Share of distinct summaries: a template summary repeated on every video gives ~0.
    return len({" ".join(candidate.split()) for candidate in candidates}) / len(candidates)


def corpus_text_metrics(candidates, references):
    if not candidates:
        return {}
    # Corpus BLEU@1-4, CIDEr and distinct_summary_ratio of a set of summaries.
    values = {name: finite(value) for name, value in compute_corpus_bleu(candidates, references).items()}
    values["cider"] = finite(compute_cider_score(candidates, references)[0])
    values["distinct_summary_ratio"] = distinct_summary_ratio(candidates)
    return values

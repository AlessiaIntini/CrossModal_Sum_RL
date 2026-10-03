"""Reward of the joint task (frame selection + summary) used by GRPO.

Static data of each video (annotations, event masks, frame features, caption embeddings) is
prepared once by VideoSummaryReward and reused for every rollout of that video.
"""

import json
import math
import re
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .parsing import (
    CompletionValidationError, expand_pair_completion, is_integer, parse_and_validate_completion,
    strict_json_loads, strip_code_fence,
)

EPSILON = 1e-8
# Weights of the reward components. A zero weight removes a component from the training reward, but
# it is still computed and logged. GRPO follows SFT and DPO, so most weight goes to which frames are
# selected (f_beta) and to the summary (grounding).
DEFAULT_REWARD_WEIGHTS = {
    "f_beta": 0.30,
    "diversity": 0.00,
    "representativeness": 0.10,
    "grounding": 0.25,
    "reference": 0.05,
    "format": 0.05,
    "summary_length": 0.10,
    "length_budget": 0.05,
    "saliency_rank": 0.00,
}
# Standard deviation of the log ratio summary words / reference words: ratio 0.7 or 1.4 gives ~0.6,
# ratio 0.5 or 2.0 gives ~0.14.
SUMMARY_LENGTH_SIGMA = 0.35
# Same form on selected frames / budget, tuned independently of the text length.
LENGTH_BUDGET_SIGMA = 0.35
# Affine rescaling of the summary-reference cosine to [0, 1], clamp((cos - floor) / scale); the
# defaults give (cos + 1) / 2.
REFERENCE_COSINE_FLOOR = -1.0
REFERENCE_COSINE_SCALE = 2.0
F_BETA_AGGREGATIONS = ("mean", "max")


class SentenceTransformerEncoder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device=None):
        from sentence_transformers import SentenceTransformer

        # Sentence embeddings of captions and summaries, used by the grounding and reference
        # components.
        self.model = SentenceTransformer(model_name, device=device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts):
        # Float tensor [len(texts), dim] of L2-normalized embeddings, computed without gradients.
        values = self.model.encode(list(texts), convert_to_tensor=True, normalize_embeddings=True)
        return values.detach().float()


@dataclass
class VideoRewardCache:
    # Per-video data shared by all the rollouts of a GRPO group.
    count: int
    max_selected: int
    vsum_onehot: torch.Tensor
    event_masks: torch.Tensor
    frame_features: torch.Tensor
    caption_embeddings: torch.Tensor
    reference_embedding: torch.Tensor
    consensus: torch.Tensor
    captions: tuple
    reference_summary: str
    frame_floor_sim: float
    frame_min_sim: float
    consensus_ranks: torch.Tensor


def compute_format_reward(completion, count, max_selected):
    # Graded format credit in four steps of 0.25. Step 1: valid JSON.
    try:
        payload = strict_json_loads(strip_code_fence(completion)) if isinstance(completion, str) else None
    except (json.JSONDecodeError, ValueError):
        return 0.0
    reward = 0.25
    if not isinstance(payload, dict):
        return reward
    intervals = payload.get("selected_frames")
    # Step 2: a non-empty list of [start, end] pairs.
    structural = (
        isinstance(intervals, list)
        and bool(intervals)
        and all(isinstance(item, list) and len(item) == 2 for item in intervals)
    )
    if structural:
        reward += 0.25
    # Step 3: integer indices, sorted, disjoint, inside the timeline and within the budget.
    indices_valid = structural
    selected_count, previous_end = 0, -2
    if structural:
        for start, end in intervals:
            if (not is_integer(start) or not is_integer(end) or not 0 <= start <= end < count
                    or start <= previous_end):
                indices_valid = False
                break
            selected_count += end - start + 1
            previous_end = end
        indices_valid = indices_valid and selected_count <= max_selected
    if indices_valid:
        reward += 0.25
    # Step 4: a non-empty summary.
    summary = payload.get("summary")
    if isinstance(summary, str) and bool(summary.strip()):
        reward += 0.25
    return reward


def log_ratio_gaussian(ratio, sigma):
    # exp(-log(ratio)^2 / 2 sigma^2): 1 at ratio 1, symmetric between half and double.
    if ratio <= 0.0:
        return 0.0
    return math.exp(-(math.log(ratio) ** 2) / (2.0 * sigma ** 2))


def compute_summary_length_reward(summary, reference, sigma=SUMMARY_LENGTH_SIGMA, target_words=None):
    # Log-ratio Gaussian of the summary length against the reference (or a fixed target number of
    # words).
    words = len(summary.split())
    reference_words = target_words if target_words is not None else len(reference.split())
    if words == 0 or reference_words <= 0:
        return 0.0
    return log_ratio_gaussian(words / reference_words, sigma)


def compute_length_budget_reward(selected, budget, sigma=LENGTH_BUDGET_SIGMA):
    # Log-ratio Gaussian of the selected frames against the budget, so selecting too few frames is
    # penalized as well.
    if selected <= 0 or budget <= 0:
        return 0.0
    return log_ratio_gaussian(selected / budget, sigma)


def average_ranks(values):
    # 1-based ranks with ties at their mean rank, as scipy.stats.rankdata.
    values = values.detach().double().flatten()
    sorted_values, order = torch.sort(values, stable=True)
    _, inverse, counts = torch.unique_consecutive(sorted_values, return_inverse=True, return_counts=True)
    counts = counts.double()
    mean_rank = counts.cumsum(0) - (counts - 1.0) / 2.0
    ranks = torch.empty_like(values)
    ranks[order] = mean_rank[inverse]
    return ranks


def compute_saliency_rank_reward(pred, consensus_ranks):
    # Spearman correlation between the binary selection and the mean annotator saliency (the
    # Spearman of the VideoXum evaluation), mapped to [0, 1] as (rho + 1) / 2.
    mask = pred.detach().double().flatten()
    mask = mask - mask.mean()
    ranks = consensus_ranks - consensus_ranks.mean()
    denominator = torch.sqrt((mask * mask).sum() * (ranks * ranks).sum())
    # A constant selection or saliency has no correlation.
    if denominator.item() <= EPSILON:
        return 0.5
    rho = ((mask * ranks).sum() / denominator).clamp(-1.0, 1.0).item()
    return (rho + 1.0) / 2.0


def split_into_sentences(text):
    # Deterministic sentence splitter on end punctuation and line breaks.
    claims = [part.strip() for part in re.split(r"(?<=[.!?])\s+|[\r\n]+", text)]
    return [claim for claim in claims if claim]


def event_masks(vsum, count):
    # vsum holds one [start, end] interval per caption and annotator; the result is a mask
    # [captions, 10, count] of the frames inside each interval.
    intervals = torch.as_tensor(vsum, dtype=torch.float64)
    if intervals.ndim != 3 or intervals.shape[1:] != (10, 2):
        raise ValueError("vsum must have shape [J, 10, 2]")
    if not torch.isfinite(intervals).all():
        raise ValueError("vsum contains non-finite values")
    if (intervals[..., 0] > intervals[..., 1]).any():
        raise ValueError("vsum contains inverted intervals")
    positions = torch.arange(count, dtype=torch.float64).view(1, 1, count)
    return (positions >= intervals[..., 0].unsqueeze(-1)) & (positions <= intervals[..., 1].unsqueeze(-1))


def normalized_embeddings(values, expected_rows):
    # L2-normalized float32 embeddings with the expected number of rows.
    tensor = torch.as_tensor(values, dtype=torch.float32).detach()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2 or tensor.shape[0] != expected_rows:
        raise ValueError(f"semantic encoder: expected [{expected_rows}, D], got {tuple(tensor.shape)}")
    return F.normalize(tensor, dim=-1, eps=EPSILON)


class VideoSummaryReward:
    def __init__(
        self,
        semantic_encoder,
        *,
        f_beta=0.5,
        f_beta_aggregation="max",
        weights=None,
        use_length_penalty=False,
        summary_length_sigma=SUMMARY_LENGTH_SIGMA,
        summary_target_words=None,
        length_budget_sigma=LENGTH_BUDGET_SIGMA,
        reference_cosine_floor=REFERENCE_COSINE_FLOOR,
        reference_cosine_scale=REFERENCE_COSINE_SCALE,
        timeline_unit="frame",
    ):
        # Unit of the indices in the completions; the cache and every component always stay on
        # frames.
        if timeline_unit not in ("frame", "pair"):
            raise ValueError(f"invalid timeline_unit: {timeline_unit}")
        self.timeline_unit = timeline_unit
        self.semantic_encoder = semantic_encoder
        self.weights = {**DEFAULT_REWARD_WEIGHTS, **(weights or {})}
        # Unknown, negative or all-zero weights are configuration errors.
        if unknown := set(self.weights) - set(DEFAULT_REWARD_WEIGHTS):
            raise ValueError(f"unknown reward weights: {sorted(unknown)}")
        if any(value < 0 for value in self.weights.values()):
            raise ValueError("reward weights must be >= 0")
        if not any(self.weights.values()):
            raise ValueError("at least one reward weight must be > 0")
        if f_beta_aggregation not in F_BETA_AGGREGATIONS:
            raise ValueError(f"invalid f_beta_aggregation: {f_beta_aggregation} "
                             f"(expected one of {F_BETA_AGGREGATIONS})")
        self.f_beta_aggregation = f_beta_aggregation
        self.use_length_penalty = bool(use_length_penalty)
        if summary_length_sigma <= 0:
            raise ValueError("summary_length_sigma must be positive")
        self.summary_length_sigma = float(summary_length_sigma)
        if summary_target_words is not None and summary_target_words <= 0:
            raise ValueError("summary_target_words must be positive")
        self.summary_target_words = int(summary_target_words) if summary_target_words is not None else None
        if length_budget_sigma <= 0:
            raise ValueError("length_budget_sigma must be positive")
        self.length_budget_sigma = float(length_budget_sigma)
        if reference_cosine_scale <= 0:
            raise ValueError("reference_cosine_scale must be positive")
        self.reference_cosine_floor = float(reference_cosine_floor)
        self.reference_cosine_scale = float(reference_cosine_scale)
        if f_beta <= 0:
            raise ValueError("f_beta must be positive")
        self.f_beta = float(f_beta)
        self._cache = {}

    def has_video(self, cache_key):
        # True if the video was already prepared.
        return cache_key in self._cache

    @torch.no_grad()
    def prepare_video(self, cache_key, *, count, vsum_onehot, vsum, tsum, frame_features):
        # The static data of a video (annotations, event masks, frame features, caption embeddings)
        # are validated and cached once, for every rollout of that video.
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if cached.count != count:
                raise ValueError("cache_key reused for a video with a different count")
            return cached
        if count <= 0:
            raise ValueError("count must be positive")
        gt = torch.as_tensor(vsum_onehot).bool().detach().cpu()
        if gt.shape != (10, count):
            raise ValueError(f"vsum_onehot must have shape [10, {count}]")
        captions = tuple(str(item).strip() for item in tsum)
        if not captions or any(not item for item in captions):
            raise ValueError("tsum must contain non-empty captions")
        masks = torch.as_tensor(event_masks(vsum, count), dtype=torch.bool).detach().cpu()
        if masks.shape != (len(captions), 10, count):
            raise ValueError(f"event masks must have shape [{len(captions)}, 10, {count}]")
        visual = torch.as_tensor(frame_features, dtype=torch.float32).detach().cpu()
        if visual.ndim != 2 or visual.shape[0] != count:
            raise ValueError(f"frame_features must have shape [{count}, D]")
        visual = F.normalize(visual, dim=-1, eps=EPSILON)
        # Redundancy baseline of the video: mean and minimum similarity between two different
        # frames. Diversity and representativeness are calibrated on it, because frames of the same
        # video are already very similar without any targeted selection.
        if count >= 2:
            similarity = visual @ visual.T
            off_diagonal = similarity[~torch.eye(count, dtype=torch.bool)]
            frame_floor_sim, frame_min_sim = off_diagonal.mean().item(), off_diagonal.min().item()
        else:
            frame_floor_sim, frame_min_sim = 0.0, -1.0
        reference_summary = " ".join(captions)
        encoded = self.semantic_encoder.encode([*captions, reference_summary])
        encoded = normalized_embeddings(encoded, len(captions) + 1).cpu()
        # Frame budget: the larger of 20% of the video (the share announced in the prompt) and 1.3x
        # the words of the reference summary. The floor keeps the budget at least as large as
        # announced, because the model cannot see the reference length.
        max_selected = max(1, min(count, max(
            math.ceil(0.20 * count), math.ceil(1.3 * len(reference_summary.split())),
        )))
        cache = VideoRewardCache(
            count=count,
            max_selected=max_selected,
            vsum_onehot=gt,
            event_masks=masks,
            frame_features=visual,
            caption_embeddings=encoded[:-1],
            reference_embedding=encoded[-1],
            consensus=gt.float().mean(dim=0),
            captions=captions,
            reference_summary=reference_summary,
            frame_floor_sim=frame_floor_sim,
            frame_min_sim=frame_min_sim,
            consensus_ranks=average_ranks(gt.float().mean(dim=0)),
        )
        self._cache[cache_key] = cache
        return cache

    def timeline_limits(self, data):
        # (number of indices, selection budget) in the unit of the completion.
        if self.timeline_unit == "pair":
            return (data.count + 1) // 2, max(1, math.ceil(data.max_selected / 2))
        return data.count, data.max_selected

    @staticmethod
    def _invalid_result(format_reward):
        # Invalid responses get format_reward - 1, below every valid response, and 0 for every other
        # component.
        return {
            "total_reward": float(format_reward - 1.0),
            "total_reward_default": float(format_reward - 1.0),
            "format_reward": float(format_reward),
            "f_beta_reward": 0.0,
            "diversity_reward": 0.0,
            "representativeness_reward": 0.0,
            "grounding_reward": 0.0,
            "grounded_coverage": 0.0,
            "grounded_precision": 0.0,
            "reference_reward": 0.0,
            "summary_length_reward": 0.0,
            "length_budget_reward": 0.0,
            "saliency_rank_reward": 0.0,
            "summary_words": 0,
            "selected_count": 0,
            "selected_ratio": 0.0,
            "selection_budget": 0,
            "length_penalty": 0.0,
            "valid": False,
            "merged_adjacent_count": 0,
            "f_beta_max": 0.0,
            "f_beta_min": 0.0,
            "f_beta_std": 0.0,
            "f_beta_per_annotator": [],
        }

    @torch.no_grad()
    def score(self, completion, cache_key):
        # Total reward, components and diagnostics of one rollout.
        if cache_key not in self._cache:
            raise KeyError(f"video {cache_key!r} is not in the cache")
        data = self._cache[cache_key]
        timeline_count, timeline_budget = self.timeline_limits(data)
        format_reward = compute_format_reward(completion, timeline_count, timeline_budget)
        try:
            parsed = parse_and_validate_completion(completion, timeline_count, timeline_budget)
        except CompletionValidationError:
            return self._invalid_result(format_reward)
        # Pair indices are mapped back to frames before scoring.
        if self.timeline_unit == "pair":
            parsed = expand_pair_completion(parsed, data.count)
        # Adjacent intervals are a syntactic error only (the selected frames do not change): a small
        # penalty instead of rejecting the response, so the other components keep their variance
        # within the group.
        if parsed.merged_adjacent_count:
            format_reward = max(0.0, format_reward - 0.1 * parsed.merged_adjacent_count)

        pred = parsed.pred_mask.cpu()
        pred_count = int(pred.sum().item())
        # F-beta against every annotator, (1 + beta^2) TP / (beta^2 GT + selected). beta < 1 weighs
        # precision more than recall: frames outside the annotated intervals cost more than the
        # recall they add.
        beta2 = self.f_beta ** 2
        intersections = (data.vsum_onehot & pred.unsqueeze(0)).sum(dim=1).float()
        gt_counts = data.vsum_onehot.sum(dim=1).float()
        f_beta_denominators = beta2 * gt_counts + pred_count
        f_beta_values = torch.where(
            f_beta_denominators > 0, (1.0 + beta2) * intersections / f_beta_denominators, 0.0,
        )
        # Max over the annotators by default: annotators disagree, the mean flattens the differences
        # between rollouts, and the max matches the F1_MAX validation metric.
        f_beta_reward = (
            f_beta_values.max() if self.f_beta_aggregation == "max" else f_beta_values.mean()
        ).item()

        # Diversity between the mean features of the selected intervals: 0 when two intervals are as
        # similar as a typical pair of frames of the video, 1 when as dissimilar as its most
        # dissimilar pair.
        prototypes = []
        for start, end in parsed.selected_frames:
            prototype = data.frame_features[start:end + 1].mean(dim=0)
            prototypes.append(F.normalize(prototype, dim=0, eps=EPSILON))
        if len(prototypes) < 2:
            diversity_reward = 0.0
        else:
            group_features = torch.stack(prototypes)
            similarity = group_features @ group_features.T
            pairs = torch.triu_indices(len(prototypes), len(prototypes), offset=1)
            raw_pair_sim = similarity[pairs[0], pairs[1]]
            diversity_denominator = max(data.frame_floor_sim - data.frame_min_sim, 0.05)
            diversity_reward = (
                (data.frame_floor_sim - raw_pair_sim) / diversity_denominator
            ).clamp(0.0, 1.0).mean().item()

        # Representativeness: every frame is covered by its most similar selected frame, 0 if no
        # more similar than the typical frame pair and 1 for an exact match; frames are weighted by
        # the annotator saliency.
        selected_features = data.frame_features[parsed.selected_indices]
        raw_coverage = (data.frame_features @ selected_features.T).max(dim=1).values
        representativeness_denominator = max(1.0 - data.frame_floor_sim, 0.05)
        coverage = ((raw_coverage - data.frame_floor_sim) / representativeness_denominator).clamp(0.0, 1.0)
        representativeness_reward = (
            (data.consensus * coverage).sum() / (data.consensus.sum() + EPSILON)
        ).item()

        # Grounding. Visual support of every caption: the fraction of its event frames that are
        # selected, averaged over the annotators.
        event_intersections = (data.event_masks & pred.view(1, 1, -1)).sum(dim=-1).float()
        event_sizes = data.event_masks.sum(dim=-1).float()
        visual_support = (event_intersections / (event_sizes + EPSILON)).mean(dim=1)
        # Semantic match between the captions and the sentences of the summary; a single-sentence
        # summary is encoded once.
        claims = split_into_sentences(parsed.summary)
        online_texts = [parsed.summary] if claims == [parsed.summary] else [*claims, parsed.summary]
        online = normalized_embeddings(self.semantic_encoder.encode(online_texts), len(online_texts)).cpu()
        claim_embeddings = online[:len(claims)]
        summary_embedding = online[-1]
        semantic_similarity = ((data.caption_embeddings @ claim_embeddings.T + 1.0) / 2.0).clamp(0.0, 1.0)
        text_match = semantic_similarity.max(dim=1).values
        # Coverage: captions that are both visually selected and described; precision: summary
        # sentences that match a selected event. The grounding reward is their harmonic mean.
        grounded_coverage = (visual_support * text_match).mean().item()
        grounded_precision = (visual_support.unsqueeze(1) * semantic_similarity).max(dim=0).values.mean().item()
        grounding_reward = min(1.0, max(0.0, (
            2.0 * grounded_coverage * grounded_precision / (grounded_coverage + grounded_precision + EPSILON)
        )))

        # Cosine similarity to the reference summary (embeddings are normalized), rescaled to [0, 1].
        reference_reward = float((
            (summary_embedding @ data.reference_embedding - self.reference_cosine_floor)
            / self.reference_cosine_scale
        ).clamp(0.0, 1.0).item())
        # Embedding similarities barely depend on the summary length, while CIDEr-D, ROUGE and
        # METEOR do: without this term GRPO converges to summaries much shorter than the references.
        summary_length_reward = compute_summary_length_reward(
            parsed.summary, data.reference_summary, self.summary_length_sigma, self.summary_target_words,
        )
        selection_ratio = pred_count / data.max_selected
        # The frame budget is an additive component: it concerns the selection only, so it must not
        # scale the text components.
        length_budget_reward = compute_length_budget_reward(pred_count, data.max_selected, self.length_budget_sigma)
        # Legacy multiplicative penalty, capped at 0.85: always logged, applied only with
        # use_length_penalty.
        length_penalty = min(0.85, (selection_ratio ** 2) / 2.0)
        saliency_rank_reward = compute_saliency_rank_reward(pred, data.consensus_ranks)

        components = {
            "f_beta": f_beta_reward,
            "diversity": diversity_reward,
            "representativeness": representativeness_reward,
            "grounding": grounding_reward,
            "reference": reference_reward,
            "format": format_reward,
            "summary_length": summary_length_reward,
            "length_budget": length_budget_reward,
            "saliency_rank": saliency_rank_reward,
        }
        # total_reward uses the experiment weights and drives the GRPO advantage;
        # total_reward_default always uses DEFAULT_REWARD_WEIGHTS, so validation stays comparable
        # across ablations. Both are normalized by the sum of their weights.
        weight_sum = sum(self.weights.values()) or 1.0
        default_sum = sum(DEFAULT_REWARD_WEIGHTS.values()) or 1.0
        weighted_reward = sum(self.weights[k] * v for k, v in components.items()) / weight_sum
        default_reward = sum(DEFAULT_REWARD_WEIGHTS[k] * v for k, v in components.items()) / default_sum
        applied_penalty = length_penalty if self.use_length_penalty else 0.0
        total_reward = weighted_reward * (1.0 - applied_penalty)
        total_reward_default = default_reward * (1.0 - applied_penalty)
        return {
            "total_reward": float(min(1.0, max(0.0, total_reward))),
            "total_reward_default": float(min(1.0, max(0.0, total_reward_default))),
            "format_reward": float(format_reward),
            "f_beta_reward": float(f_beta_reward),
            "diversity_reward": float(diversity_reward),
            "representativeness_reward": float(representativeness_reward),
            "grounding_reward": float(grounding_reward),
            "grounded_coverage": float(grounded_coverage),
            "grounded_precision": float(grounded_precision),
            "reference_reward": float(reference_reward),
            "summary_length_reward": float(summary_length_reward),
            "length_budget_reward": float(length_budget_reward),
            "saliency_rank_reward": float(saliency_rank_reward),
            "summary_words": len(parsed.summary.split()),
            "reference_words": len(data.reference_summary.split()),
            "selected_count": pred_count,
            "selected_ratio": pred_count / data.count,
            "selection_budget": data.max_selected,
            "length_penalty": float(length_penalty),
            "valid": True,
            "merged_adjacent_count": parsed.merged_adjacent_count,
            "f_beta_max": float(f_beta_values.max().item()),
            "f_beta_min": float(f_beta_values.min().item()),
            "f_beta_std": float(f_beta_values.std(unbiased=False).item()),
            "f_beta_per_annotator": [float(value) for value in f_beta_values.tolist()],
        }


def ensure_video_prepared(reward, sample):
    # Features are loaded and the reward cache is prepared once per video.
    if reward.has_video(sample["cache_key"]):
        return
    with np.load(sample["feature_path"]) as archive:
        features = archive["features"] if "features" in archive else archive[archive.files[0]]
    # Feature files may contain one extra trailing frame beyond the 1 fps grid: it is dropped.
    if features.ndim == 2 and features.shape[0] == sample["count"] + 1:
        features = features[:sample["count"]]
    reward.prepare_video(sample["cache_key"], count=sample["count"], vsum_onehot=sample["vsum_onehot"],
                         vsum=sample["vsum"], tsum=sample["tsum"], frame_features=features)

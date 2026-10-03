#!/usr/bin/env python3
"""Consensus targets and DPO preference pairs for the VideoXum training set.

Every training video has 10 binary annotations at 1 fps (vsum_onehot). This script adds to
every record a single consensus target, used as SFT target and as DPO chosen, and a rejected
selection that is worse than it.

Consensus target. The saliency of a frame is the fraction of annotators that selected it.
The target selects exactly `budget` indices, the median annotation length, and maximizes the
summed saliency minus a penalty for every interval it opens. The optimum is found exactly by
dynamic programming. If it has more than --max-intervals intervals, the penalty is doubled
and the optimization is repeated.

On the pair timeline (--timeline-unit pair, matching the temporal patches of Qwen3-VL) the
same optimization runs on segments of two consecutive frames:

- the saliency of a segment is the mean of its two frames;
- the budget is halved;
- the default penalty is halved, so that the selection is as fragmented as on frames.

Rejected selections. Every video gets one rejected selection of one of three kinds, in the
proportions --hard-ratio / --intermediate-ratio / --overcoverage-ratio:

- hard: the most shared event of the chosen moves to an unselected window of the same
  length. The most shared event is the chosen interval with the highest mean number of
  votes; ties go to the shortest. Among the windows whose mean F1 against the annotators is
  at least 0.05 below the chosen, the highest-F1 one is kept. Only videos with at least two
  chosen intervals and such a window are eligible.
- intermediate: the chosen shifts left or right without changing its size. The shift is 5%
  of the timeline, between 2 and --max-shift indices. It is kept only if the F1 drops by at
  least 0.05.
- overcoverage: the chosen plus the unselected indices nearest to it, until a coverage drawn
  from the 30-40%, 40-60% or 70-100% band (--overcoverage-band-ratios). Its F1 may even be
  higher: this pair teaches conciseness.

Hard videos are drawn first among the eligible ones, intermediate videos among the others
with a valid shift, and the remaining videos get overcoverage. Every random choice is seeded
with sha256(seed:video id), so the output is deterministic. On the pair timeline, rejected
selections are built and scored against pair-level annotations, where a segment counts as
selected if one of its frames is. vsum_onehot itself stays on frames.

Output fields: vsum_onehot_consensus (the target), vsum_onehot_chosen and
vsum_onehot_rejected (the DPO pair), dpo_rejected_difficulty (hard, intermediate or
overcoverage) and the F1 scores and margin of the pair. The temporal medoid of the
annotations (vsum_onehot_medoid) is also stored, only as a reference for the consensus.

Example:
  python configuration/create_consensus_dpo_data.py --input .../metadata/train_videoxum.json \
      --output .../consensus/train_videoxum_dpo_consensus_pairs.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import tempfile
from pathlib import Path
from typing import Any, Sequence

VIDEOXUM_ROOT = Path(__file__).resolve().parents[1] / "dataset" / "aintini" / "datasets" / "videoxum"
DEFAULT_INPUT = VIDEOXUM_ROOT / "metadata" / "train_videoxum.json"
DEFAULT_OUTPUT = VIDEOXUM_ROOT / "consensus" / "train_videoxum_dpo_consensus_pairs.json"
TIMELINE_UNITS = ("frame", "pair")
UNIT_DEFAULTS = {
    "frame": {"interval_penalty": 0.15, "max_shift": 8},
    "pair": {"interval_penalty": 0.075, "max_shift": 4},
}
OVERCOVERAGE_BANDS = (("30-40", 0.30, 0.40), ("40-60", 0.40, 0.60), ("70-100", 0.70, 1.00))
MIN_PREFERENCE_MARGIN = 0.05


def temporal_iou(first: Sequence[int], second: Sequence[int]) -> float:
    # IoU of two binary selections; 1.0 if both are empty.
    if len(first) != len(second):
        raise ValueError(f"annotations with different lengths: {len(first)} and {len(second)}")
    intersection = sum(a == 1 and b == 1 for a, b in zip(first, second))
    union = sum(a == 1 or b == 1 for a, b in zip(first, second))
    return intersection / union if union else 1.0


def binary_f1(first: Sequence[int], second: Sequence[int]) -> float:
    # F1 of two binary selections; 1.0 if both are empty.
    denominator = sum(first) + sum(second)
    if denominator == 0:
        return 1.0
    return 2.0 * sum(a == 1 and b == 1 for a, b in zip(first, second)) / denominator


def mean_annotation_f1(prediction: Sequence[int], annotations: Sequence[Sequence[int]]) -> float:
    # Mean F1 of a selection against every annotation. The values are added left to right, as sum()
    # did before Python 3.12 (which compensates float sums): the paper data were built with Python
    # 3.11, and last-digit differences would change the tie-breaks of some hard rejected selections.
    total = 0.0
    for annotation in annotations:
        total += binary_f1(prediction, annotation)
    return total / len(annotations)


def check_annotations(annotations: Sequence[Sequence[int]]) -> int:
    # Number of frames of a set of binary annotations of equal length.
    if not annotations:
        raise ValueError("vsum_onehot has no annotations")
    frame_count = len(annotations[0])
    for index, annotation in enumerate(annotations):
        if len(annotation) != frame_count:
            raise ValueError(f"annotation {index} has {len(annotation)} frames, expected {frame_count}")
        if set(annotation) - {0, 1}:
            raise ValueError(f"annotation {index} contains non-binary values")
    return frame_count


def temporal_medoid(annotations: Sequence[Sequence[int]]) -> tuple[list[int], int, float, list[float]]:
    # Medoid: the annotation with the highest mean IoU to the other annotations. Returns (medoid,
    # its index, its score, all scores).
    check_annotations(annotations)
    count = len(annotations)
    if count == 1:
        return list(annotations[0]), 0, 1.0, [1.0]
    iou_sums = [0.0] * count
    for first in range(count):
        for second in range(first + 1, count):
            score = temporal_iou(annotations[first], annotations[second])
            iou_sums[first] += score
            iou_sums[second] += score
    mean_scores = [total / (count - 1) for total in iou_sums]
    medoid_index = max(range(count), key=mean_scores.__getitem__)
    return list(annotations[medoid_index]), medoid_index, mean_scores[medoid_index], mean_scores


def selected_segments(label: Sequence[int]) -> list[tuple[int, int]]:
    # Inclusive (start, end) runs of a binary selection.
    segments, start = [], None
    for index, value in enumerate(label):
        if value == 1 and start is None:
            start = index
        elif value == 0 and start is not None:
            segments.append((start, index - 1))
            start = None
    if start is not None:
        segments.append((start, len(label) - 1))
    return segments


def record_rng(seed: int, record_key: Any) -> random.Random:
    # Random generator of one video, seeded by sha256(seed:video id), so every video gets the same
    # random choices whatever the order or subset processed.
    return random.Random(int.from_bytes(hashlib.sha256(f"{seed}:{record_key}".encode()).digest()[:8], "big"))


def best_consensus_path(scores: Sequence[float], budget: int, interval_penalty: float) -> tuple[list[int], int]:
    # Exact selection of `budget` indices that maximizes the summed saliency minus interval_penalty
    # for every interval it opens. Dynamic programming over (index, selected so far, current index
    # selected or not).
    frame_count = len(scores)
    negative_infinity = float("-inf")
    previous = [[negative_infinity] * (budget + 1) for _ in range(2)]
    previous[0][0] = 0.0
    parents = []
    for frame_index, score in enumerate(scores):
        current = [[negative_infinity] * (budget + 1) for _ in range(2)]
        frame_parents = [[-1] * (budget + 1) for _ in range(2)]
        for selected in range(min(budget, frame_index + 1) + 1):
            # Index not selected: the best of the two previous states.
            if previous[0][selected] >= previous[1][selected]:
                current[0][selected], frame_parents[0][selected] = previous[0][selected], 0
            else:
                current[0][selected], frame_parents[0][selected] = previous[1][selected], 1
            if selected == 0:
                continue
            # Index selected: it either continues the current interval or opens a new one, which
            # pays the penalty.
            continuing = previous[1][selected - 1] + score
            starting = previous[0][selected - 1] + score - interval_penalty
            if continuing >= starting:
                current[1][selected], frame_parents[1][selected] = continuing, 1
            else:
                current[1][selected], frame_parents[1][selected] = starting, 0
        previous = current
        parents.append(frame_parents)
    # Backtracking from the best final state with exactly `budget` selected indices.
    state = 0 if previous[0][budget] >= previous[1][budget] else 1
    if previous[state][budget] == negative_infinity:
        raise ValueError("cannot build the consensus target")
    label, selected = [0] * frame_count, budget
    for frame_index in range(frame_count - 1, -1, -1):
        label[frame_index] = state
        parent_state = parents[frame_index][state][selected]
        if state:
            selected -= 1
        state = parent_state
    # Number of intervals of the label.
    intervals = sum(value == 1 and (index == 0 or label[index - 1] == 0) for index, value in enumerate(label))
    return label, intervals


def segment_scores(scores: Sequence[float]) -> list[float]:
    # Saliency of every segment of two consecutive frames: the mean of its frames (the last segment
    # may have one frame).
    return [sum(scores[start:start + 2]) / len(scores[start:start + 2]) for start in range(0, len(scores), 2)]


def pair_annotations(annotations: Sequence[Sequence[int]]) -> list[list[int]]:
    # Annotations on segments: a segment is selected if one of its frames is.
    return [[max(annotation[start:start + 2]) for start in range(0, len(annotation), 2)] for annotation in annotations]


def expand_segments(label: Sequence[int], frame_count: int) -> list[int]:
    # Selection on frames of a selection on segments (segment g covers frames 2g and 2g+1).
    return [label[index // 2] for index in range(frame_count)]


def consensus_label(annotations: Sequence[Sequence[int]], max_intervals: int = 12, interval_penalty: float = 0.15,
                    timeline_unit: str = "frame") -> tuple[list[float], list[int], int, int, float]:
    # Consensus target of one video: (saliency, label, budget, intervals, effective penalty).
    frame_count = check_annotations(annotations)
    if frame_count < 2:
        raise ValueError("at least two frames are needed")
    # Saliency of a frame: the fraction of annotators that selected it.
    scores = [sum(annotation[t] for annotation in annotations) / len(annotations) for t in range(frame_count)]
    # Budget: the median annotation length, rounded half up.
    raw_budget = statistics.median(sum(annotation) for annotation in annotations)
    budget = max(1, min(frame_count - 1, math.floor(raw_budget + 0.5)))
    # On segments the budget is halved, so the covered share of the video stays the same.
    if timeline_unit == "pair":
        scores = segment_scores(scores)
        budget = max(1, min(max(1, len(scores) - 1), math.floor(budget / 2 + 0.5)))
    elif timeline_unit != "frame":
        raise ValueError(f"invalid timeline_unit: {timeline_unit}")
    effective_penalty = interval_penalty
    label, intervals = best_consensus_path(scores, budget, effective_penalty)
    # The penalty is doubled until the label has at most max_intervals intervals.
    for _ in range(20):
        if intervals <= max_intervals:
            break
        effective_penalty = max(1e-6, effective_penalty * 2 or 1e-6)
        label, intervals = best_consensus_path(scores, budget, effective_penalty)
    if intervals > max_intervals:
        raise ValueError(f"cannot respect max_intervals={max_intervals}")
    return scores, label, budget, intervals, effective_penalty


def create_hard_rejected(chosen: Sequence[int], annotations: Sequence[Sequence[int]],
                         rng: random.Random) -> list[int]:
    # Hard rejected: the most shared event of the chosen moves to the most plausible unselected
    # window.
    frame_count, selected_count = len(chosen), sum(chosen)
    if not selected_count or selected_count == frame_count:
        raise ValueError("the chosen must contain both selected and unselected indices")
    # Votes of every index: the number of annotators that selected it.
    votes = [0] * frame_count
    for annotation_index, annotation in enumerate(annotations):
        if len(annotation) != frame_count:
            raise ValueError(f"annotation {annotation_index} has {len(annotation)} frames, expected {frame_count}")
        for index, value in enumerate(annotation):
            votes[index] += int(value == 1)
    chosen_segments = selected_segments(chosen)
    empty_segments = selected_segments([1 - value for value in chosen])
    max_empty_length = max(end - start + 1 for start, end in empty_segments)
    # Only events that fit in an unselected gap can move.
    removable = [segment for segment in chosen_segments if segment[1] - segment[0] + 1 <= max_empty_length]
    if removable:
        # Most shared event: the highest mean number of votes, the shortest on ties, then a random
        # draw.
        def importance(segment):
            start, end = segment
            length = end - start + 1
            return sum(votes[start:end + 1]) / length, -length, rng.random()

        remove_start, remove_end = max(removable, key=importance)
    else:
        # If no event fits, the start of the longest event is moved, cut to the longest gap.
        source_start, source_end = max(chosen_segments, key=lambda segment: segment[1] - segment[0])
        remove_start, remove_end = source_start, min(source_end, source_start + max_empty_length - 1)

    replacement_length = remove_end - remove_start + 1
    # Every window of the same length inside an unselected gap is a candidate destination.
    windows = []
    for gap_start, gap_end in empty_segments:
        for window_start in range(gap_start, gap_end - replacement_length + 2):
            window_end = window_start + replacement_length - 1
            windows.append((sum(votes[window_start:window_end + 1]), window_start, window_end, rng.random()))
    if not windows:
        raise ValueError("no replacement window for a hard rejected")
    chosen_score = mean_annotation_f1(chosen, annotations)
    valid = []
    for _, replacement_start, replacement_end, _ in windows:
        candidate = list(chosen)
        for index in range(remove_start, remove_end + 1):
            candidate[index] = 0
        for index in range(replacement_start, replacement_end + 1):
            candidate[index] = 1
        candidate_score = mean_annotation_f1(candidate, annotations)
        # Only candidates at least MIN_PREFERENCE_MARGIN below the chosen in mean F1 are valid.
        if chosen_score - candidate_score >= MIN_PREFERENCE_MARGIN:
            valid.append((candidate_score, rng.random(), candidate))
    if not valid:
        raise ValueError(f"no hard rejected with an F1 margin of at least {MIN_PREFERENCE_MARGIN}")
    # The most plausible valid candidate: the highest F1, so the rejected stays close to the chosen.
    return max(valid, key=lambda item: (item[0], item[1]))[2]


def create_intermediate_rejected(chosen: Sequence[int], rng: random.Random, max_shift: int = 8) -> list[int]:
    # Intermediate rejected: the chosen shifted left or right, with the same number of indices.
    selected = [index for index, value in enumerate(chosen) if value == 1]
    if not selected or len(selected) == len(chosen):
        raise ValueError("the chosen must contain both selected and unselected indices")
    # The shift is 5% of the timeline, between 2 and max_shift indices.
    shift = max(2, min(max_shift, round(len(chosen) * 0.05)))
    candidates = []
    for direction in (-1, 1):
        # Monotone projection that stops at the timeline borders: indices pile up at the border
        # instead of being lost or wrapped around.
        if direction < 0:
            translated, lower_bound = [], 0
            for index in selected:
                new_index = max(index - shift, lower_bound)
                translated.append(new_index)
                lower_bound = new_index + 1
        else:
            reversed_translation, upper_bound = [], len(chosen) - 1
            for index in reversed(selected):
                new_index = min(index + shift, upper_bound)
                reversed_translation.append(new_index)
                upper_bound = new_index - 1
            translated = list(reversed(reversed_translation))
        if translated != selected:
            candidates.append(translated)
    if not candidates:
        raise ValueError("the chosen cannot be shifted")
    label = [0] * len(chosen)
    # One of the two directions at random.
    for index in rng.choice(candidates):
        label[index] = 1
    return label


def create_overcoverage_rejected(chosen: Sequence[int], rng: random.Random, min_ratio: float = 0.80,
                                 max_ratio: float = 1.00) -> list[int]:
    # Overcoverage rejected: the chosen plus the unselected indices nearest to it.
    frame_count = len(chosen)
    selected = {index for index, value in enumerate(chosen) if value == 1}
    available = [index for index, value in enumerate(chosen) if value == 0]
    if not selected or not available:
        raise ValueError("the chosen must contain both selected and unselected indices")
    if not 0 < min_ratio <= max_ratio <= 1:
        raise ValueError("the overcoverage ratios must lie in (0, 1]")
    # Target coverage drawn uniformly in [min_ratio, max_ratio], with at least one added index.
    target = round(frame_count * rng.uniform(min_ratio, max_ratio))
    target = max(len(selected) + 1, min(frame_count, target))
    distance = {index: min(abs(index - chosen_index) for chosen_index in selected) for index in available}
    # Unselected indices are added from the nearest to the chosen; ties are broken at random.
    ranked = sorted(available, key=lambda index: (distance[index], rng.random()))
    rejected = list(chosen)
    for index in ranked[:target - len(selected)]:
        rejected[index] = 1
    return rejected


def records_of(payload: Any, limit: int | None) -> list[tuple[Any, dict]]:
    # (key, record) pairs of a metadata object or list, optionally the first `limit` ones. The key
    # seeds the random choices of the video: it is the video id, taken from the object keys or from
    # the `video_id` field of a list (the list index when it is missing), so both metadata layouts
    # give the same pairs.
    if isinstance(payload, dict):
        records = list(payload.items())
    elif isinstance(payload, list):
        records = [(record.get("video_id", index) if isinstance(record, dict) else index, record)
                   for index, record in enumerate(payload)]
    else:
        raise ValueError("the JSON must contain an object or a list of videos")
    records = records[:limit] if limit is not None else records
    for key, record in records:
        if not isinstance(record, dict) or "vsum_onehot" not in record:
            raise ValueError(f"record {key!r}: missing or invalid vsum_onehot")
    return records


def add_medoid_labels(records: list[tuple[Any, dict]]) -> None:
    # Medoid annotation of every video, kept only as a diagnostic reference for the consensus.
    for key, record in records:
        try:
            label, index, score, scores = temporal_medoid(record["vsum_onehot"])
        except ValueError as error:
            raise ValueError(f"record {key!r}: {error}") from error
        sampled_frames = record.get("sampled_frames")
        if sampled_frames is not None and len(label) != sampled_frames:
            raise ValueError(f"record {key!r}: label of {len(label)} frames, sampled_frames={sampled_frames}")
        record["vsum_onehot_medoid"] = label
        record["vsum_onehot_medoid_index"] = index
        record["vsum_onehot_medoid_score"] = score
        record["vsum_onehot_mean_iou_scores"] = scores


def add_consensus_labels(records: list[tuple[Any, dict]], max_intervals: int, interval_penalty: float,
                         timeline_unit: str) -> dict[str, float | int]:
    # Consensus target of every video, compared with the medoid.
    delta_sum, wins = 0.0, 0
    for key, record in records:
        scores, label, budget, intervals, penalty = consensus_label(
            record["vsum_onehot"], max_intervals, interval_penalty, timeline_unit)
        frame_count = len(record["vsum_onehot"][0])
        if int(record.get("sampled_frames", frame_count)) != frame_count:
            raise ValueError(f"record {key!r}: inconsistent sampled_frames")
        # The F1 against the annotators is always on frames.
        frame_label = expand_segments(label, frame_count) if timeline_unit == "pair" else label
        consensus_f1 = mean_annotation_f1(frame_label, record["vsum_onehot"])
        medoid_f1 = mean_annotation_f1(record["vsum_onehot_medoid"], record["vsum_onehot"])
        record["timeline_unit"] = timeline_unit
        record["vsum_soft_consensus_scores"] = scores
        record["vsum_onehot_consensus"] = label
        record["vsum_consensus_budget"] = budget
        record["vsum_consensus_intervals"] = intervals
        record["vsum_consensus_interval_penalty"] = penalty
        record["vsum_consensus_f1_mean"] = consensus_f1
        record["vsum_consensus_vs_medoid_f1_delta"] = consensus_f1 - medoid_f1
        delta_sum += consensus_f1 - medoid_f1
        wins += int(consensus_f1 >= medoid_f1)
    return {"processed": len(records), "mean_f1_delta": delta_sum / len(records) if records else 0.0, "wins": wins}


def assign_kinds(records, seed, max_shift, hard_ratio, overcoverage_ratio):
    # Kind of rejected of every video. Hard first: eligible videos have at least two chosen
    # intervals and a valid hard rejected.
    hard_by_position = {}
    for position, (key, record) in enumerate(records):
        if len(selected_segments(record.get("vsum_onehot_consensus", []))) < 2:
            continue
        try:
            hard_by_position[position] = create_hard_rejected(
                record["vsum_onehot_consensus"], record["vsum_onehot"], record_rng(seed, key))
        except ValueError:
            continue
    # --hard-ratio of all videos are drawn among the eligible ones.
    hard_candidates = list(hard_by_position)
    random.Random(seed).shuffle(hard_candidates)
    hard_target = round(len(records) * hard_ratio)
    overcoverage_target = round(len(records) * overcoverage_ratio)
    intermediate_target = len(records) - hard_target - overcoverage_target
    if intermediate_target < 0:
        raise ValueError("the requested proportions give invalid counts")
    if len(hard_candidates) < hard_target:
        raise ValueError(f"not enough videos for {hard_target} hard rejected (available: {len(hard_candidates)})")
    hard = {position: hard_by_position[position] for position in hard_candidates[:hard_target]}
    remaining = [position for position in range(len(records)) if position not in hard]

    # Intermediate among the remaining videos whose shifted chosen loses at least
    # MIN_PREFERENCE_MARGIN of F1.
    intermediate_by_position = {}
    for position in remaining:
        key, record = records[position]
        try:
            candidate = create_intermediate_rejected(record["vsum_onehot_consensus"], record_rng(seed, key), max_shift)
        except ValueError:
            continue
        margin = (mean_annotation_f1(record["vsum_onehot_consensus"], record["vsum_onehot"])
                  - mean_annotation_f1(candidate, record["vsum_onehot"]))
        if margin >= MIN_PREFERENCE_MARGIN:
            intermediate_by_position[position] = candidate
    intermediate_candidates = list(intermediate_by_position)
    random.Random(seed + 1).shuffle(intermediate_candidates)
    if len(intermediate_candidates) < intermediate_target:
        raise ValueError(f"not enough videos for {intermediate_target} intermediate rejected "
                         f"(available after the hard ones: {len(intermediate_candidates)})")
    intermediate = {position: intermediate_by_position[position]
                    for position in intermediate_candidates[:intermediate_target]}
    # All the other videos get an overcoverage rejected.
    overcoverage = set(remaining) - set(intermediate)
    if len(overcoverage) != overcoverage_target:
        raise AssertionError("inconsistent hard/intermediate/overcoverage allocation")
    return hard, intermediate, overcoverage


def overcoverage_bands(positions, seed, band_ratios):
    # Coverage band of every overcoverage video, in the --overcoverage-band-ratios proportions; the
    # last band absorbs rounding.
    ordered = sorted(positions)
    random.Random(seed + 2).shuffle(ordered)
    first = min(round(len(ordered) * band_ratios[0]), len(ordered))
    second = min(round(len(ordered) * band_ratios[1]), len(ordered) - first)
    counts = (first, second, len(ordered) - first - second)
    bands, cursor = {}, 0
    for band, count in zip(OVERCOVERAGE_BANDS, counts):
        for position in ordered[cursor:cursor + count]:
            bands[position] = band
        cursor += count
    return bands


def add_dpo_pairs(records, seed=42, max_shift=4, hard_ratio=0.45, intermediate_ratio=0.25,
                  overcoverage_ratio=0.30, band_ratios=(0.30, 0.50, 0.20)) -> dict[str, int]:
    # Chosen (the consensus target) and rejected selection of every video, with diagnostics.
    hard, intermediate, overcoverage = assign_kinds(records, seed, max_shift, hard_ratio, overcoverage_ratio)
    bands = overcoverage_bands(overcoverage, seed, band_ratios)
    counts = {"intermediate": 0, "hard": 0, "overcoverage": 0}
    band_counts = {name: 0 for name, _, _ in OVERCOVERAGE_BANDS}
    for position, (key, record) in enumerate(records):
        chosen = list(record["vsum_onehot_consensus"])
        if position in hard:
            kind, rejected = "hard", hard[position]
        elif position in intermediate:
            kind, rejected = "intermediate", intermediate[position]
        else:
            kind = "overcoverage"
            band_name, coverage_min, coverage_max = bands[position]
            try:
                rejected = create_overcoverage_rejected(chosen, record_rng(seed, key), coverage_min, coverage_max)
            except ValueError as error:
                raise ValueError(f"record {key!r}: {error}") from error
        chosen_score = mean_annotation_f1(chosen, record["vsum_onehot"])
        rejected_score = mean_annotation_f1(rejected, record["vsum_onehot"])
        margin = chosen_score - rejected_score
        # Sanity checks: hard and intermediate keep the size of the chosen, overcoverage adds
        # indices.
        if rejected == chosen:
            raise AssertionError(f"record {key!r}: chosen and rejected are identical")
        if kind != "overcoverage" and sum(rejected) != sum(chosen):
            raise AssertionError(f"record {key!r}: chosen and rejected have different sizes")
        if kind == "overcoverage" and sum(rejected) <= sum(chosen):
            raise AssertionError(f"record {key!r}: overcoverage adds no index")
        record["vsum_onehot_chosen"] = chosen
        record["vsum_onehot_rejected"] = rejected
        record["dpo_rejected_difficulty"] = kind
        record["dpo_chosen_rejected_iou"] = temporal_iou(chosen, rejected)
        record["dpo_chosen_f1_mean"] = chosen_score
        record["dpo_rejected_f1_mean"] = rejected_score
        record["dpo_preference_margin"] = margin
        # Overcoverage may even have a higher F1: that pair teaches conciseness rather than
        # accuracy.
        if kind != "overcoverage" and margin < MIN_PREFERENCE_MARGIN:
            raise ValueError(f"record {key!r} ({kind}): insufficient margin: chosen={chosen_score:.4f}, "
                             f"rejected={rejected_score:.4f}")
        if kind == "overcoverage":
            record["dpo_overcoverage_band"] = band_name
            record["dpo_overcoverage_ratio"] = sum(rejected) / len(rejected)
            band_counts[band_name] += 1
        else:
            record.pop("dpo_overcoverage_band", None)
            record.pop("dpo_overcoverage_ratio", None)
        counts[kind] += 1
    counts.update({f"overcoverage_{name}": value for name, value in band_counts.items()})
    return counts


def write_json_atomically(payload: Any, output_path: Path) -> None:
    # The JSON is written to a temporary file and then renamed, so a failure leaves no partial
    # output.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output_path.parent,
                                         prefix=f".{output_path.name}.", suffix=".tmp", delete=False) as handle:
            temporary_path = handle.name
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_path, output_path)
    except BaseException:
        if temporary_path is not None:
            Path(temporary_path).unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    # The defaults reproduce the consensus pairs used in the paper.
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="VideoXum training metadata")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, help="only the first N videos (quick tests)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeline-unit", choices=TIMELINE_UNITS, default="pair",
                        help="frame: labels on the N frames; pair: labels on ceil(N/2) pairs of frames")
    parser.add_argument("--max-intervals", type=int, default=12)
    parser.add_argument("--interval-penalty", type=float, default=None, help="default 0.15 (frame), 0.075 (pair)")
    parser.add_argument("--max-shift", type=int, default=None,
                        help="maximum shift of intermediate rejected; default 8 (frame), 4 (pair)")
    parser.add_argument("--hard-ratio", type=float, default=0.45)
    parser.add_argument("--intermediate-ratio", type=float, default=0.25)
    parser.add_argument("--overcoverage-ratio", type=float, default=0.30)
    parser.add_argument("--overcoverage-band-ratios", type=float, nargs=3, default=(0.30, 0.50, 0.20),
                        metavar=("LOW", "MEDIUM", "HIGH"),
                        help="shares of the 30-40%%, 40-60%% and 70-100%% coverage bands")
    args = parser.parse_args()
    for name, value in UNIT_DEFAULTS[args.timeline_unit].items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.max_intervals <= 0 or args.max_shift <= 0:
        parser.error("--max-intervals and --max-shift must be positive")
    if args.interval_penalty < 0:
        parser.error("--interval-penalty must be >= 0")
    for ratios, name in (((args.hard_ratio, args.intermediate_ratio, args.overcoverage_ratio), "rejected kinds"),
                         (args.overcoverage_band_ratios, "overcoverage bands")):
        if any(ratio < 0 or ratio > 1 for ratio in ratios) or not math.isclose(sum(ratios), 1.0, abs_tol=1e-9):
            parser.error(f"the proportions of the {name} must lie in [0, 1] and sum to 1")
    if args.output.is_dir():
        args.output = args.output / DEFAULT_OUTPUT.name
    if args.input.resolve() == args.output.resolve():
        parser.error("input and output must be different files")
    return args


def main() -> None:
    # Add medoid, consensus and DPO pair fields to the metadata and write the result.
    args = parse_args()
    with args.input.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    records = records_of(payload, args.limit)
    add_medoid_labels(records)
    stats = add_consensus_labels(records, args.max_intervals, args.interval_penalty, args.timeline_unit)
    # On the pair timeline the rejected selections are built and scored against pair-level
    # annotations; vsum_onehot is restored on frames afterwards.
    if args.timeline_unit == "pair":
        original_annotations = [record["vsum_onehot"] for _, record in records]
        for _, record in records:
            record["vsum_onehot"] = pair_annotations(record["vsum_onehot"])
    try:
        counts = add_dpo_pairs(records, args.seed, args.max_shift, args.hard_ratio, args.intermediate_ratio,
                               args.overcoverage_ratio, args.overcoverage_band_ratios)
    finally:
        if args.timeline_unit == "pair":
            for (_, record), frame_annotations in zip(records, original_annotations):
                record["vsum_onehot"] = frame_annotations
    for _, record in records:
        record["dpo_scores_timeline_unit"] = args.timeline_unit
    write_json_atomically(payload, args.output)
    print(f"Processed {stats['processed']} videos; consensus >= medoid in {stats['wins']}/{stats['processed']}, "
          f"mean F1 delta={stats['mean_f1_delta']:+.4f}. Rejected: hard={counts['hard']}, "
          f"intermediate={counts['intermediate']}, overcoverage={counts['overcoverage']} "
          f"(30-40%={counts['overcoverage_30-40']}, 40-60%={counts['overcoverage_40-60']}, "
          f"70-100%={counts['overcoverage_70-100']}). Output: {args.output}")


if __name__ == "__main__":
    main()

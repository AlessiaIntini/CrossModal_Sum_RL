"""Parsers of model responses for the three tasks.

Selection (V2V) responses are JSON arrays of intervals, joint (V2VT) responses are objects with
selected_frames and summary, summary (V2T) responses are objects with only summary. Strict
parsers define validity on the test set; lenient ones are used for zero-shot baselines.
"""

import json
import math
import re
from dataclasses import dataclass

import torch

from .timeline import segments_to_frames, selected_runs

JOINT_FIELDS = {"selected_frames", "summary"}
JOINT_BUDGET_RATIO = 0.50
_INTERVAL_PATTERN = re.compile(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]")
_CODE_FENCE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


class CompletionValidationError(ValueError):
    # Joint response that violates the output contract.
    pass


@dataclass(frozen=True)
class ParsedCompletion:
    # Validated joint response; pred_mask and selected_indices are on the parsed timeline.
    selected_frames: list
    summary: str
    pred_mask: torch.Tensor
    selected_indices: list
    merged_adjacent_count: int = 0


def is_integer(value):
    # bool is a subclass of int but is not a valid index.
    return isinstance(value, int) and not isinstance(value, bool)


def parse_interval_response(response, frame_count):
    # Strict V2V parser: the first JSON array of the response, whose intervals must be sorted,
    # disjoint and inside the timeline.
    array_start = re.search(r"\[", response)
    if array_start is None:
        raise ValueError("no JSON array in the response")
    intervals, _ = json.JSONDecoder().raw_decode(response[array_start.start():])
    if not isinstance(intervals, list):
        raise ValueError("the response is not a JSON array")

    prediction = [0] * frame_count
    previous_end = -1
    for interval in intervals:
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError("every interval must be [start, end]")
        start, end = interval
        if not is_integer(start) or not is_integer(end):
            raise ValueError("interval bounds must be integers")
        if not 0 <= start <= end < frame_count:
            raise ValueError(f"interval outside the timeline: [{start}, {end}]")
        if start <= previous_end:
            raise ValueError("intervals must be sorted and must not overlap")
        for index in range(start, end + 1):
            prediction[index] = 1
        previous_end = end
    return prediction


def parse_timeline_response(response, sample):
    # The response is parsed on the model timeline and mapped back to the original frames.
    frame_count = sample["sampled_frames"]
    prediction = parse_interval_response(response, sample.get("timeline_count", frame_count))
    return segments_to_frames(prediction, frame_count, sample.get("timeline_unit", "frame"))


def extract_intervals(response, allow_truncated=False):
    # Intervals of the first JSON array of the response.
    array_start = response.find("[")
    if array_start < 0:
        raise ValueError("no array in the response")
    try:
        intervals, _ = json.JSONDecoder().raw_decode(response[array_start:])
    except json.JSONDecodeError:
        # With allow_truncated, a response cut by the token limit keeps its complete [start, end]
        # pairs.
        if not allow_truncated:
            raise
        intervals = [
            [int(start), int(end)]
            for start, end in _INTERVAL_PATTERN.findall(response[array_start:])
        ]
        if not intervals:
            raise ValueError("no complete interval in the truncated response")
    if not isinstance(intervals, list):
        raise ValueError("the response does not contain an array of intervals")
    return intervals


def normalize_intervals(intervals, frame_count):
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    # Malformed intervals are dropped, bounds swapped and clipped; every repair is recorded in
    # `actions`.
    cleaned, actions = [], set()
    for interval in intervals:
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            actions.add("dropped_malformed_interval")
            continue
        start, end = interval
        if not is_integer(start) or not is_integer(end):
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
        raise ValueError("no valid interval after normalization")
    if cleaned != sorted(cleaned):
        actions.add("sorted_intervals")
    cleaned.sort(key=lambda interval: (interval[0], interval[1]))

    # Sorted intervals are merged when they overlap or touch.
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
    # mode off: strict parser; normalize: repair the intervals of a complete array; recover: also
    # accept truncated arrays.
    if mode == "off":
        prediction = parse_interval_response(response, frame_count)
        intervals = [list(run) for run in selected_runs(prediction)]
        return prediction, intervals, []
    allow_truncated = mode == "recover"
    intervals = extract_intervals(response, allow_truncated=allow_truncated)
    prediction, normalized, actions = normalize_intervals(intervals, frame_count)
    # Responses that needed the truncated-array fallback are marked as recovered.
    if allow_truncated:
        try:
            extract_intervals(response, allow_truncated=False)
        except (json.JSONDecodeError, ValueError):
            actions = sorted(set(actions) | {"recovered_truncated_json"})
    return prediction, normalized, actions


def strip_code_fence(completion):
    # Content of a ```json fenced block, or the completion unchanged.
    if not isinstance(completion, str):
        return completion
    match = _CODE_FENCE.match(completion.strip())
    return match.group(1) if match else completion


def strict_json_loads(completion):
    # json.loads that rejects NaN/Infinity and duplicate keys.
    def reject_constant(value):
        raise ValueError(f"non-standard JSON constant: {value}")

    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field: {key}")
            result[key] = value
        return result

    return json.loads(completion, parse_constant=reject_constant, object_pairs_hook=reject_duplicates)


def joint_budget(timeline_count):
    # Selection budget passed to the strict joint parser at test time.
    return max(1, math.floor(JOINT_BUDGET_RATIO * timeline_count))


def parse_and_validate_completion(completion, count, max_selected):
    # Strict parser of a joint (V2VT) response over a timeline of `count` indices.
    if count <= 0:
        raise ValueError("count must be positive")
    if max_selected <= 0:
        raise ValueError("max_selected must be positive")
    if not isinstance(completion, str):
        raise CompletionValidationError("the completion must be a string")
    try:
        payload = strict_json_loads(strip_code_fence(completion))
    except (json.JSONDecodeError, ValueError) as error:
        raise CompletionValidationError(f"invalid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise CompletionValidationError("the JSON root must be an object")
    if set(payload) != JOINT_FIELDS:
        raise CompletionValidationError("the object must contain exactly selected_frames and summary")

    intervals = payload["selected_frames"]
    if not isinstance(intervals, list) or not intervals:
        raise CompletionValidationError("selected_frames must be a non-empty list")
    selected_indices, clean_intervals = [], []
    previous_end, merged_adjacent_count = -2, 0
    for item in intervals:
        if not isinstance(item, list) or len(item) != 2:
            raise CompletionValidationError("every interval must be [start, end]")
        start, end = item
        if not is_integer(start) or not is_integer(end):
            raise CompletionValidationError("start and end must be integers")
        if not 0 <= start <= end < count:
            raise CompletionValidationError(f"interval out of range: [{start}, {end}]")
        if start <= previous_end:
            raise CompletionValidationError("intervals are not sorted or overlap")
        # Adjacent intervals are merged and counted in merged_adjacent_count instead of being
        # rejected.
        if previous_end >= 0 and start == previous_end + 1:
            clean_intervals[-1][1] = end
            merged_adjacent_count += 1
        else:
            clean_intervals.append([start, end])
        selected_indices.extend(range(start, end + 1))
        previous_end = end
    if len(selected_indices) != len(set(selected_indices)):
        raise CompletionValidationError("the same frame is selected more than once")

    summary = payload["summary"]
    if not isinstance(summary, str) or not summary.strip():
        raise CompletionValidationError("summary must be a non-empty string")
    # max_selected is not enforced here: selections beyond the budget are penalized by the reward.
    pred_mask = torch.zeros(count, dtype=torch.bool)
    pred_mask[selected_indices] = True
    return ParsedCompletion(
        selected_frames=clean_intervals,
        summary=summary.strip(),
        pred_mask=pred_mask,
        selected_indices=selected_indices,
        merged_adjacent_count=merged_adjacent_count,
    )


def expand_pair_completion(parsed, frame_count):
    # Segment g covers frames 2g and 2g+1, clipped to the last frame.
    intervals = [
        [2 * start, min(2 * end + 1, frame_count - 1)]
        for start, end in parsed.selected_frames
    ]
    indices = [index for start, end in intervals for index in range(start, end + 1)]
    mask = torch.zeros(frame_count, dtype=torch.bool)
    mask[indices] = True
    return ParsedCompletion(
        selected_frames=intervals,
        summary=parsed.summary,
        pred_mask=mask,
        selected_indices=indices,
        merged_adjacent_count=parsed.merged_adjacent_count,
    )


def parse_joint_response(response, frame_count, timeline_count, unit):
    # Strict joint parse on the model timeline, mapped back to the original frames.
    parsed = parse_and_validate_completion(response, timeline_count, joint_budget(timeline_count))
    if unit == "pair":
        parsed = expand_pair_completion(parsed, frame_count)
    return parsed


def parse_summary_response(response):
    # Strict V2T parser: a JSON object whose only field is a non-empty summary.
    payload = json.loads(response)
    if not isinstance(payload, dict) or set(payload) != {"summary"}:
        raise ValueError('the response must contain only the "summary" field')
    if not isinstance(payload["summary"], str) or not payload["summary"].strip():
        raise ValueError("summary must be a non-empty string")
    return payload["summary"].strip()


def first_json_object(response):
    # First JSON value starting with "{" in the response.
    start = response.find("{")
    if start < 0:
        raise ValueError("no JSON object in the response")
    payload, _ = json.JSONDecoder().raw_decode(response[start:])
    return payload


def parse_lenient_summary(response):
    # Lenient V2T parser for zero-shot models: the summary of the first JSON object, ignoring
    # fences, extra text and fields.
    payload = first_json_object(response)
    summary = payload.get("summary") if isinstance(payload, dict) else None
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError('missing "summary" field with a non-empty string')
    return summary.strip()


def parse_lenient_selection(response, count):
    # Lenient selection parser: selected_frames of the first JSON object, with the rules of
    # parse_interval_response.
    payload = first_json_object(response)
    intervals = payload.get("selected_frames") if isinstance(payload, dict) else None
    if not isinstance(intervals, list):
        raise ValueError('missing "selected_frames" field with a list of intervals')
    return parse_interval_response(json.dumps(intervals), count)

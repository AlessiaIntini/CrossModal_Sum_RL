"""Timeline units and conversions between binary selections and index intervals.

With unit "frame" the model indexes the 1 fps frames; with unit "pair" it indexes pairs of
consecutive frames, which Qwen3-VL already merges (temporal_patch_size=2). Labels and metrics
always stay on the original frames.
"""

import json

TIMELINE_UNITS = ("frame", "pair")


def timeline_length(frame_count, unit="frame"):
    # On the pair timeline an odd last frame forms a segment of its own.
    if unit == "pair":
        return (frame_count + 1) // 2
    return frame_count


def segments_to_frames(prediction, frame_count, unit="frame"):
    if unit != "pair":
        return list(prediction)
    expected = timeline_length(frame_count, unit)
    if len(prediction) != expected:
        raise ValueError(
            f"pair prediction has {len(prediction)} segments, expected {expected} "
            f"for {frame_count} frames"
        )
    # Segment g covers frames 2g and 2g+1; with an odd frame count the last segment covers only
    # the last frame.
    return [prediction[index // 2] for index in range(frame_count)]


def selected_runs(prediction):
    # Inclusive (start, end) runs of consecutive selected indices.
    runs, start = [], None
    for index, value in enumerate(prediction):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index - 1))
            start = None
    # A run still open at the end reaches the last index.
    if start is not None:
        runs.append((start, len(prediction) - 1))
    return runs


def intervals_json(onehot):
    if not isinstance(onehot, list) or not onehot:
        raise ValueError("empty or invalid one-hot label")
    if any(value not in (0, 1, False, True) for value in onehot):
        raise ValueError("one-hot labels must contain only 0 and 1")
    # Target format of the model: compact JSON array of inclusive [start, end] intervals.
    return json.dumps([list(run) for run in selected_runs(onehot)], separators=(",", ":"))

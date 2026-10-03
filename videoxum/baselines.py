"""Reference selections with the same number of frames as a prediction.

F1 grows with the number of selected frames even without looking at the video, so every
prediction is compared with a random selection and with a positional prior (the relative
positions most often chosen by the training annotators) of the same size.
"""

import json
import random
from pathlib import Path

POSITION_BINS = 20
BASELINE_NAMES = ("random", "positional")


def _annotations(record):
    # vsum_onehot is a JSON string in some metadata files and a list in others.
    value = record["vsum_onehot"]
    return json.loads(value) if isinstance(value, str) else value


def position_bin(index, frame_count, bins=POSITION_BINS):
    # Bin of the relative position of the frame, so videos of any length share the prior.
    return min(bins - 1, index * bins // frame_count)


def load_positional_prior(path, bins=POSITION_BINS):
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload.values() if isinstance(payload, dict) else payload
    totals, counts = [0.0] * bins, [0] * bins
    # Mean saliency (fraction of annotators selecting the frame) of every relative-position bin.
    for record in records:
        annotations = _annotations(record)
        frame_count = len(annotations[0])
        for index in range(frame_count):
            position = position_bin(index, frame_count, bins)
            totals[position] += sum(a[index] for a in annotations) / len(annotations)
            counts[position] += 1
    return [total / count if count else 0.0 for total, count in zip(totals, counts)]


def random_selection(frame_count, selected, rng):
    # `selected` frames drawn uniformly at random.
    prediction = [0] * frame_count
    for index in rng.sample(range(frame_count), selected):
        prediction[index] = 1
    return prediction


def positional_selection(prior, frame_count, selected):
    bins = len(prior)
    # Frames sorted by the prior saliency of their position; ties go to the earliest frame.
    order = sorted(
        range(frame_count),
        key=lambda index: (-prior[position_bin(index, frame_count, bins)], index),
    )
    prediction = [0] * frame_count
    # The `selected` frames with the highest prior saliency.
    for index in order[:selected]:
        prediction[index] = 1
    return prediction


def matched_baselines(prediction, references, metric_fn, prior=None, rng=None):
    rng = rng or random.Random(0)
    # Baselines select as many frames as the prediction, so they isolate the effect of its size.
    frame_count, selected = len(prediction), sum(prediction)
    results = {"random": metric_fn(random_selection(frame_count, selected, rng), references)}
    # The positional baseline needs the prior estimated on the training annotations.
    if prior is not None:
        results["positional"] = metric_fn(positional_selection(prior, frame_count, selected), references)
    return results

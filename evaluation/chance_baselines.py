"""Baseline di riferimento per le selezioni binarie VideoXum.

Le metriche F1 crescono con il numero di frame selezionati anche senza guardare
il video. Per giudicare un modello si confronta ogni sua predizione con due
selezioni che usano esattamente lo stesso numero di frame k sullo stesso video:

- random: k frame scelti a caso;
- positional: i k frame nelle posizioni relative che gli annotatori del
  training scelgono piu' spesso, uguali per ogni video (non guarda il video).
"""

import json
import random
from pathlib import Path

POSITION_BINS = 20


def _annotations(record):
    value = record["vsum_onehot"]
    return json.loads(value) if isinstance(value, str) else value


def _position_bin(index, frame_count, bins=POSITION_BINS):
    return min(bins - 1, index * bins // frame_count)


def load_positional_prior(path, bins=POSITION_BINS):
    """Saliency media per posizione relativa, stimata su un file di metadata."""
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload.values() if isinstance(payload, dict) else payload
    totals, counts = [0.0] * bins, [0] * bins
    for record in records:
        annotations = _annotations(record)
        frame_count = len(annotations[0])
        for index in range(frame_count):
            position = _position_bin(index, frame_count, bins)
            totals[position] += sum(a[index] for a in annotations) / len(annotations)
            counts[position] += 1
    return [total / count if count else 0.0 for total, count in zip(totals, counts)]


def random_selection(frame_count, selected, rng):
    prediction = [0] * frame_count
    for index in rng.sample(range(frame_count), selected):
        prediction[index] = 1
    return prediction


def positional_selection(prior, frame_count, selected):
    bins = len(prior)
    order = sorted(
        range(frame_count),
        key=lambda index: (-prior[_position_bin(index, frame_count, bins)], index),
    )
    prediction = [0] * frame_count
    for index in order[:selected]:
        prediction[index] = 1
    return prediction


def matched_baselines(prediction, references, metric_fn, prior=None, rng=None):
    """Metriche delle baseline con lo stesso numero di frame della predizione."""
    rng = rng or random.Random(0)
    frame_count, selected = len(prediction), sum(prediction)
    results = {
        "random": metric_fn(random_selection(frame_count, selected, rng), references)
    }
    if prior is not None:
        results["positional"] = metric_fn(
            positional_selection(prior, frame_count, selected), references
        )
    return results

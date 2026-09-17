#!/usr/bin/env python3
"""F1 della selezione frame VideoXum normalizzata sul caso a parita' di lunghezza.

La f1_mean di evalVtV cresce con il numero di frame selezionati anche senza
scegliere meglio. Qui ogni video viene confrontato con una selezione casuale
dello stesso numero di frame:

  por  = f1_mean / f1_random   (>1 = meglio del caso)
  lift = f1_mean - f1_random

Con n_pred fissato il denominatore della F1 e' costante (F1 = 2*inter/(n_pred+|g|)),
quindi il valore atteso sotto selezione uniforme e' esatto:

  E[F1_random(g)] = 2 * n_pred * |g| / L / (n_pred + |g|)

Il Monte Carlo (--mc-seeds) serve solo a verificare la formula.

Esempio:
  python evaluation/eval_vsum_por.py \
      --predictions results/.../predictions-test-0-4000-*.jsonl \
      --annotations /leonardo_work/.../metadata/test_videoxum.json \
      --output results/evaluation/videoxum_test_..._por.jsonl --index-base 0
"""

import argparse
import json
from pathlib import Path

import numpy as np

from evalVtV import (
    annotation_references,
    load_vsum_annotations,
    load_vsum_predictions,
    prediction_mask,
)


def f1_per_annotator(prediction, references):
    """F1 della maschera predetta contro ciascuna delle annotazioni binarie."""
    intersections = (references * prediction[None, :]).sum(axis=1)
    denominators = references.sum(axis=1) + prediction.sum()
    return np.divide(2.0 * intersections, denominators,
                     out=np.zeros(len(references), dtype=float), where=denominators != 0)


def expected_random_f1(n_pred, references):
    """F1 attesa (media sugli annotatori) di n_pred frame scelti a caso."""
    frame_count = references.shape[1]
    gt_sizes = references.sum(axis=1).astype(float)
    denominators = n_pred + gt_sizes
    values = np.divide(2.0 * n_pred * gt_sizes / frame_count, denominators,
                       out=np.zeros(len(references), dtype=float), where=denominators != 0)
    return float(values.mean())


def monte_carlo_random_f1(n_pred, references, seeds, rng):
    """Stessa quantita' di expected_random_f1 stimata per campionamento."""
    frame_count = references.shape[1]
    values = []
    for _ in range(seeds):
        mask = np.zeros(frame_count, dtype=np.int8)
        mask[rng.choice(frame_count, n_pred, replace=False)] = 1
        values.append(f1_per_annotator(mask, references).mean())
    return float(np.mean(values))


def bootstrap_ci(values, rounds=2000, seed=0):
    """IC95 percentile della media."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(rounds, len(values)))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def evaluate_por(predictions_path, annotations_path, output_path, index_base=1,
                 mc_seeds=0, bootstrap_rounds=2000):
    predictions = load_vsum_predictions(predictions_path, index_base)
    annotations = load_vsum_annotations(annotations_path)
    rng = np.random.default_rng(0)

    f1s, randoms, lifts, coverages, pors, mc_errors = [], [], [], [], [], []
    missing = empty = 0
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for video_id, row in annotations.items():
            if video_id not in predictions:
                missing += 1
                continue
            frame_count, references = annotation_references(video_id, row)
            prediction, _ = prediction_mask(*predictions[video_id], frame_count)
            n_pred = int(prediction.sum())

            f1_mean = float(f1_per_annotator(prediction, references).mean())
            f1_random = expected_random_f1(n_pred, references)
            # Senza frame selezionati il rapporto e' 0/0: il video resta negli
            # aggregati di f1 e lift ma non nella media per-video di por.
            por = f1_mean / f1_random if n_pred and f1_random > 0 else None
            if n_pred == 0:
                empty += 1
            record = {
                "record_type": "sample", "video_id": video_id,
                "selected_frames": n_pred, "coverage": n_pred / frame_count,
                "f1_mean": f1_mean, "f1_random": f1_random,
                "lift": f1_mean - f1_random, "por": por,
            }
            if mc_seeds and n_pred:
                record["f1_random_mc"] = monte_carlo_random_f1(n_pred, references, mc_seeds, rng)
                mc_errors.append(abs(record["f1_random_mc"] - f1_random))

            f1s.append(f1_mean)
            randoms.append(f1_random)
            lifts.append(record["lift"])
            coverages.append(record["coverage"])
            if por is not None:
                pors.append(por)
            output.write(json.dumps(record) + "\n")

        if not f1s:
            raise ValueError("Nessuna predizione corrisponde alle annotazioni")
        summary = {
            "record_type": "summary", "videos_evaluated": len(f1s),
            "predictions_total": len(predictions), "annotations_total": len(annotations),
            "missing_predictions": missing, "empty_predictions": empty,
            "index_base": index_base,
            "coverage": float(np.mean(coverages)),
            "f1_mean": float(np.mean(f1s)),
            "f1_random": float(np.mean(randoms)),
            "por_aggregate": float(np.mean(f1s) / np.mean(randoms)) if np.mean(randoms) > 0 else None,
            "por_mean": float(np.mean(pors)) if pors else None,
            "por_mean_ci95": bootstrap_ci(pors, bootstrap_rounds) if pors else None,
            "por_median": float(np.median(pors)) if pors else None,
            "lift": float(np.mean(lifts)),
            "lift_ci95": bootstrap_ci(lifts, bootstrap_rounds),
        }
        if mc_errors:
            summary["mc_seeds"] = mc_seeds
            summary["mc_abs_error_mean"] = float(np.mean(mc_errors))
            summary["mc_abs_error_max"] = float(np.max(mc_errors))
        output.write(json.dumps(summary) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Risultati salvati in {output_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index-base", type=int, choices=(0, 1), default=1)
    parser.add_argument("--mc-seeds", type=int, default=0,
                        help="se >0 stima anche f1_random per campionamento (solo verifica)")
    parser.add_argument("--bootstrap-rounds", type=int, default=2000)
    cli = parser.parse_args()
    evaluate_por(cli.predictions, cli.annotations, cli.output, cli.index_base,
                 cli.mc_seeds, cli.bootstrap_rounds)

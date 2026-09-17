#!/usr/bin/env python3
"""F_CLIP della selezione frame VideoXum (definizione V2Xum-LLM, arXiv 2404.12353).

Per un summary di riferimento v e uno predetto v_hat, con feature CLIP
L2-normalizzate e similarita' troncate a zero (max(cos, 0)):

  R_CLIP = 1/|v|     sum_{v_i in v}         max_{v_hat_j in v_hat} v_i . v_hat_j
  P_CLIP = 1/|v_hat| sum_{v_hat_j in v_hat} max_{v_i in v}         v_i . v_hat_j
  F_CLIP = 2 * P_CLIP * R_CLIP / (P_CLIP + R_CLIP)

Ogni video ha 10 annotazioni: si riportano media e massimo sugli annotatori.
Come per eval_vsum_por.py, F_CLIP viene confrontata con selezioni casuali dello
stesso numero di frame (Monte Carlo, qui non esiste una forma chiusa), perche'
anche F_CLIP cambia con la lunghezza della selezione.

Le feature vengono da configuration/extract_clip_embeddings.py
(<feature-dir>/<video_id>.npz, chiave "features", shape [sampled_frames, dim]).

Esempio:
  python evaluation/eval_vsum_fclip.py \
      --predictions results/.../predictions-test-0-4000-*.jsonl \
      --annotations /leonardo_work/.../metadata/test_videoxum.json \
      --feature-dir /leonardo_work/.../videoxum/feat/clip-vit-large-patch14-336 \
      --output results/evaluation/videoxum_test_..._fclip.jsonl --index-base 0
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
from eval_vsum_por import bootstrap_ci


def fclip_per_annotator(similarity, prediction, references):
    """F_CLIP (P, R, F) della maschera predetta contro ciascuna annotazione.

    ``similarity`` e' la matrice frame x frame gia' troncata a zero.
    """
    predicted = np.flatnonzero(prediction)
    scores = np.zeros((len(references), 3), dtype=float)
    if not len(predicted):
        return scores
    for k, reference in enumerate(references):
        selected = np.flatnonzero(reference)
        if not len(selected):
            continue
        block = similarity[np.ix_(selected, predicted)]   # |v| x |v_hat|
        recall = block.max(axis=1).mean()
        precision = block.max(axis=0).mean()
        f = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        scores[k] = (precision, recall, f)
    return scores


def random_fclip(similarity, n_pred, references, seeds, rng):
    """F_CLIP media (sugli annotatori) di n_pred frame scelti a caso, mediata sui seed."""
    frame_count = similarity.shape[0]
    values = []
    for _ in range(seeds):
        mask = np.zeros(frame_count, dtype=np.int8)
        mask[rng.choice(frame_count, n_pred, replace=False)] = 1
        values.append(fclip_per_annotator(similarity, mask, references)[:, 2].mean())
    return float(np.mean(values))


def load_features(feature_dir, video_id, frame_count):
    path = Path(feature_dir) / f"{video_id}.npz"
    if not path.is_file():
        return None
    with np.load(path) as cached:
        features = cached["features"].astype(np.float32)
    if features.shape[0] != frame_count:
        raise ValueError(f"{video_id}: {features.shape[0]} feature, attesi {frame_count} frame")
    # Le feature sono salvate normalizzate; rinormalizzare costa poco e protegge
    # da cache prodotte con versioni diverse dell'estrattore.
    return features / np.linalg.norm(features, axis=1, keepdims=True).clip(min=1e-12)


def evaluate_fclip(predictions_path, annotations_path, feature_dir, output_path,
                   index_base=1, random_seeds=20, bootstrap_rounds=2000):
    predictions = load_vsum_predictions(predictions_path, index_base)
    annotations = load_vsum_annotations(annotations_path)
    rng = np.random.default_rng(0)

    names = ("fclip_p", "fclip_r", "fclip_mean", "fclip_max", "fclip_random", "lift", "coverage")
    collected = {name: [] for name in names}
    ratios = []
    missing = missing_features = empty = 0
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for video_id, row in annotations.items():
            if video_id not in predictions:
                missing += 1
                continue
            frame_count, references = annotation_references(video_id, row)
            features = load_features(feature_dir, video_id, frame_count)
            if features is None:
                missing_features += 1
                continue
            prediction, _ = prediction_mask(*predictions[video_id], frame_count)
            n_pred = int(prediction.sum())
            empty += n_pred == 0

            similarity = np.maximum(features @ features.T, 0.0)
            scores = fclip_per_annotator(similarity, prediction, references)
            fclip_random = (random_fclip(similarity, n_pred, references, random_seeds, rng)
                            if n_pred and random_seeds else 0.0)
            record = {
                "record_type": "sample", "video_id": video_id,
                "selected_frames": n_pred, "coverage": n_pred / frame_count,
                "fclip_p": float(scores[:, 0].mean()), "fclip_r": float(scores[:, 1].mean()),
                "fclip_mean": float(scores[:, 2].mean()), "fclip_max": float(scores[:, 2].max()),
                "fclip_per_annotator": scores[:, 2].tolist(),
                "fclip_random": fclip_random,
            }
            record["lift"] = record["fclip_mean"] - fclip_random
            record["por"] = record["fclip_mean"] / fclip_random if fclip_random > 0 else None
            for name in names:
                collected[name].append(record[name])
            if record["por"] is not None:
                ratios.append(record["por"])
            output.write(json.dumps(record) + "\n")

        if not collected["fclip_mean"]:
            raise ValueError("Nessuna predizione con feature corrisponde alle annotazioni")
        means = {name: float(np.mean(values)) for name, values in collected.items()}
        summary = {
            "record_type": "summary", "videos_evaluated": len(collected["fclip_mean"]),
            "predictions_total": len(predictions), "annotations_total": len(annotations),
            "missing_predictions": missing, "missing_features": missing_features,
            "empty_predictions": int(empty), "index_base": index_base,
            "random_seeds": random_seeds,
            **means,
            "fclip_mean_ci95": bootstrap_ci(collected["fclip_mean"], bootstrap_rounds),
            "lift_ci95": bootstrap_ci(collected["lift"], bootstrap_rounds),
        }
        if random_seeds:
            summary["por_aggregate"] = (means["fclip_mean"] / means["fclip_random"]
                                        if means["fclip_random"] > 0 else None)
            summary["por_mean"] = float(np.mean(ratios)) if ratios else None
        output.write(json.dumps(summary) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Risultati salvati in {output_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index-base", type=int, choices=(0, 1), default=1)
    parser.add_argument("--random-seeds", type=int, default=20,
                        help="selezioni casuali per video per la baseline a parita' di lunghezza (0 = disattiva)")
    parser.add_argument("--bootstrap-rounds", type=int, default=2000)
    cli = parser.parse_args()
    evaluate_fclip(cli.predictions, cli.annotations, cli.feature_dir, cli.output,
                   cli.index_base, cli.random_seeds, cli.bootstrap_rounds)

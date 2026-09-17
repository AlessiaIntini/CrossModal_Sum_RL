#!/usr/bin/env python3
"""Confronto tra run di inferenza VideoXum con metriche che non premiano la quantita'.

Legge i file predictions-test-*.jsonl di una o piu' directory di risultati (come
prodotte da infer_videoxum.py / infer_grpo_videoxum.py), si limita ai video
presenti in tutte le run e riporta per ciascuna:

- selection_ratio, f1_max, f1_mean, kappa, lift (sugli N frame);
- le stesse metriche per la selezione casuale con lo stesso numero di frame;
- intervallo di confidenza bootstrap al 95% di kappa.

Con --matched A B confronta A e B a parita' di frame: sui video in cui A seleziona
almeno quanti frame di B, riduce a caso la selezione di A al numero di frame di B
(la qualita' media dei frame di A resta invariata) e riporta la differenza con B.
f1_max non e' affidabile in questo confronto: togliere frame a caso spezza gli
intervalli contigui.

Esempio:
  python evaluation/compare_runs.py \
      --test_json /path/test_videoxum.json \
      results/inference-dpo/dpo-test results/inference-dpo-sft-ConsensusDataset/dpo-test \
      --matched results/inference-dpo/dpo-test results/inference-dpo-sft-ConsensusDataset/dpo-test
"""

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.chance_baselines import random_selection  # noqa: E402
from sft_train_videoxum import binary_f1, chance_corrected_metrics  # noqa: E402

REPORTED = ("selection_ratio", "f1_max", "f1_mean", "kappa", "lift")


def load_references(path):
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    items = payload.items() if isinstance(payload, dict) else enumerate(payload)
    references = {}
    for key, record in items:
        value = record["vsum_onehot"]
        references[str(record.get("video_id", key))] = (
            json.loads(value) if isinstance(value, str) else value
        )
    return references


def load_run(run_dir, references):
    """Selezioni sugli N frame per video; le predizioni non valide sono vuote."""
    selections = {}
    for path in sorted(Path(run_dir).glob("predictions-test-*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("record_type") != "sample":
                    continue
                video_id = record["video_id"]
                if video_id not in references:
                    continue
                frame_count = len(references[video_id][0])
                valid = record.get("normalized_valid", record.get("prediction_valid"))
                prediction = [0] * frame_count
                for index in (record.get("selected_frame_indices") or []) if valid else []:
                    if 0 <= index < frame_count:
                        prediction[index] = 1
                selections[video_id] = prediction
    if not selections:
        raise SystemExit(f"nessuna predizione in {run_dir}")
    return selections


def metrics(prediction, references):
    f1_scores = [binary_f1(prediction, reference) for reference in references]
    return {
        "selection_ratio": sum(prediction) / len(prediction),
        "f1_max": max(f1_scores),
        "f1_mean": sum(f1_scores) / len(f1_scores),
        **chance_corrected_metrics(prediction, references),
    }


def bootstrap_ci(values, rng, repeats=1000):
    size = len(values)
    means = sorted(
        statistics.fmean(values[rng.randrange(size)] for _ in range(size))
        for _ in range(repeats)
    )
    return means[int(0.025 * repeats)], means[int(0.975 * repeats)]


def summarize(rows):
    return {name: statistics.fmean(row[name] for row in rows) for name in REPORTED}


def matched_comparison(first, second, video_ids, references, rng, repeats=5):
    differences = {name: [] for name in REPORTED}
    skipped = 0
    for video_id in video_ids:
        prediction_a, prediction_b = first[video_id], second[video_id]
        budget = sum(prediction_b)
        selected = [index for index, value in enumerate(prediction_a) if value]
        if len(selected) < budget:
            skipped += 1
            continue
        refs = references[video_id]
        reduced = []
        for _ in range(repeats):
            prediction = [0] * len(prediction_a)
            for index in rng.sample(selected, budget):
                prediction[index] = 1
            reduced.append(metrics(prediction, refs))
        reference_metrics = metrics(prediction_b, refs)
        for name in REPORTED:
            differences[name].append(
                statistics.fmean(row[name] for row in reduced) - reference_metrics[name]
            )
    return differences, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path, help="directory con predictions-test-*.jsonl")
    parser.add_argument("--test_json", type=Path, required=True)
    parser.add_argument("--matched", nargs=2, type=Path, metavar=("A", "B"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    references = load_references(args.test_json)
    runs = {run: load_run(run, references) for run in args.runs}
    if args.matched:
        for run in args.matched:
            runs.setdefault(run, load_run(run, references))
    video_ids = sorted(set.intersection(*(set(selections) for selections in runs.values())))
    print(f"video comuni a tutte le run: {len(video_ids)}")

    rng = random.Random(args.seed)
    header = " ".join(f"{name:>15}" for name in REPORTED)
    print(f"{'run':60} {header} {'kappa CI95':>18}")
    for run in args.runs:
        rows = [metrics(runs[run][video_id], references[video_id]) for video_id in video_ids]
        chance = [
            metrics(
                random_selection(len(runs[run][video_id]), sum(runs[run][video_id]), rng),
                references[video_id],
            )
            for video_id in video_ids
        ]
        low, high = bootstrap_ci([row["kappa"] for row in rows], rng)
        values = summarize(rows)
        chance_values = summarize(chance)
        print(f"{str(run)[-60:]:60} " + " ".join(f"{values[name]:15.4f}" for name in REPORTED)
              + f"   [{low:.4f},{high:.4f}]")
        print(f"{'  caso, stessi frame':60} " + " ".join(f"{chance_values[name]:15.4f}" for name in REPORTED))

    if args.matched:
        first, second = args.matched
        differences, skipped = matched_comparison(
            runs[first], runs[second], video_ids, references, rng
        )
        print(f"\nA parita' di frame: {first} ridotto a {second} "
              f"({len(differences['kappa'])} video, {skipped} esclusi perche' A seleziona meno frame)")
        for name in REPORTED[1:]:
            values = differences[name]
            low, high = bootstrap_ci(values, rng)
            note = "  (non affidabile: la riduzione spezza gli intervalli)" if name == "f1_max" else ""
            print(f"  {name:8} A-B = {statistics.fmean(values):+.4f}  CI95 [{low:+.4f},{high:+.4f}]{note}")


if __name__ == "__main__":
    main()

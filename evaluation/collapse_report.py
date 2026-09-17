#!/usr/bin/env python3
"""Report di collasso e Kendall per un file di predizioni VideoXum.

Legge i predictions-*.jsonl scritti da infer_videoxum.py / infer_grpo_videoxum.py
e misura quanto l'output dipende dal video (vedi collapse_metrics in
sft_train_videoxum.py), piu' Kendall/Spearman con intervallo bootstrap. Non usa
la GPU: serve per confrontare run gia' valutate senza rigenerare.

Esempio:
  python evaluation/collapse_report.py \
      --predictions results/inference-sft-ConsensusDataset/*/predictions-test-0-4000-*.jsonl \
      --annotations /leonardo_work/.../metadata/test_videoxum.json
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sft_train_videoxum import (  # noqa: E402
    collapse_metrics,
    parse_interval_response,
    videoxum_metrics,
)


def load_annotations(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return {str(key): value for key, value in payload.items()}
    return {str(row.get("video_id") or row.get("id")): row for row in payload}


def load_predictions(path, annotations):
    responses, predictions, references_list, video_mode = [], [], [], None
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("record_type") == "config":
                video_mode = (row.get("arguments") or {}).get("video_time_mode", "legacy (pre-fix)")
                continue
            if row.get("record_type") not in (None, "sample"):
                continue
            annotation = annotations.get(str(row.get("video_id")))
            if annotation is None:
                continue
            frame_count = int(annotation["sampled_frames"])
            response = str(row.get("generated_response") or "")
            if row.get("selected_frame_indices") is not None:
                prediction = [0] * frame_count
                for index in row["selected_frame_indices"]:
                    if 0 <= int(index) < frame_count:
                        prediction[int(index)] = 1
            else:
                try:
                    prediction = parse_interval_response(response, frame_count)
                except (json.JSONDecodeError, ValueError):
                    prediction = [0] * frame_count
            responses.append(response)
            predictions.append(prediction)
            references_list.append(annotation["vsum_onehot"])
    return responses, predictions, references_list, video_mode


def bootstrap_mean(values, rounds=2000, seed=0):
    rng = random.Random(seed)
    means = sorted(
        sum(rng.choice(values) for _ in values) / len(values) for _ in range(rounds)
    )
    return sum(values) / len(values), means[int(0.025 * rounds)], means[int(0.975 * rounds)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--bootstrap_rounds", type=int, default=2000)
    parser.add_argument("--output", type=Path, help="scrive anche il report JSON")
    args = parser.parse_args()

    annotations = load_annotations(args.annotations)
    reports = {}
    for path in args.predictions:
        responses, predictions, references_list, video_mode = load_predictions(path, annotations)
        if not predictions:
            print(f"[skip] {path}: nessuna predizione associata alle annotazioni")
            continue
        per_video = [
            videoxum_metrics(prediction, references)
            for prediction, references in zip(predictions, references_list)
        ]
        report = {"videos": len(predictions), "video_time_mode": video_mode}
        for name in ("f1_mean", "f1_max", "kendall", "spearman"):
            mean, low, high = bootstrap_mean(
                [metrics[name] for metrics in per_video], args.bootstrap_rounds
            )
            report[name] = {"mean": mean, "ci95": [low, high]}
        report["collapse"] = collapse_metrics(responses, predictions, references_list)
        reports[str(path)] = report

        collapse = report["collapse"]
        print(f"\n== {path}")
        print(f"   video={report['videos']}  video_time_mode={video_mode}")
        for name in ("kendall", "spearman", "f1_mean", "f1_max"):
            low, high = report[name]["ci95"]
            print(f"   {name:<9} {report[name]['mean']:.4f}  IC95 [{low:.4f}, {high:.4f}]")
        for key in (
            "distinct_output_ratio", "top4_output_share", "prefix2_share",
            "first_start_mode_share", "modal_interval_length_share",
            "mean_interval_length", "mean_density", "cross_video_iou",
            "cross_video_iou_null", "decile_profile_l1", "empty_prediction_rate",
        ):
            if collapse.get(key) is not None:
                print(f"   {key:<28} {collapse[key]:.3f}")
        print(f"   first_start_top5             {collapse.get('first_start_top5')}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
        print(f"\n[done] report JSON: {args.output}")


if __name__ == "__main__":
    main()

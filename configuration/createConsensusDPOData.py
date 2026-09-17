#!/usr/bin/env python3
"""Crea un dataset SFT/DPO con un unico target di consenso per video."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Sequence

from createDPODate import (
    DEFAULT_INPUT,
    add_dpo_pairs,
    add_medoid_labels,
    mean_annotation_f1,
    write_json_atomically,
)


DEFAULT_OUTPUT = (
    DEFAULT_INPUT.parent / "derived" / "train_videoxum_dpo_consensus.json"
)

def resolve_output_path(output_path: Path) -> Path:
    """Accetta sia un file JSON sia una directory di destinazione."""
    if output_path.is_dir():
        resolved = output_path / DEFAULT_OUTPUT.name
        print(
            f"[output] {output_path} e' una directory; uso il file {resolved}"
        )
        return resolved
    return output_path



def _best_consensus_path(
    scores: Sequence[float],
    budget: int,
    interval_penalty: float,
) -> tuple[list[int], int]:
    """Seleziona esattamente budget frame penalizzando l'inizio di intervalli."""
    frame_count = len(scores)
    negative_infinity = float("-inf")
    # Stato 0/1: il frame corrente e' escluso/selezionato.
    previous = [[negative_infinity] * (budget + 1) for _ in range(2)]
    previous[0][0] = 0.0
    parents: list[list[list[int]]] = []

    for frame_index, score in enumerate(scores):
        current = [[negative_infinity] * (budget + 1) for _ in range(2)]
        frame_parents = [[-1] * (budget + 1) for _ in range(2)]
        max_selected = min(budget, frame_index + 1)
        for selected in range(max_selected + 1):
            # Escludere il frame: si puo' arrivare da entrambi gli stati.
            if previous[0][selected] >= previous[1][selected]:
                current[0][selected] = previous[0][selected]
                frame_parents[0][selected] = 0
            else:
                current[0][selected] = previous[1][selected]
                frame_parents[0][selected] = 1

            if selected == 0:
                continue
            continuing = previous[1][selected - 1] + score
            starting = previous[0][selected - 1] + score - interval_penalty
            if continuing >= starting:
                current[1][selected] = continuing
                frame_parents[1][selected] = 1
            else:
                current[1][selected] = starting
                frame_parents[1][selected] = 0
        previous = current
        parents.append(frame_parents)

    state = 0 if previous[0][budget] >= previous[1][budget] else 1
    if previous[state][budget] == negative_infinity:
        raise ValueError("impossibile costruire il target di consenso")
    label = [0] * frame_count
    selected = budget
    for frame_index in range(frame_count - 1, -1, -1):
        label[frame_index] = state
        parent_state = parents[frame_index][state][selected]
        if state:
            selected -= 1
        state = parent_state
    intervals = sum(
        value == 1 and (index == 0 or label[index - 1] == 0)
        for index, value in enumerate(label)
    )
    return label, intervals


TIMELINE_UNITS = ("frame", "pair")
# Con le coppie il guadagno di un segmento e' la media di due frame, quindi la
# penalita' per intervallo va dimezzata per ottenere la stessa frammentazione;
# lo spostamento dei rejected intermediate e' misurato in segmenti.
UNIT_DEFAULTS = {
    "frame": {"interval_penalty": 0.15, "max_shift": 8},
    "pair": {"interval_penalty": 0.075, "max_shift": 4},
}


def segment_scores(scores: Sequence[float]) -> list[float]:
    """Saliency media per coppia di frame consecutivi (l'ultima puo' essere singola)."""
    return [
        sum(scores[start:start + 2]) / len(scores[start:start + 2])
        for start in range(0, len(scores), 2)
    ]


def pair_annotations(annotations: Sequence[Sequence[int]]) -> list[list[int]]:
    """Annotazioni portate a coppie: un segmento vale 1 se contiene un frame scelto."""
    return [
        [max(annotation[start:start + 2]) for start in range(0, len(annotation), 2)]
        for annotation in annotations
    ]


def expand_segments(label: Sequence[int], frame_count: int) -> list[int]:
    """Il segmento g copre i frame 2g e 2g+1."""
    return [label[index // 2] for index in range(frame_count)]


def consensus_label(
    annotations: Sequence[Sequence[int]],
    max_intervals: int = 12,
    interval_penalty: float = 0.15,
    timeline_unit: str = "frame",
) -> tuple[list[float], list[int], int, int, float]:
    """Fonde tutte le annotazioni in score soft e in un'unica label coerente.

    Con timeline_unit='pair' score, label e budget sono sui segmenti: la
    saliency di un segmento e' la media dei suoi due frame e il budget e' meta'
    di quello a frame, cosi' la copertura del video resta la stessa."""
    if not annotations:
        raise ValueError("vsum_onehot non contiene annotazioni")
    frame_count = len(annotations[0])
    if frame_count < 2:
        raise ValueError("servono almeno due frame")
    for index, annotation in enumerate(annotations):
        if len(annotation) != frame_count:
            raise ValueError(
                f"annotazione {index} lunga {len(annotation)}; attesi {frame_count}"
            )
        if set(annotation) - {0, 1}:
            raise ValueError(f"annotazione {index} contiene valori non binari")

    scores = [
        sum(annotation[t] for annotation in annotations) / len(annotations)
        for t in range(frame_count)
    ]
    raw_budget = statistics.median(sum(annotation) for annotation in annotations)
    budget = max(1, min(frame_count - 1, math.floor(raw_budget + 0.5)))
    if timeline_unit == "pair":
        scores = segment_scores(scores)
        budget = max(1, min(max(1, len(scores) - 1), math.floor(budget / 2 + 0.5)))
    elif timeline_unit != "frame":
        raise ValueError(f"timeline_unit non valida: {timeline_unit}")

    # Aumenta automaticamente la penalita' solo se il limite non e' rispettato.
    effective_penalty = interval_penalty
    label, intervals = _best_consensus_path(scores, budget, effective_penalty)
    for _ in range(20):
        if intervals <= max_intervals:
            break
        effective_penalty = max(1e-6, effective_penalty * 2 or 1e-6)
        label, intervals = _best_consensus_path(
            scores, budget, effective_penalty
        )
    if intervals > max_intervals:
        raise ValueError(
            f"impossibile rispettare max_intervals={max_intervals}"
        )
    return scores, label, budget, intervals, effective_penalty


def add_consensus_labels(
    payload: Any,
    limit: int | None,
    max_intervals: int,
    interval_penalty: float,
    timeline_unit: str = "frame",
) -> dict[str, float | int]:
    if isinstance(payload, dict):
        records = list(payload.items())
    elif isinstance(payload, list):
        records = list(enumerate(payload))
    else:
        raise ValueError("Il JSON deve contenere un oggetto o una lista di video")
    if limit is not None:
        records = records[:limit]

    score_delta_sum = 0.0
    consensus_wins = 0
    for record_key, record in records:
        if not isinstance(record, dict) or "vsum_onehot" not in record:
            raise ValueError(
                f"Record {record_key!r}: vsum_onehot mancante o non valido"
            )
        scores, label, budget, intervals, effective_penalty = consensus_label(
            record["vsum_onehot"], max_intervals, interval_penalty, timeline_unit
        )
        frame_count = len(record["vsum_onehot"][0])
        if int(record.get("sampled_frames", frame_count)) != frame_count:
            raise ValueError(
                f"Record {record_key!r}: sampled_frames incoerente"
            )
        # F1 di diagnostica sempre sui frame, confrontabile con il medoid.
        frame_label = (
            expand_segments(label, frame_count) if timeline_unit == "pair" else label
        )
        consensus_f1 = mean_annotation_f1(frame_label, record["vsum_onehot"])
        medoid_f1 = mean_annotation_f1(
            record["vsum_onehot_medoid"], record["vsum_onehot"]
        )
        record["timeline_unit"] = timeline_unit
        record["vsum_soft_consensus_scores"] = scores
        record["vsum_onehot_consensus"] = label
        record["vsum_consensus_budget"] = budget
        record["vsum_consensus_intervals"] = intervals
        record["vsum_consensus_interval_penalty"] = effective_penalty
        record["vsum_consensus_f1_mean"] = consensus_f1
        record["vsum_consensus_vs_medoid_f1_delta"] = (
            consensus_f1 - medoid_f1
        )
        score_delta_sum += consensus_f1 - medoid_f1
        consensus_wins += int(consensus_f1 >= medoid_f1)

    count = len(records)
    return {
        "processed": count,
        "mean_f1_delta": score_delta_sum / count if count else 0.0,
        "wins": consensus_wins,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crea chosen di consenso dalle annotazioni VideoXum e coppie DPO."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--timeline-unit", choices=TIMELINE_UNITS, default="frame",
        help="frame: label su N frame; pair: label su ceil(N/2) coppie di frame, "
             "come le posizioni temporali di Qwen3-VL (vsum_onehot resta su N frame)",
    )
    parser.add_argument("--max-intervals", type=int, default=12)
    parser.add_argument(
        "--interval-penalty", type=float, default=None,
        help="default 0.15 con frame, 0.075 con pair",
    )
    parser.add_argument(
        "--max-shift", type=int, default=None,
        help="default 8 con frame, 4 con pair (in segmenti)",
    )
    parser.add_argument("--hard-ratio", type=float, default=0.50)
    parser.add_argument("--intermediate-ratio", type=float, default=0.30)
    parser.add_argument("--overcoverage-ratio", type=float, default=0.20)
    parser.add_argument(
        "--overcoverage-profile",
        choices=("mixed", "uniform"),
        default="mixed",
    )
    parser.add_argument("--overcoverage-min-ratio", type=float, default=0.80)
    parser.add_argument("--overcoverage-max-ratio", type=float, default=1.00)
    parser.add_argument(
        "--overcoverage-band-ratios",
        type=float,
        nargs=3,
        default=(0.50, 0.30, 0.20),
        metavar=("LOW", "MEDIUM", "HIGH"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name, value in UNIT_DEFAULTS[args.timeline_unit].items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit deve essere positivo")
    if args.max_intervals <= 0 or args.max_shift <= 0:
        raise SystemExit(
            "--max-intervals e --max-shift devono essere positivi"
        )
    if args.interval_penalty < 0:
        raise SystemExit("--interval-penalty non puo' essere negativo")
    ratios = (
        args.hard_ratio,
        args.intermediate_ratio,
        args.overcoverage_ratio,
    )
    if (
        any(ratio < 0 or ratio > 1 for ratio in ratios)
        or not math.isclose(sum(ratios), 1.0, abs_tol=1e-9)
    ):
        raise SystemExit(
            "le proporzioni hard/intermediate/overcoverage devono sommare a 1"
        )
    if not (
        0
        < args.overcoverage_min_ratio
        <= args.overcoverage_max_ratio
        <= 1
    ):
        raise SystemExit(
            "i rapporti overcoverage devono rispettare 0 < min <= max <= 1"
        )
    band_ratios = args.overcoverage_band_ratios
    if (
        any(ratio < 0 or ratio > 1 for ratio in band_ratios)
        or not math.isclose(sum(band_ratios), 1.0, abs_tol=1e-9)
    ):
        raise SystemExit(
            "le tre quote delle bande overcoverage devono sommare a 1"
        )
    args.output = resolve_output_path(args.output)
    if args.input.resolve() == args.output.resolve():
        raise SystemExit("Input e output devono essere diversi")

    with args.input.open(encoding="utf-8") as input_file:
        payload = json.load(input_file)

    add_medoid_labels(payload, args.limit)
    stats = add_consensus_labels(
        payload,
        args.limit,
        args.max_intervals,
        args.interval_penalty,
        args.timeline_unit,
    )
    records = list(payload.values() if isinstance(payload, dict) else payload)
    if args.limit is not None:
        records = records[:args.limit]
    if args.timeline_unit == "pair":
        # add_dpo_pairs costruisce e valuta i rejected contro vsum_onehot: per
        # tenerli sulla stessa timeline del chosen gli si passano temporaneamente
        # le annotazioni a coppie. vsum_onehot originale (N frame) viene
        # ripristinato subito dopo, perche' serve alle metriche di training.
        original_annotations = [record["vsum_onehot"] for record in records]
        for record in records:
            record["vsum_onehot"] = pair_annotations(record["vsum_onehot"])
    try:
        counts = add_dpo_pairs(
            payload,
            args.limit,
            seed=args.seed,
            max_shift=args.max_shift,
            overcoverage_min_ratio=args.overcoverage_min_ratio,
            overcoverage_max_ratio=args.overcoverage_max_ratio,
            hard_ratio=args.hard_ratio,
            intermediate_ratio=args.intermediate_ratio,
            overcoverage_ratio=args.overcoverage_ratio,
            overcoverage_profile=args.overcoverage_profile,
            overcoverage_band_ratios=band_ratios,
            chosen_field="vsum_onehot_consensus",
        )
    finally:
        if args.timeline_unit == "pair":
            for record, annotations in zip(records, original_annotations):
                record["vsum_onehot"] = annotations
    for record in records:
        # dpo_chosen_f1_mean/dpo_rejected_f1_mean sono calcolati su questa unita'.
        record["dpo_scores_timeline_unit"] = args.timeline_unit
    write_json_atomically(payload, args.output)
    print(
        f"Elaborati {stats['processed']} video; consensus >= medoid in "
        f"{stats['wins']}/{stats['processed']}, delta F1 medio="
        f"{stats['mean_f1_delta']:+.4f}. Rejected: hard={counts['hard']}, "
        f"intermediate={counts['intermediate']}, "
        f"overcoverage={counts['overcoverage']}. Risultato: {args.output}"
    )


if __name__ == "__main__":
    main()


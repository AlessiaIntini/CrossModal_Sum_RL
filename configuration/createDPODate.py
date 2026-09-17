#!/usr/bin/env python3
"""Calcola chosen e rejected VideoXum per il training DPO.

Per ogni video, ``vsum_onehot`` contiene 10 annotazioni binarie, una per
 frame. Il chosen e' il medoid temporale; il rejected e' creato con difficolta'
 intermedia, difficile oppure con over-coverage (molti frame extra).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Sequence


DEFAULT_INPUT = Path(
    "/leonardo_work/IscrC_HydroFM/aintini0/datasets/videoxum/metadata/"
    "train_videoxum.json"
)
DEFAULT_OUTPUT = Path(
    "/leonardo_work/IscrC_HydroFM/aintini0/datasets/videoxum/metadata/derived/"
    "train_videoxum_dpo.json"
)


def temporal_iou(first: Sequence[int], second: Sequence[int]) -> float:
    """Restituisce la IoU tra gli insiemi di frame selezionati.

    Se entrambe le annotazioni sono vuote, esse sono considerate identiche e
    la IoU vale 1.0.
    """
    if len(first) != len(second):
        raise ValueError(
            f"Annotazioni con lunghezze diverse: {len(first)} e {len(second)}"
        )

    intersection = sum(a == 1 and b == 1 for a, b in zip(first, second))
    union = sum(a == 1 or b == 1 for a, b in zip(first, second))
    return intersection / union if union else 1.0

def binary_f1(first: Sequence[int], second: Sequence[int]) -> float:
    selected_first = sum(first)
    selected_second = sum(second)
    true_positive = sum(
        a == 1 and b == 1 for a, b in zip(first, second)
    )

    denominator = selected_first + selected_second
    if denominator == 0:
        return 1.0
    return 2.0 * true_positive / denominator


def mean_annotation_f1(
    prediction: Sequence[int],
    annotations: Sequence[Sequence[int]],
) -> float:
    return sum(
        binary_f1(prediction, annotation)
        for annotation in annotations
    ) / len(annotations)

def temporal_medoid(
    annotations: Sequence[Sequence[int]],
) -> tuple[list[int], int, float, list[float]]:
    """Sceglie l'annotazione con la IoU media piu' alta verso le altre."""
    if not annotations:
        raise ValueError("vsum_onehot non contiene annotazioni")

    frame_count = len(annotations[0])
    for index, annotation in enumerate(annotations):
        if len(annotation) != frame_count:
            raise ValueError(
                f"L'annotazione {index} ha {len(annotation)} frame; "
                f"attesi {frame_count}"
            )
        invalid_values = set(annotation) - {0, 1}
        if invalid_values:
            raise ValueError(
                f"L'annotazione {index} contiene valori non binari: "
                f"{sorted(invalid_values, key=str)}"
            )

    annotation_count = len(annotations)
    if annotation_count == 1:
        return list(annotations[0]), 0, 1.0, [1.0]

    iou_sums = [0.0] * annotation_count
    for first_index in range(annotation_count):
        for second_index in range(first_index + 1, annotation_count):
            score = temporal_iou(
                annotations[first_index], annotations[second_index]
            )
            iou_sums[first_index] += score
            iou_sums[second_index] += score

    mean_scores = [
        score_sum / (annotation_count - 1) for score_sum in iou_sums
    ]
    # max() mantiene il primo indice in caso di parita'.
    medoid_index = max(range(annotation_count), key=mean_scores.__getitem__)
    return (
        list(annotations[medoid_index]),
        medoid_index,
        mean_scores[medoid_index],
        mean_scores,
    )


def add_medoid_labels(payload: Any, limit: int | None = None) -> int:
    """Aggiunge i campi del medoid ai record e restituisce quanti ne elabora."""
    if isinstance(payload, dict):
        records = payload.items()
    elif isinstance(payload, list):
        records = enumerate(payload)
    else:
        raise ValueError("Il JSON deve contenere un oggetto o una lista di video")

    processed = 0
    for record_key, record in records:
        if limit is not None and processed >= limit:
            break
        if not isinstance(record, dict):
            raise ValueError(f"Il record {record_key!r} non e' un oggetto JSON")
        if "vsum_onehot" not in record:
            raise ValueError(f"Campo vsum_onehot mancante nel record {record_key!r}")

        try:
            label, medoid_index, medoid_score, all_scores = temporal_medoid(
                record["vsum_onehot"]
            )
        except ValueError as error:
            raise ValueError(f"Record {record_key!r}: {error}") from error

        sampled_frames = record.get("sampled_frames")
        if sampled_frames is not None and len(label) != sampled_frames:
            raise ValueError(
                f"Record {record_key!r}: label di {len(label)} frame, "
                f"ma sampled_frames vale {sampled_frames}"
            )

        record["vsum_onehot_medoid"] = label
        record["vsum_onehot_medoid_index"] = medoid_index
        record["vsum_onehot_medoid_score"] = medoid_score
        record["vsum_onehot_mean_iou_scores"] = all_scores
        processed += 1

    return processed



def _selected_segments(label: Sequence[int]) -> list[tuple[int, int]]:
    """Restituisce gli intervalli inclusivi dei segmenti selezionati."""
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(label):
        if value == 1 and start is None:
            start = index
        elif value == 0 and start is not None:
            segments.append((start, index - 1))
            start = None
    if start is not None:
        segments.append((start, len(label) - 1))
    return segments


def _indices_to_label(indices: Sequence[int], frame_count: int) -> list[int]:
    label = [0] * frame_count
    for index in indices:
        label[index] = 1
    return label


def create_intermediate_rejected(
    chosen: Sequence[int], rng: random.Random, max_shift: int = 8
) -> list[int]:
    """Trasla leggermente il chosen preservandone esattamente la cardinalita'."""
    selected = [index for index, value in enumerate(chosen) if value == 1]
    if not selected or len(selected) == len(chosen):
        raise ValueError("Il chosen deve contenere sia frame scelti sia non scelti")

    # shift = min(max_shift, max(1, round(len(chosen) * 0.02)))
    shift = max(2, min(max_shift, round(len(chosen) * 0.05)))
    candidates: list[list[int]] = []

    # La proiezione monotona evita wrap-around e perdita di frame ai bordi.
    for direction in (-1, 1):
        translated: list[int] = []
        if direction < 0:
            lower_bound = 0
            for index in selected:
                new_index = max(index - shift, lower_bound)
                translated.append(new_index)
                lower_bound = new_index + 1
        else:
            upper_bound = len(chosen) - 1
            reversed_translation: list[int] = []
            for index in reversed(selected):
                new_index = min(index + shift, upper_bound)
                reversed_translation.append(new_index)
                upper_bound = new_index - 1
            translated = list(reversed(reversed_translation))

        if translated != selected:
            candidates.append(translated)

    if not candidates:
        raise ValueError("Impossibile traslare il chosen")
    return _indices_to_label(rng.choice(candidates), len(chosen))


def create_hard_rejected(
    chosen: Sequence[int],
    annotations: Sequence[Sequence[int]],
    rng: random.Random,
) -> list[int]:
    """Sostituisce un evento importante mantenendo gli altri segmenti giusti.

    Il sostituto e' preso tra i frame non presenti nel medoid, privilegiando
    le zone che hanno comunque ricevuto voti dagli altri annotatori. In questo
    modo il negativo resta plausibile e non usa solo frame irrilevanti.
    """
    frame_count = len(chosen)
    selected_count = sum(chosen)
    if not selected_count or selected_count == frame_count:
        raise ValueError("Il chosen deve contenere sia frame scelti sia non scelti")

    votes = [0] * frame_count
    for annotation_index, annotation in enumerate(annotations):
        if len(annotation) != frame_count:
            raise ValueError(
                f"L'annotazione {annotation_index} ha {len(annotation)} frame; "
                f"attesi {frame_count}"
            )
        for index, value in enumerate(annotation):
            votes[index] += int(value == 1)

    chosen_segments = _selected_segments(chosen)
    empty_segments = _selected_segments([1 - value for value in chosen])
    max_empty_length = max(end - start + 1 for start, end in empty_segments)

    removable = [
        segment
        for segment in chosen_segments
        if segment[1] - segment[0] + 1 <= max_empty_length
    ]
    if removable:
        # Rimuove l'evento mediamente piu' condiviso. A parita' preferisce un
        # evento breve, per ottenere un errore difficile da riconoscere.
        def importance(segment: tuple[int, int]) -> tuple[float, int, float]:
            start, end = segment
            length = end - start + 1
            return (
                sum(votes[start : end + 1]) / length,
                -length,
                rng.random(),
            )

        remove_start, remove_end = max(removable, key=importance)
    else:
        # Caso raro: nessun gap contiene un evento intero. Ne rimuove una
        # porzione coerente, continuando a conservare il numero totale di frame.
        source_start, source_end = max(
            chosen_segments, key=lambda segment: segment[1] - segment[0]
        )
        remove_start = source_start
        remove_end = min(source_end, source_start + max_empty_length - 1)

    replacement_length = remove_end - remove_start + 1
    windows: list[tuple[int, int, int, float]] = []
    for gap_start, gap_end in empty_segments:
        last_start = gap_end - replacement_length + 1
        for window_start in range(gap_start, last_start + 1):
            window_end = window_start + replacement_length - 1
            windows.append(
                (
                    sum(votes[window_start : window_end + 1]),
                    window_start,
                    window_end,
                    rng.random(),
                )
            )

    if not windows:
        raise ValueError("Impossibile trovare un segmento rejected sostitutivo")
    chosen_score = mean_annotation_f1(chosen, annotations)
    valid_candidates = []

    for _, replacement_start, replacement_end, _ in windows:
        candidate = list(chosen)

        for index in range(remove_start, remove_end + 1):
            candidate[index] = 0

        for index in range(replacement_start, replacement_end + 1):
            candidate[index] = 1

        candidate_score = mean_annotation_f1(candidate, annotations)
        margin = chosen_score - candidate_score

        if margin >= 0.05:
            valid_candidates.append((candidate_score, rng.random(), candidate))

    if not valid_candidates:
        raise ValueError(
        "Nessun rejected hard con margine F1 di almeno 0.05"
    )

    # Il più plausibile fra quelli certamente peggiori.
    _, _, rejected = max(
        valid_candidates,
        key=lambda item: (item[0], item[1]),
    )
    return rejected


def create_overcoverage_rejected(
    chosen: Sequence[int],
    rng: random.Random,
    min_ratio: float = 0.80,
    max_ratio: float = 1.00,
) -> list[int]:
    """Aggiunge frame fino a coprire gran parte della timeline.

    Questo negativo ha intenzionalmente una cardinalita' maggiore del chosen:
    espone il DPO a una coppia che distingue la qualita' anche lungo la
    dimensione della compressione.
    """
    frame_count = len(chosen)
    selected = {index for index, value in enumerate(chosen) if value == 1}
    available = [index for index, value in enumerate(chosen) if value == 0]
    if not selected or not available:
        raise ValueError(
            "Il chosen deve contenere sia frame scelti sia frame non scelti"
        )
    if not 0 < min_ratio <= max_ratio <= 1:
        raise ValueError("i rapporti di over-coverage devono stare in (0, 1]")

    target = round(frame_count * rng.uniform(min_ratio, max_ratio))
    target = max(len(selected) + 1, min(frame_count, target))
    extra_count = target - len(selected)
    rejected = list(chosen)
    # for index in rng.sample(available, extra_count):
    #     rejected[index] = 1
    distance_to_chosen = {
    index: min(abs(index - selected_index) for selected_index in selected)
    for index in available
    }

    ranked_available = sorted(
    available,
    key=lambda index: (
        distance_to_chosen[index],
        rng.random(),
    ),
    )

    for index in ranked_available[:extra_count]:
        rejected[index] = 1

    return rejected


def add_dpo_pairs(
    payload: Any,
    limit: int | None = None,
    seed: int = 42,
    max_shift: int = 2,
    overcoverage_min_ratio: float = 0.80,
    overcoverage_max_ratio: float = 1.00,
    hard_ratio: float = 0.50,
    intermediate_ratio: float = 0.30,
    overcoverage_ratio: float = 0.20,
    overcoverage_profile: str = "mixed",
    overcoverage_band_ratios: Sequence[float] = (0.50, 0.30, 0.20),
    chosen_field: str = "vsum_onehot_medoid",
) -> dict[str, int]:
    """Aggiunge chosen/rejected nelle proporzioni richieste."""
    if isinstance(payload, dict):
        records = list(payload.items())
    elif isinstance(payload, list):
        records = list(enumerate(payload))
    else:
        raise ValueError("Il JSON deve contenere un oggetto o una lista di video")
    if limit is not None:
        records = records[:limit]

    # Essere multi-evento non basta: per alcuni video nessuna sostituzione
    # produce il margine F1 minimo. Precalcoliamo quindi i rejected hard e
    # candidiamo soltanto i record per cui ne esiste davvero uno valido.
    hard_rejected_by_position: dict[int, list[int]] = {}
    for position, (record_key, record) in enumerate(records):
        if (
            not isinstance(record, dict)
            or len(_selected_segments(record.get(chosen_field, []))) < 2
        ):
            continue
        candidate_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{record_key}".encode()).digest()[:8], "big"
        )
        try:
            hard_rejected_by_position[position] = create_hard_rejected(
                record[chosen_field],
                record["vsum_onehot"],
                random.Random(candidate_seed),
            )
        except ValueError:
            # Il record resta utilizzabile per intermediate/overcoverage.
            continue
    hard_candidates = list(hard_rejected_by_position)
    random.Random(seed).shuffle(hard_candidates)
    hard_target = round(len(records) * hard_ratio)
    overcoverage_target = round(len(records) * overcoverage_ratio)
    intermediate_target = len(records) - hard_target - overcoverage_target
    if intermediate_target < 0:
        raise ValueError("Le proporzioni richieste producono conteggi non validi")
    if len(hard_candidates) < hard_target:
        raise ValueError(
            f"Non ci sono abbastanza video multi-evento per creare {hard_target} "
            f"rejected hard (disponibili: {len(hard_candidates)})"
        )
    hard_indices = set(hard_candidates[:hard_target])
    remaining = [
        position for position in range(len(records))
        if position not in hard_indices
    ]
    intermediate_rejected_by_position: dict[int, list[int]] = {}
    for position in remaining:
        record_key, record = records[position]
        candidate_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{record_key}".encode()).digest()[:8], "big"
        )
        try:
            candidate = create_intermediate_rejected(
                record[chosen_field], random.Random(candidate_seed), max_shift
            )
        except ValueError:
            continue
        chosen_score = mean_annotation_f1(
            record[chosen_field], record["vsum_onehot"]
        )
        rejected_score = mean_annotation_f1(candidate, record["vsum_onehot"])
        if chosen_score - rejected_score >= 0.05:
            intermediate_rejected_by_position[position] = candidate

    intermediate_candidates = list(intermediate_rejected_by_position)
    random.Random(seed + 1).shuffle(intermediate_candidates)
    if len(intermediate_candidates) < intermediate_target:
        raise ValueError(
            f"Non ci sono abbastanza record con un rejected intermediate valido "
            f"per crearne {intermediate_target} (disponibili dopo gli hard: "
            f"{len(intermediate_candidates)})"
        )
    intermediate_indices = set(intermediate_candidates[:intermediate_target])
    overcoverage_indices = set(remaining) - intermediate_indices
    if len(overcoverage_indices) != overcoverage_target:
        raise AssertionError("Allocazione hard/intermediate/overcoverage incoerente")
    overcoverage_bands: dict[int, tuple[str, float, float]] = {}
    if overcoverage_profile == "mixed":
        ordered_overcoverage = sorted(overcoverage_indices)
        random.Random(seed + 2).shuffle(ordered_overcoverage)
        first_count = round(overcoverage_target * overcoverage_band_ratios[0])
        second_count = round(overcoverage_target * overcoverage_band_ratios[1])
        # L'ultima banda assorbe l'eventuale residuo di arrotondamento.
        first_count = min(first_count, overcoverage_target)
        second_count = min(second_count, overcoverage_target - first_count)
        band_specs = (
            ("30-40", 0.30, 0.40, first_count),
            ("40-60", 0.40, 0.60, second_count),
            (
                "70-100", 0.70, 1.00,
                overcoverage_target - first_count - second_count,
            ),
        )
        cursor = 0
        for band_name, band_min, band_max, band_count in band_specs:
            for record_position in ordered_overcoverage[cursor:cursor + band_count]:
                overcoverage_bands[record_position] = (
                    band_name, band_min, band_max
                )
            cursor += band_count
    counts = {"intermediate": 0, "hard": 0, "overcoverage": 0}
    band_counts = {"30-40": 0, "40-60": 0, "70-100": 0}

    for position, (record_key, record) in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Il record {record_key!r} non e' un oggetto JSON")
        if chosen_field not in record:
            raise ValueError(
                f"Campo {chosen_field} mancante nel record {record_key!r}"
            )

        chosen = list(record[chosen_field])
        record_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{record_key}".encode()).digest()[:8], "big"
        )
        rng = random.Random(record_seed)
        if position in hard_indices:
            difficulty = "hard"
        elif position in intermediate_indices:
            difficulty = "intermediate"
        else:
            difficulty = "overcoverage"
        try:
            if difficulty == "intermediate":
                rejected = intermediate_rejected_by_position[position]
            elif difficulty == "hard":
                rejected = hard_rejected_by_position[position]
            else:
                if overcoverage_profile == "mixed":
                    band_name, coverage_min, coverage_max = overcoverage_bands[position]
                else:
                    band_name = "uniform"
                    coverage_min = overcoverage_min_ratio
                    coverage_max = overcoverage_max_ratio
                rejected = create_overcoverage_rejected(
                    chosen, rng, coverage_min, coverage_max
                )
        except ValueError as error:
            raise ValueError(f"Record {record_key!r}: {error}") from error

        chosen_score = mean_annotation_f1(
        chosen, record["vsum_onehot"])
        rejected_score = mean_annotation_f1(
        rejected, record["vsum_onehot"])
        preference_margin = chosen_score - rejected_score

        if rejected == chosen:
            raise AssertionError(f"Record {record_key!r}: chosen e rejected coincidono")
        if difficulty != "overcoverage" and sum(rejected) != sum(chosen):
            raise AssertionError(
                f"Record {record_key!r}: chosen e rejected hanno cardinalita' diverse"
            )
        if difficulty == "overcoverage" and sum(rejected) <= sum(chosen):
            raise AssertionError(
                f"Record {record_key!r}: overcoverage non aggiunge frame"
            )

        record["vsum_onehot_chosen"] = chosen
        record["vsum_onehot_rejected"] = rejected
        record["dpo_rejected_difficulty"] = difficulty
        record["dpo_chosen_rejected_iou"] = temporal_iou(chosen, rejected)
        record["dpo_chosen_f1_mean"] = chosen_score
        record["dpo_rejected_f1_mean"] = rejected_score
        record["dpo_preference_margin"] = preference_margin
        # Per hard/intermediate il rejected deve essere peggiore anche rispetto
        # alle annotazioni. Un overcoverage, invece, contiene il chosen e frame
        # extra: soprattutto nella banda 30-40% puo' aumentare il recall e avere
        # un F1 simile o superiore. In quel caso la preferenza DPO deriva dalla
        # violazione della concisione, quindi il margine resta una diagnostica
        # ma non e' un vincolo di validita'.
        if difficulty != "overcoverage" and preference_margin < 0.05:
            raise ValueError(
                f"Record {record_key!r} ({difficulty}): margine DPO "
                f"insufficiente: chosen={chosen_score:.4f}, "
                f"rejected={rejected_score:.4f}, "
                f"margin={preference_margin:.4f}"
            )
        if difficulty == "overcoverage":
            record["dpo_overcoverage_band"] = band_name
            record["dpo_overcoverage_ratio"] = sum(rejected) / len(rejected)
            if band_name in band_counts:
                band_counts[band_name] += 1
        else:
            record.pop("dpo_overcoverage_band", None)
            record.pop("dpo_overcoverage_ratio", None)
        counts[difficulty] += 1

    counts.update({f"overcoverage_{name}": value for name, value in band_counts.items()})
    return counts


def write_json_atomically(payload: Any, output_path: Path) -> None:
    """Scrive il risultato senza lasciare un file parziale in caso di errore."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_file:
            temporary_path = output_file.name
            json.dump(payload, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")
        os.replace(temporary_path, output_path)
    except BaseException:
        if temporary_path is not None:
            Path(temporary_path).unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crea coppie DPO: chosen medoid e rejected temporalmente plausibile "
            "con lo stesso numero di frame."
        )
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help=f"JSON VideoXum sorgente (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"JSON arricchito da creare (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--limit", type=int,
        help="Elabora solo i primi N video (utile per test veloci)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed per generazione e bilanciamento dei rejected (default: 42)",
    )
    parser.add_argument(
    "--max-shift",
    type=int,
    default=8,
    help="Traslazione massima dei rejected intermediate (default: 8)",
)
    parser.add_argument(
        "--overcoverage-min-ratio", type=float, default=0.80,
        help="Copertura minima della timeline per gli overcoverage (default: 0.80)",
    )
    parser.add_argument(
        "--overcoverage-max-ratio", type=float, default=1.00,
        help="Copertura massima della timeline per gli overcoverage (default: 1.00)",
    )
    parser.add_argument(
        "--hard-ratio", type=float, default=0.50,
        help="Proporzione di rejected hard (default: 0.50)",
    )
    parser.add_argument(
        "--intermediate-ratio", type=float, default=0.30,
        help="Proporzione di rejected intermediate (default: 0.30)",
    )
    parser.add_argument(
        "--overcoverage-ratio", type=float, default=0.20,
        help="Proporzione di rejected overcoverage (default: 0.20)",
    )
    parser.add_argument(
        "--overcoverage-profile", choices=("mixed", "uniform"), default="mixed",
        help=(
            "mixed usa le bande 30-40/40-60/70-100; uniform usa "
            "--overcoverage-min-ratio e --overcoverage-max-ratio (default: mixed)"
        ),
    )
    parser.add_argument(
        "--overcoverage-band-ratios", type=float, nargs=3,
        metavar=("LOW", "MEDIUM", "HIGH"), default=(0.50, 0.30, 0.20),
        help=(
            "quote interne alle bande 30-40, 40-60 e 70-100 "
            "(default: 0.50 0.30 0.20)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit deve essere un intero positivo")
    if args.max_shift <= 0:
        raise SystemExit("--max-shift deve essere un intero positivo")
    if not 0 < args.overcoverage_min_ratio <= args.overcoverage_max_ratio <= 1:
        raise SystemExit(
            "i rapporti di over-coverage devono rispettare 0 < min <= max <= 1"
        )
    type_ratios = (
        args.hard_ratio,
        args.intermediate_ratio,
        args.overcoverage_ratio,
    )
    if any(ratio < 0 or ratio > 1 for ratio in type_ratios):
        raise SystemExit("le proporzioni dei rejected devono essere comprese tra 0 e 1")
    if not math.isclose(sum(type_ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise SystemExit(
            "--hard-ratio + --intermediate-ratio + --overcoverage-ratio "
            f"deve essere 1.0 (somma ricevuta: {sum(type_ratios):.12g})"
        )
    if any(ratio < 0 or ratio > 1 for ratio in args.overcoverage_band_ratios):
        raise SystemExit("le quote delle bande overcoverage devono stare tra 0 e 1")
    if not math.isclose(
        sum(args.overcoverage_band_ratios), 1.0, rel_tol=0.0, abs_tol=1e-9
    ):
        raise SystemExit(
            "le tre quote di --overcoverage-band-ratios devono sommare a 1.0"
        )
    if args.input.resolve() == args.output.resolve():
        raise SystemExit("Input e output devono essere file diversi")

    with args.input.open(encoding="utf-8") as input_file:
        payload = json.load(input_file)

    processed = add_medoid_labels(payload, args.limit)
    difficulty_counts = add_dpo_pairs(
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
        overcoverage_band_ratios=args.overcoverage_band_ratios,
    )
    write_json_atomically(payload, args.output)
    print(
        f"Elaborati {processed} video: "
        f"{difficulty_counts['intermediate']} rejected intermediate, "
        f"{difficulty_counts['hard']} rejected hard, "
        f"{difficulty_counts['overcoverage']} rejected overcoverage. "
        f"Bande overcoverage: 30-40%={difficulty_counts['overcoverage_30-40']}, "
        f"40-60%={difficulty_counts['overcoverage_40-60']}, "
        f"70-100%={difficulty_counts['overcoverage_70-100']}. "
        f"Risultato salvato in: {args.output}"
    )


if __name__ == "__main__":
    main()

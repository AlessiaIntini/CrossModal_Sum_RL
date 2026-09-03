#!/usr/bin/env python3
"""Calcola chosen e rejected VideoXum per il training DPO.

Per ogni video, ``vsum_onehot`` contiene 10 annotazioni binarie, una per
frame. Il chosen e' il medoid temporale; il rejected ha lo stesso numero di
frame selezionati ed e' creato con difficolta' intermedia oppure difficile.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Sequence


DEFAULT_INPUT = Path(
    "/leonardo_work/IscrC_MSMU/aintini0/datasets/videoxum/metadata/"
    "train_videoxum.json"
)
DEFAULT_OUTPUT = Path(
    "/leonardo_work/IscrC_MSMU/aintini0/datasets/videoxum/metadata/derived/"
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
    chosen: Sequence[int], rng: random.Random, max_shift: int = 2
) -> list[int]:
    """Trasla leggermente il chosen preservandone esattamente la cardinalita'."""
    selected = [index for index, value in enumerate(chosen) if value == 1]
    if not selected or len(selected) == len(chosen):
        raise ValueError("Il chosen deve contenere sia frame scelti sia non scelti")

    shift = min(max_shift, max(1, round(len(chosen) * 0.02)))
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
    _, replacement_start, replacement_end, _ = max(
        windows, key=lambda item: (item[0], item[3])
    )

    rejected = list(chosen)
    for index in range(remove_start, remove_end + 1):
        rejected[index] = 0
    for index in range(replacement_start, replacement_end + 1):
        rejected[index] = 1
    return rejected


def add_dpo_pairs(
    payload: Any,
    limit: int | None = None,
    seed: int = 42,
    max_shift: int = 2,
) -> dict[str, int]:
    """Aggiunge chosen/rejected bilanciati tra intermediate e hard."""
    if isinstance(payload, dict):
        records = list(payload.items())
    elif isinstance(payload, list):
        records = list(enumerate(payload))
    else:
        raise ValueError("Il JSON deve contenere un oggetto o una lista di video")
    if limit is not None:
        records = records[:limit]

    # Un hard deve poter conservare almeno un intero evento corretto.
    # I record con un solo segmento vengono quindi assegnati a intermediate;
    # gli altri sono mescolati per evitare correlazioni con l'ordine del JSON.
    hard_candidates = [
        position
        for position, (_, record) in enumerate(records)
        if isinstance(record, dict)
        and len(_selected_segments(record.get("vsum_onehot_medoid", []))) >= 2
    ]
    random.Random(seed).shuffle(hard_candidates)
    hard_target = len(records) // 2
    if len(hard_candidates) < hard_target:
        raise ValueError(
            "Non ci sono abbastanza video multi-evento per creare il 50% di hard"
        )
    hard_indices = set(hard_candidates[:hard_target])
    counts = {"intermediate": 0, "hard": 0}

    for position, (record_key, record) in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Il record {record_key!r} non e' un oggetto JSON")
        if "vsum_onehot_medoid" not in record:
            raise ValueError(
                f"Campo vsum_onehot_medoid mancante nel record {record_key!r}"
            )

        chosen = list(record["vsum_onehot_medoid"])
        record_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{record_key}".encode()).digest()[:8], "big"
        )
        rng = random.Random(record_seed)
        difficulty = "hard" if position in hard_indices else "intermediate"
        try:
            if difficulty == "intermediate":
                rejected = create_intermediate_rejected(chosen, rng, max_shift)
            else:
                rejected = create_hard_rejected(chosen, record["vsum_onehot"], rng)
        except ValueError as error:
            raise ValueError(f"Record {record_key!r}: {error}") from error

        if rejected == chosen:
            raise AssertionError(f"Record {record_key!r}: chosen e rejected coincidono")
        if sum(rejected) != sum(chosen):
            raise AssertionError(
                f"Record {record_key!r}: chosen e rejected hanno cardinalita' diverse"
            )

        record["vsum_onehot_chosen"] = chosen
        record["vsum_onehot_rejected"] = rejected
        record["dpo_rejected_difficulty"] = difficulty
        record["dpo_chosen_rejected_iou"] = temporal_iou(chosen, rejected)
        counts[difficulty] += 1

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
        "--max-shift", type=int, default=2,
        help="Traslazione massima dei rejected intermediate (default: 2)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit deve essere un intero positivo")
    if args.max_shift <= 0:
        raise SystemExit("--max-shift deve essere un intero positivo")
    if args.input.resolve() == args.output.resolve():
        raise SystemExit("Input e output devono essere file diversi")

    with args.input.open(encoding="utf-8") as input_file:
        payload = json.load(input_file)

    processed = add_medoid_labels(payload, args.limit)
    difficulty_counts = add_dpo_pairs(
        payload, args.limit, seed=args.seed, max_shift=args.max_shift
    )
    write_json_atomically(payload, args.output)
    print(
        f"Elaborati {processed} video: "
        f"{difficulty_counts['intermediate']} rejected intermediate, "
        f"{difficulty_counts['hard']} rejected hard. "
        f"Risultato salvato in: {args.output}"
    )


if __name__ == "__main__":
    main()

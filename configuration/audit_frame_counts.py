#!/usr/bin/env python3
"""Verifica che la decodifica a 1 fps produca esattamente sampled_frames frame.

decode_video_frames (sft_train_videoxum.py) non controlla il numero di frame:
se un video ne produce meno del previsto, il modello vede una timeline piu'
corta di quella a cui si riferiscono prompt e label. Questo script conta i
frame con la stessa logica, senza convertirli in immagini, su un campione di
video. Usa solo CPU.

Esempio:
  python configuration/audit_frame_counts.py --samples 300 --workers 8 \
      --output results/audit_frame_counts.jsonl
"""

import argparse
import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import av

DATA_ROOT = Path("/leonardo_work/IscrC_HydroFM/aintini0/datasets/videoxum")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v")


def resolve_video(video_dir, video_id):
    for extension in ("",) + VIDEO_EXTENSIONS:
        candidate = video_dir / f"{video_id}{extension}"
        if candidate.is_file():
            return candidate
    return None


def count_frames_at_1fps(path, expected):
    """Stessa regola di decode_video_frames: un frame per secondo trascorso."""
    decoded, first, next_second = 0, None, 0
    with av.open(str(path)) as container:
        for frame in container.decode(container.streams.video[0]):
            if frame.pts is None:
                continue
            timestamp = float(frame.pts * frame.time_base)
            if first is None:
                first = timestamp
            if timestamp - first + 1e-6 >= next_second:
                decoded += 1
                next_second += 1
                if decoded == expected:
                    break
    return decoded


def audit(video_id, path, expected):
    try:
        decoded = count_frames_at_1fps(path, expected)
        return {"video_id": video_id, "expected": expected, "decoded": decoded,
                "delta": decoded - expected}
    except Exception as error:  # video corrotto: va riportato, non nascosto
        return {"video_id": video_id, "expected": expected, "error": str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--metadata", type=Path, default=DATA_ROOT / "metadata" / "train_videoxum.json")
    parser.add_argument("--video_dir", type=Path, default=DATA_ROOT / "train")
    parser.add_argument("--samples", type=int, default=300, help="0 controlla tutti i video")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.metadata.read_text(encoding="utf-8"))
    items = list(payload.items()) if isinstance(payload, dict) else [
        (row.get("video_id") or index, row) for index, row in enumerate(payload)
    ]
    random.Random(args.seed).shuffle(items)
    if args.samples:
        items = items[:args.samples]

    jobs, missing = [], 0
    for key, record in items:
        video_id = str(record.get("video_id") or key)
        path = resolve_video(args.video_dir, video_id)
        if path is None:
            missing += 1
            continue
        jobs.append((video_id, path, int(record["sampled_frames"])))

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(audit, *job) for job in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if done % 50 == 0:
                print(f"[audit] {done}/{len(jobs)}", flush=True)

    errors = [row for row in results if "error" in row]
    checked = [row for row in results if "error" not in row]
    short = [row for row in checked if row["delta"] < 0]
    print(f"\nvideo controllati={len(checked)} mancanti={missing} errori={len(errors)}")
    print(f"con meno frame del previsto={len(short)} ({100 * len(short) / max(len(checked), 1):.1f}%)")
    if short:
        deltas = sorted(row["delta"] for row in short)
        print(f"deficit: minimo={deltas[-1]} massimo={deltas[0]} mediano={deltas[len(deltas) // 2]}")
        for row in sorted(short, key=lambda row: row["delta"])[:10]:
            print(f"   {row['video_id']}: attesi={row['expected']} decodificati={row['decoded']}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            for row in results:
                handle.write(json.dumps(row) + "\n")
        print(f"[done] dettaglio: {args.output}")


if __name__ == "__main__":
    main()

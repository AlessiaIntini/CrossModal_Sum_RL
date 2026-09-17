#!/usr/bin/env python3
"""Estrae feature CLIP normalizzate dei frame VideoXum campionati a 1 FPS."""

import argparse
import json
from pathlib import Path

import av
import numpy as np
import torch
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Crea la cache frame-level CLIP per il calcolo di FCLIP"
    )
    parser.add_argument("--metadata-json", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_counts(path):
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    items = payload.items() if isinstance(payload, dict) else enumerate(payload)
    counts = {}
    for key, record in items:
        video_id = str(record.get("video_id", key))
        count = int(record["sampled_frames"])
        if count <= 0:
            raise ValueError(f"{video_id}: sampled_frames non positivo")
        counts[video_id] = count
    if not counts:
        raise ValueError(f"Nessun record in {path}")
    return counts


def resolve_video(video_dir, video_id):
    direct = video_dir / video_id
    if direct.is_file():
        return direct
    for extension in VIDEO_EXTENSIONS:
        candidate = video_dir / f"{video_id}{extension}"
        if candidate.is_file():
            return candidate
    return None


def decode_frames(video_path, expected):
    """Replica il campionamento VideoXum: un frame ai secondi 0, 1, ..., T-1."""
    frames, first_timestamp, next_second = [], None, 0
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            timestamp = float(frame.pts * frame.time_base)
            if first_timestamp is None:
                first_timestamp = timestamp
            if timestamp - first_timestamp + 1e-6 >= next_second:
                frames.append(frame.to_image())
                next_second += 1
                if len(frames) == expected:
                    break
    return frames


@torch.inference_mode()
def embed_frames(frames, processor, model, device, batch_size):
    embeddings = []
    for start in range(0, len(frames), batch_size):
        inputs = processor(
            images=frames[start:start + batch_size], return_tensors="pt"
        )
        pixel_values = inputs["pixel_values"].to(device)
        # get_image_features include la proiezione contrastiva CLIP richiesta
        # da FCLIP; il solo pooler_output non sarebbe nello spazio CLIP finale.
        features = model.get_image_features(pixel_values=pixel_values)
        # Alcune versioni di Transformers restituiscono qui direttamente il
        # tensore proiettato, altre restituiscono BaseModelOutputWithPooling
        # proveniente dal vision encoder. Nel secondo caso applichiamo
        # esplicitamente la visual projection di CLIP.
        if not isinstance(features, torch.Tensor):
            if not hasattr(features, "pooler_output"):
                raise TypeError(
                    "get_image_features ha restituito un oggetto senza "
                    "pooler_output: impossibile estrarre le feature CLIP"
                )
            pooled = features.pooler_output
            projection = model.visual_projection
            # In alcune release l'oggetto restituito contiene già il vettore
            # proiettato (B, projection_dim); in altre contiene il pooler
            # grezzo del vision encoder (B, vision_hidden_size). Applichiamo
            # la projection soltanto nel secondo caso, evitando una doppia
            # proiezione (ad es. 768 -> 1024).
            if pooled.shape[-1] == projection.in_features:
                features = projection(pooled)
            elif pooled.shape[-1] == projection.out_features:
                features = pooled
            else:
                raise ValueError(
                    "dimensione pooler CLIP inattesa: "
                    f"{pooled.shape[-1]} (attese {projection.in_features} "
                    f"o {projection.out_features})"
                )
        features = torch.nn.functional.normalize(features.float(), dim=-1)
        embeddings.append(features.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def cache_is_compatible(path, expected, model_id):
    try:
        with np.load(path) as cached:
            return (
                cached["features"].ndim == 2
                and cached["features"].shape[0] == expected
                and str(cached["encoder"]) == model_id
                and str(cached["feature_type"]) == "clip_projected_l2_normalized"
            )
    except (OSError, KeyError, ValueError):
        return False


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size deve essere positivo")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA richiesta ma non disponibile")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_id = str(args.model.resolve())
    processor = CLIPProcessor.from_pretrained(args.model, local_files_only=True)
    model = CLIPModel.from_pretrained(args.model, local_files_only=True)
    model = model.to(args.device).eval()
    counts = load_counts(args.metadata_json)

    generated = compatible = missing = mismatched = 0
    for video_id, expected in tqdm(counts.items(), desc="Embedding CLIP", unit="video"):
        output_path = args.output_dir / f"{video_id}.npz"
        if (output_path.is_file() and not args.overwrite
                and cache_is_compatible(output_path, expected, model_id)):
            compatible += 1
            continue
        video_path = resolve_video(args.video_dir, video_id)
        if video_path is None:
            print(f"[WARN] video mancante: {video_id}")
            missing += 1
            continue
        frames = decode_frames(video_path, expected)
        if len(frames) != expected:
            print(f"[WARN] {video_id}: frame estratti={len(frames)}, attesi={expected}")
            mismatched += 1
            continue
        features = embed_frames(
            frames, processor, model, args.device, args.batch_size
        ).astype(np.float32)
        np.savez_compressed(
            output_path,
            features=features,
            encoder=np.array(model_id),
            feature_type=np.array("clip_projected_l2_normalized"),
            count=np.array(expected, dtype=np.int64),
            sampling_fps=np.array(1.0, dtype=np.float32),
        )
        generated += 1

    print(
        f"[DONE] generati={generated} compatibili={compatible} "
        f"mancanti={missing} mismatch={mismatched} totale={len(counts)}"
    )
    if missing or mismatched:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

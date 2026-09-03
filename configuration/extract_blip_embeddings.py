#!/usr/bin/env python3
"""
extract_blip_embeddings.py

Estrae embedding visivi con BLIP dai video del dataset VideoXum.

Per ogni video:
  1. Campiona i frame a intervalli regolari (per FPS target o numero fisso di frame)
  2. Calcola l'embedding BLIP di ogni frame campionato
  3. Salva:
       - gli embedding per-frame (array [n_frame, dim])
       - l'embedding aggregato del video (mean pooling sui frame, [dim])

Uso tipico:
    python extract_blip_embeddings.py \
        --video_dir /path/to/videoxum/videos \
        --output_dir /path/to/output/embeddings \
        --sample_fps 1.0

oppure con un numero fisso di frame per video:
    python extract_blip_embeddings.py \
        --video_dir /path/to/videoxum/videos \
        --output_dir /path/to/output/embeddings \
        --num_frames 16

Requisiti:
    pip install torch transformers opencv-python numpy tqdm
"""

import argparse
import os
import sys
import glob
import numpy as np
import cv2
import torch
from tqdm import tqdm
from transformers import BlipForConditionalGeneration, BlipProcessor


VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov", ".webm")


def parse_args():
    p = argparse.ArgumentParser(description="Estrae embedding BLIP dai video di VideoXum")
    p.add_argument("--video_dir", type=str, required=True,
                    help="Cartella contenente i video di VideoXum (ricerca ricorsiva).")
    p.add_argument("--output_dir", type=str, required=True,
                    help="Cartella dove salvare gli embedding (.npz per video).")
    p.add_argument("--model_name", type=str, default="Salesforce/blip-image-captioning-base",
                    help="Nome/percorso del modello BLIP su HuggingFace Hub.")

    sampling = p.add_mutually_exclusive_group()
    sampling.add_argument("--sample_fps", type=float, default=1.0,
                           help="Frame al secondo da campionare (default: 1.0). "
                                "Ignorato se --num_frames è specificato.")
    sampling.add_argument("--num_frames", type=int, default=None,
                           help="Numero fisso di frame equispaziati da estrarre per video "
                                "(sovrascrive --sample_fps).")

    p.add_argument("--batch_size", type=int, default=32,
                    help="Numero di frame processati insieme dal modello.")
    p.add_argument("--device", type=str, default=None,
                    help="cuda / cpu / mps. Default: auto-detect.")
    p.add_argument("--save_per_frame", action="store_true", default=True,
                    help="Salva anche gli embedding per singolo frame (default: True).")
    p.add_argument("--no_save_per_frame", action="store_false", dest="save_per_frame",
                    help="Salva solo l'embedding aggregato del video, non i singoli frame.")
    p.add_argument("--overwrite", action="store_true",
                    help="Ricalcola anche se il file di output esiste già.")
    return p.parse_args()


def find_videos(video_dir):
    paths = []
    for ext in VIDEO_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(video_dir, "**", f"*{ext}"), recursive=True))
    return sorted(paths)


def sample_frame_indices(total_frames, fps, sample_fps=None, num_frames=None):
    """Restituisce gli indici dei frame da estrarre."""
    if total_frames <= 0:
        return []

    if num_frames is not None:
        if total_frames <= num_frames:
            return list(range(total_frames))
        return np.linspace(0, total_frames - 1, num_frames).astype(int).tolist()

    # campionamento per fps target
    if fps <= 0:
        fps = 25.0  # fallback ragionevole se il video non riporta fps validi
    step = max(1, round(fps / sample_fps))
    return list(range(0, total_frames, step))


def read_frames(video_path, indices):
    """Legge dal video solo i frame agli indici richiesti (in ordine crescente)."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    idx_set = sorted(set(indices))
    idx_iter = iter(idx_set)
    target = next(idx_iter, None)
    current = 0

    while target is not None:
        ret, frame = cap.read()
        if not ret:
            break
        if current == target:
            # BGR -> RGB per BLIP/PIL
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            target = next(idx_iter, None)
        current += 1

    cap.release()
    return frames


@torch.no_grad()
def embed_frames(frames, processor, model, device, batch_size):
    """Calcola gli embedding BLIP per una lista di frame RGB (numpy array)."""
    all_embeds = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]
        pixel_values = processor(
            images=batch, return_tensors="pt"
        ).pixel_values.to(device)
        # Il checkpoint image-captioning contiene un vision encoder BLIP completo.
        # Usiamo direttamente il pooled output [B, 768], senza la visual_projection
        # contrastiva di BlipModel (che in questo checkpoint non e' addestrata).
        image_features = model(pixel_values=pixel_values).pooler_output
        image_features = torch.nn.functional.normalize(image_features, dim=-1)
        all_embeds.append(image_features.cpu().numpy())
    if not all_embeds:
        return np.zeros(
            (0, model.config.hidden_size), dtype=np.float32
        )
    return np.concatenate(all_embeds, axis=0)


def process_video(video_path, args, processor, model, device):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    indices = sample_frame_indices(
        total_frames, fps,
        sample_fps=args.sample_fps,
        num_frames=args.num_frames,
    )
    if not indices:
        print(f"[WARN] Nessun frame leggibile per {video_path}, salto.", file=sys.stderr)
        return None

    frames = read_frames(video_path, indices)
    if not frames:
        print(f"[WARN] Lettura frame fallita per {video_path}, salto.", file=sys.stderr)
        return None

    per_frame_embeds = embed_frames(frames, processor, model, device, args.batch_size)
    video_embed = per_frame_embeds.mean(axis=0)
    video_embed = video_embed / (np.linalg.norm(video_embed) + 1e-8)

    return per_frame_embeds, video_embed, indices


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available()
                              else "mps" if torch.backends.mps.is_available()
                              else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Carico modello BLIP: {args.model_name}")

    # use_fast=False rende esplicito e riproducibile il preprocessing salvato con
    # questo checkpoint, eliminando il warning sul cambio di default futuro.
    processor = BlipProcessor.from_pretrained(args.model_name, use_fast=False)
    captioning_model = BlipForConditionalGeneration.from_pretrained(args.model_name)
    model = captioning_model.vision_model.to(device)
    del captioning_model  # il decoder testuale non serve per feature visuali
    model.eval()

    videos = find_videos(args.video_dir)
    if not videos:
        print(f"[ERRORE] Nessun video trovato in {args.video_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] Trovati {len(videos)} video.")

    for video_path in tqdm(videos, desc="Estrazione embedding"):
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        out_path = os.path.join(args.output_dir, f"{video_id}.npz")

        if os.path.exists(out_path) and not args.overwrite:
            # I file prodotti dalla vecchia versione avevano solo
            # "frame_embeddings" ed erano basati su una proiezione casuale.
            # Vengono rigenerati automaticamente; quelli nuovi si possono saltare.
            try:
                with np.load(out_path) as existing:
                    compatible = (
                        "features" in existing
                        and existing["features"].ndim == 2
                        and str(existing.get("encoder", "")) == args.model_name
                    )
                if compatible:
                    continue
                print(f"[INFO] Rigenero file legacy/incompatibile: {out_path}")
            except (OSError, ValueError):
                print(f"[INFO] Rigenero file non leggibile: {out_path}")

        result = process_video(video_path, args, processor, model, device)
        if result is None:
            continue

        per_frame_embeds, video_embed, indices = result

        save_kwargs = {
            # Nome consumato direttamente da train_grpo.py.
            "features": per_frame_embeds.astype(np.float32),
            "video_embedding": video_embed.astype(np.float32),
            "frame_indices": np.array(indices, dtype=np.int64),
            "encoder": np.array(args.model_name),
            "sample_fps": np.array(args.sample_fps, dtype=np.float32),
        }

        np.savez_compressed(out_path, **save_kwargs)

    print(f"[INFO] Fatto. Embedding salvati in: {args.output_dir}")


if __name__ == "__main__":
    main()

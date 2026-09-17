#!/usr/bin/env python3
"""
extract_blip_embeddings.py

Estrae embedding visivi con BLIP dai video del dataset VideoXum.

Per ogni video, guidato da --metadata_json (stesso file con "sampled_frames"
letto da DPO/SFT/train_grpo.py):
  1. Decodifica 'sampled_frames' frame, un frame al secondo via
     timestamp reali (av), identico a decode_frames() in train_grpo.py: questo
     garantisce che l'output abbia sempre shape [count, D], senza i mismatch
     occasionali del vecchio campionamento OpenCV (fps/total_frames del
     container, non affidabile su framerate variabile). Se il file termina
     leggermente prima dei metadata, ripete l'ultimo frame disponibile.
  2. Calcola l'embedding BLIP di ogni frame (vision_model.pooler_output, non
     la visual_projection contrastiva - non addestrata in questo checkpoint).
  3. Salva "features" [count, D] e "video_embedding" [D] in un .npz.

Uso:
    python extract_blip_embeddings.py \
        --metadata_json /path/to/train_videoxum.json \
        --video_dir /path/to/videoxum/train \
        --output_dir /path/to/output/embeddings

Requisiti:
    pip install torch transformers av numpy tqdm
"""

import argparse
import json
import os
import sys
import tempfile
import numpy as np
import av
import torch
from tqdm import tqdm
from transformers import BlipForConditionalGeneration, BlipProcessor


VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov", ".webm")


def parse_args():
    p = argparse.ArgumentParser(description="Estrae embedding BLIP dai video di VideoXum")
    p.add_argument("--metadata_json", type=str, required=True,
                    help="JSON con video_id -> sampled_frames (stesso file letto da "
                         "DPO/SFT/train_grpo.py). Il 'count' di ogni video guida "
                         "l'estrazione: garantisce che 'features' abbia sempre "
                         "shape [count, D], eliminando i mismatch occasionali con "
                         "prepare_video() causati dal vecchio campionamento OpenCV "
                         "indipendente (fps/total_frames del container).")
    p.add_argument("--video_dir", type=str, required=True,
                    help="Cartella contenente i video di VideoXum.")
    p.add_argument("--output_dir", type=str, required=True,
                    help="Cartella dove salvare gli embedding (.npz per video).")
    p.add_argument("--model_name", type=str, default="Salesforce/blip-image-captioning-base",
                    help="Nome/percorso del modello BLIP su HuggingFace Hub.")
    p.add_argument("--batch_size", type=int, default=32,
                    help="Numero di frame processati insieme dal modello.")
    p.add_argument("--device", type=str, default=None,
                    help="cuda / cpu / mps. Default: auto-detect.")
    p.add_argument("--overwrite", action="store_true",
                    help="Ricalcola anche se il file di output esiste già.")
    p.add_argument("--shard-index", type=int, default=0,
                    help="Indice zero-based dello shard da elaborare (default: 0).")
    p.add_argument("--shard-count", type=int, default=1,
                    help="Numero totale di shard disgiunti (default: 1).")
    return p.parse_args()


def load_counts(path):
    """Legge {video_id: count} dallo stesso JSON di metadata usato da DPO/SFT/GRPO."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    items = payload.items() if isinstance(payload, dict) else enumerate(payload)
    counts = {}
    for key, record in items:
        if "sampled_frames" not in record:
            raise ValueError(f"Record {key!r}: manca 'sampled_frames'")
        video_id = str(record.get("video_id", key))
        counts[video_id] = int(record["sampled_frames"])
    if not counts:
        raise ValueError(f"Nessun record in {path}")
    return counts


def resolve_video(video_dir, video_id):
    direct = os.path.join(video_dir, video_id)
    if os.path.isfile(direct):
        return direct
    for extension in VIDEO_EXTENSIONS:
        candidate = os.path.join(video_dir, f"{video_id}{extension}")
        if os.path.isfile(candidate):
            return candidate
    return None


def decode_frames(video_path, expected):
    """IDENTICA alla decode_frames() di train_grpo.py: av, timestamp-based,
    1 frame/secondo, si ferma esattamente a 'expected' frame. Usare qualunque
    altra logica (es. la vecchia basata su fps/total_frames di OpenCV) puo'
    produrre un conteggio diverso da 'expected' su video con framerate
    variabile o metadata container imprecisi, causando il mismatch di shape
    in prepare_video()."""
    frames, first, next_second = [], None, 0
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            ts = float(frame.pts * frame.time_base)
            if first is None:
                first = ts
            if ts - first + 1e-6 >= next_second:
                # BLIP/PIL vogliono RGB; frame.to_image() di PyAV lo restituisce gia' cosi'.
                frames.append(frame.to_image())
                next_second += 1
                if len(frames) == expected:
                    break
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


def process_video(video_path, expected, processor, model, device, batch_size):
    frames = decode_frames(video_path, expected)
    n_decoded = len(frames)
    if n_decoded == 0:
        # Non esiste un ultimo frame sensato da usare come padding: il file e'
        # vuoto, corrotto o non decodificabile e richiede un intervento reale.
        return None, n_decoded
    if n_decoded < expected:
        # I metadata VideoXum definiscono la timeline supervisionata. Quando il
        # container termina prima, estendiamo l'ultima immagine: in questo modo
        # non perdiamo il campione e features/label conservano gli stessi indici.
        frames.extend([frames[-1]] * (expected - n_decoded))

    per_frame_embeds = embed_frames(frames, processor, model, device, batch_size)
    video_embed = per_frame_embeds.mean(axis=0)
    video_embed = video_embed / (np.linalg.norm(video_embed) + 1e-8)
    return (per_frame_embeds, video_embed), n_decoded


def main():
    args = parse_args()
    if args.shard_count < 1:
        raise ValueError("--shard-count deve essere almeno 1")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError(
            "--shard-index deve essere compreso tra 0 e shard-count - 1"
        )
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

    counts = load_counts(args.metadata_json)
    all_items = list(counts.items())
    shard_items = all_items[args.shard_index::args.shard_count]
    print(
        f"[INFO] {len(all_items)} video nei metadata; shard "
        f"{args.shard_index + 1}/{args.shard_count}: {len(shard_items)} video."
    )

    n_ok, n_skip_compatible, n_padded, n_undecodable, n_missing = 0, 0, 0, 0, 0
    for video_id, expected in tqdm(
        shard_items, desc=f"Estrazione embedding shard {args.shard_index + 1}/{args.shard_count}"
    ):
        out_path = os.path.join(args.output_dir, f"{video_id}.npz")

        if os.path.exists(out_path) and not args.overwrite:
            # I file prodotti dalla vecchia versione (campionamento OpenCV
            # indipendente e/o BlipModel con visual_projection random) vanno
            # rigenerati: controlliamo sia il modello sia lo shape[0]==count.
            try:
                with np.load(out_path) as existing:
                    compatible = (
                        "features" in existing
                        and existing["features"].ndim == 2
                        and existing["features"].shape[0] == expected
                        and str(existing.get("encoder", "")) == args.model_name
                    )
                if compatible:
                    n_skip_compatible += 1
                    continue
                print(f"[INFO] Rigenero file legacy/incompatibile: {out_path}")
            except (OSError, ValueError):
                print(f"[INFO] Rigenero file non leggibile: {out_path}")

        video_path = resolve_video(args.video_dir, video_id)
        if video_path is None:
            print(f"[WARN] Video mancante per {video_id}, salto.", file=sys.stderr)
            n_missing += 1
            continue

        result, n_decoded = process_video(
            video_path, expected, processor, model, device, args.batch_size
        )
        if result is None:
            print(
                f"[WARN] {video_id}: nessun frame decodificabile; impossibile "
                "creare gli embedding. Salto.",
                file=sys.stderr,
            )
            n_undecodable += 1
            if os.path.exists(out_path):
                # Non lasciare in giro la cache legacy con shape sbagliata: senza
                # questo, un video genuinamente troncato continuerebbe a far
                # scattare lo stesso ValueError in prepare_video() ad ogni
                # riesecuzione dello script, anche dopo il fix del campionamento.
                os.remove(out_path)
                print(f"[INFO] Rimosso file stantio: {out_path}", file=sys.stderr)
            continue

        if n_decoded < expected:
            print(
                f"[WARN] {video_id}: estratti {n_decoded} frame, attesi "
                f"{expected}; aggiungo {expected - n_decoded} copie "
                "dell'ultimo frame.",
                file=sys.stderr,
            )
            n_padded += 1

        per_frame_embeds, video_embed = result
        save_kwargs = {
            # Nome consumato direttamente da train_grpo.py.
            "features": per_frame_embeds.astype(np.float32),
            "video_embedding": video_embed.astype(np.float32),
            "encoder": np.array(args.model_name),
            "count": np.array(expected, dtype=np.int64),
        }
        # Scrittura atomica: un timeout/cancel non lascia un .npz parziale.
        with tempfile.NamedTemporaryFile(
            dir=args.output_dir, prefix=f".{video_id}.", suffix=".npz", delete=False
        ) as temporary:
            temporary_path = temporary.name
        try:
            np.savez_compressed(temporary_path, **save_kwargs)
            os.replace(temporary_path, out_path)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)
        n_ok += 1

    print(
        f"[DONE] generati={n_ok} gia_compatibili={n_skip_compatible} "
        f"video_padded={n_padded} non_decodificabili={n_undecodable} "
        f"video_mancanti={n_missing} "
        f"totale_shard={len(shard_items)}"
    )
    if n_undecodable:
        print(
            "[NOTE] I video senza alcun frame decodificabile richiedono il "
            "ripristino o la sostituzione del file sorgente.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()

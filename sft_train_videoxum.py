#!/usr/bin/env python3
"""SFT Qwen2.5-VL su video VideoXum, usando come target le risposte 'chosen'
dello stesso dataset di preferenze usato per la DPO."""

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import av
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import AutoProcessor
from transformers import get_cosine_schedule_with_warmup
from evaluation.evalVtV import compute_kendall, compute_spearman
from vl_model_loader import load_qwen_vl_model

PROJECT_ROOT = Path(__file__).resolve().parent
VIDEOXUM_ROOT = PROJECT_ROOT / "dataset" / "aintini" / "datasets" / "videoxum"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "Qwen2.5-VL-7B-Instruct"
# Stesso file JSON usato dalla DPO: contiene gia' vsum_onehot_chosen/rejected.
# La SFT usa solo il campo "chosen".
DEFAULT_DATA = (
    VIDEOXUM_ROOT / "metadata" / "derived" / "train_videoxum_dpo.json"
)
DEFAULT_VIDEO_DIR = VIDEOXUM_ROOT / "train"
DEFAULT_VAL_DATA = VIDEOXUM_ROOT / "metadata" / "val_videoxum.json"
DEFAULT_VAL_VIDEO_DIR = VIDEOXUM_ROOT / "val"
DEFAULT_SAVEROOT = PROJECT_ROOT / "outputs" / "sft" / "checkpoints"
DEFAULT_RESULTSROOT = PROJECT_ROOT / "outputs" / "sft" / "results"
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v")
TRAINING_STATE_NAME = "training_state.pt"
VIDEO_ENCODING_NAME = "video_encoding.json"
# Exit code con cui il training segnala di essersi fermato per --time_budget_hours
# dopo aver salvato un checkpoint riprendibile: gli script SLURM lo usano per
# accodare il job successivo della catena.
TIME_BUDGET_EXIT_CODE = 85
RESUME_POINTER_NAME = "resume_pointer.txt"
SYSTEM_PROMPT = (
    "You are a video summarization assistant. Select the frames that best "
    "summarize the important events in the video."
)
# Unita' della timeline che il modello vede e predice.
# frame: un indice per frame a 1 fps (checkpoint esistenti).
# pair: un indice per coppia di frame consecutivi. Qwen3-VL fonde comunque i
# frame a coppie (temporal_patch_size=2, fisso nei pesi della patch embedding)
# e, con i frame passati come lista, i marker <g.2 seconds> coincidono con
# l'indice di coppia g. Le label e le metriche restano sugli N frame originali:
# la predizione a coppie si riespande con segments_to_frames.
TIMELINE_UNITS = ("frame", "pair")
PROMPT_VERSIONS = {"frame": 1, "pair": 2}
DEFAULT_MAX_PIXELS = 256 * 28 * 28
DEFAULT_MIN_PIXELS = 4 * 28 * 28


def timeline_length(frame_count, unit="frame"):
    """Numero di indici della timeline per un video di frame_count frame."""
    if unit == "pair":
        return (frame_count + 1) // 2
    return frame_count


def segments_to_frames(prediction, frame_count, unit="frame"):
    """Riporta una selezione sulla timeline del modello agli N frame originali.

    Con unit='pair' il segmento g copre i frame 2g e 2g+1; l'ultimo segmento di
    un video con N dispari copre solo il frame N-1 (Qwen duplica quel frame).
    """
    if unit != "pair":
        return list(prediction)
    expected = timeline_length(frame_count, unit)
    if len(prediction) != expected:
        raise ValueError(
            f"predizione a coppie lunga {len(prediction)}, attesi {expected} "
            f"segmenti per {frame_count} frame"
        )
    return [prediction[index // 2] for index in range(frame_count)]


def build_timeline_intro(count, index_phrase=", numbered from", unit="frame"):
    """``index_phrase`` esiste solo per riprodurre la clausola con cui sono stati
    addestrati i checkpoint esistenti: SFT usa ", numbered from", DPO " indexed
    from". E' l'unica differenza testuale tra i due prompt, quindi resta
    parametrica per non disallineare i checkpoint gia' addestrati. Con
    unit='pair' il testo e' unico per tutte le fasi e index_phrase e' ignorato."""
    if unit == "pair":
        return (
            f"The timeline contains {count} candidate segments, numbered from 0 "
            f"to {count - 1}. Each segment covers two consecutive seconds of the "
            "video. Select the segments that form the best extractive video "
            "summary.\n\n"
        )
    return (
        f"The timeline contains {count} candidate frames{index_phrase} 0 to "
        f"{count - 1}, sampled at one frame per second. Select the frames that "
        "form the best extractive video summary.\n\n"
    )


def build_selection_requirements(unit="frame"):
    item = "segments" if unit == "pair" else "frames"
    return (
        "Selection requirements:\n"
        f"- Select only {item} containing important, representative events.\n"
        "- Aim to select approximately 20% of the timeline; normally keep the "
        f"selection between 15% and 25% of all {item}.\n"
        f"- Prefer a small number of coherent temporal segments over many isolated {item}.\n"
        "- Use at most 12 intervals.\n\n"
    )


def build_interval_rules(count, unit="frame"):
    """Semantica degli intervalli, identica in tutte le fasi. GRPO la punisce
    esplicitamente nella reward (merged_adjacent_count), quindi deve enunciarla
    con la stessa precisione con cui la enunciano SFT e DPO."""
    item = "segment" if unit == "pair" else "frame"
    return (
        f"- Every index must be an integer between 0 and {count - 1}, inclusive.\n"
        "- Intervals must be sorted chronologically.\n"
        "- Intervals must not overlap.\n"
        "- Merge overlapping or consecutive intervals into one maximal interval.\n"
        "- Therefore, for consecutive output intervals [a,b] and [c,d], c must "
        "be greater than b + 1.\n"
        f"- Represent an isolated selected {item} i as [i,i].\n"
    )


def build_user_prompt(count, index_phrase=", numbered from", unit="frame"):
    """Prompt condiviso con DPO/GRPO: unica fonte di verita' per le regole di
    selezione. ``train_dpo.py`` la richiama direttamente, ``train_grpo.py``
    riusa i blocchi condivisi (intro, selection requirements, interval rules) e
    sostituisce solo la sezione di output, perche' la sua reward richiede anche
    un summary testuale. Se cambia una regola di selezione, deve cambiare qui:
    altrimenti si ripropone il disallineamento di prompt tra SFT/DPO/GRPO che
    degrada la qualita' della selezione dei frame salienti a valle.
    ``count`` e' la lunghezza della timeline nell'unita' ``unit``."""
    item = "segment" if unit == "pair" else "frame"
    return (
        build_timeline_intro(count, index_phrase, unit)
        + build_selection_requirements(unit)

        + "Output requirements:\n"
        "- Return exactly one valid JSON array of inclusive integer intervals:\n"
        f"  [[start_{item},end_{item}],...]\n"
        + build_interval_rules(count, unit)
        + f"- Return [] only if no {item} can be selected.\n"
        "- Do not return timestamps, seconds, objects, strings, decimal numbers, "
        "Markdown, code fences, comments, or explanatory text.\n\n"

        "Valid example: [[0,2],[5,6]]\n"
        "Invalid example: [[0,2],[2,4]] because the intervals overlap.\n"
        "Invalid example: [[0,2],[3,4]] because consecutive intervals must be "
        "merged into [[0,4]].\n\n"

        "Return only the JSON array."
    )


def format_response(onehot):
    """Converte una label one-hot in intervalli JSON inclusivi [inizio, fine]."""
    if not isinstance(onehot, list) or not onehot:
        raise ValueError("label chosen vuota o non valida")
    if any(value not in (0, 1, False, True) for value in onehot):
        raise ValueError("la label chosen deve contenere solo 0 e 1")

    intervals = []
    start = None
    for index, value in enumerate(onehot):
        if value and start is None:
            start = index
        elif not value and start is not None:
            intervals.append([start, index - 1])
            start = None
    if start is not None:
        intervals.append([start, len(onehot) - 1])

    return json.dumps(intervals, separators=(",", ":"))


def resolve_video(video_dir, video_id):
    direct = video_dir / video_id
    if direct.is_file():
        return direct
    for extension in VIDEO_EXTENSIONS:
        candidate = video_dir / f"{video_id}{extension}"
        if candidate.is_file():
            return candidate
    return video_dir / f"{video_id}.mp4"


def load_records(path, require_chosen=True, timeline_unit="frame"):
    """Come nella DPO, ma qui serve solo 'vsum_onehot_chosen', non 'rejected'.

    vsum_onehot resta sempre sugli N frame (serve alle metriche); la label
    chosen e' invece sulla timeline del modello, lunga timeline_length(N)."""
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    items = payload.items() if isinstance(payload, dict) else enumerate(payload)
    records = []
    for key, record in items:
        required = {"sampled_frames", "vsum_onehot"}
        if require_chosen:
            required |= {"vsum_onehot_chosen"}
        if missing := required - set(record):
            raise ValueError(f"Record {key!r}: campi mancanti {sorted(missing)}")
        references = record["vsum_onehot"]
        frame_count = int(record["sampled_frames"])
        if len(references) != 10 or any(
            len(reference) != frame_count for reference in references
        ):
            raise ValueError(
                f"Record {key!r}: vsum_onehot deve contenere 10 annotazioni "
                f"da {frame_count} frame"
            )
        item = {
            "video_id": str(record.get("video_id", key)),
            "sampled_frames": frame_count,
            "references": references,
        }
        if require_chosen:
            chosen = record["vsum_onehot_chosen"]
            expected = timeline_length(frame_count, timeline_unit)
            if len(chosen) != expected:
                raise ValueError(
                    f"Record {key!r}: sampled_frames={frame_count}, "
                    f"timeline_unit={timeline_unit} richiede len(chosen)={expected}, "
                    f"trovato {len(chosen)}"
                )
            item["target"] = format_response(chosen)
        records.append(item)
    if not records:
        raise ValueError(f"Nessun record in {path}")
    return records


def split_records(records, val_ratio, seed):
    if not 0 < val_ratio < 1:
        raise ValueError("--val_ratio deve essere compreso tra 0 e 1")
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    val_indices = set(indices[:max(1, round(len(records) * val_ratio))])
    return (
        [item for i, item in enumerate(records) if i not in val_indices],
        [item for i, item in enumerate(records) if i in val_indices],
    )


class VideoXumSFTDataset(Dataset):
    def __init__(self, records, video_dir, timeline_unit="frame"):
        self.records = records
        self.timeline_unit = timeline_unit
        self.paths = [resolve_video(video_dir, item["video_id"]) for item in records]
        missing = [str(path) for path in self.paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Mancano {len(missing)} video in {video_dir}. Primi file:\n  "
                + "\n  ".join(missing[:10])
            )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        count = record["sampled_frames"]
        timeline_count = timeline_length(count, self.timeline_unit)
        sample = {
            "video_path": str(self.paths[index]),
            "sampled_frames": count,
            "timeline_unit": self.timeline_unit,
            "timeline_count": timeline_count,
            "prompt": build_user_prompt(timeline_count, unit=self.timeline_unit),
            "references": record["references"],
        }
        if "target" in record:
            sample["target"] = record["target"]
        return sample


def decode_video_frames(sample, args):
    """Estrae un frame al secondo, coerentemente con le label VideoXum."""
    expected_frames = sample["sampled_frames"]
    if args.max_frames and args.max_frames < expected_frames:
        raise ValueError(
            f"Il video richiede {expected_frames} frame a 1 FPS, ma "
            f"--max_frames={args.max_frames}. Usa 0 (nessun limite) oppure un "
            "valore almeno pari a sampled_frames."
        )

    with av.open(sample["video_path"]) as container:
        stream = container.streams.video[0]
        frames = []
        first_timestamp = None
        next_second = 0

        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            timestamp = float(frame.pts * frame.time_base)
            if first_timestamp is None:
                first_timestamp = timestamp
            elapsed = timestamp - first_timestamp

            # Un solo frame per secondo: 0 s, 1 s, 2 s, ...
            if elapsed + 1e-6 >= next_second:
                frames.append(frame.to_image())
                next_second += 1
                if len(frames) == expected_frames:
                    break

    return pad_decoded_frames(frames, sample)


def pad_decoded_frames(frames, sample):
    """Allinea i frame decodificati alle label, come train_grpo.decode_frames.

    Alcuni video terminano prima di sampled_frames secondi decodificabili: senza
    padding il modello vedrebbe meno posizioni temporali delle label e degli
    indici chiesti nel prompt. Si ripete l'ultimo frame in coda."""
    expected_frames = sample["sampled_frames"]
    decoded = len(frames)
    if decoded == 0:
        raise ValueError(
            f"nessun frame decodificabile in {sample['video_path']} "
            f"(attesi={expected_frames})"
        )
    if decoded < expected_frames:
        frames = frames + [frames[-1]] * (expected_frames - decoded)
        print(
            f"[WARN] {sample['video_path']}: frame estratti={decoded}, "
            f"attesi={expected_frames}; padding finale={expected_frames - decoded}"
        )
    return frames


def apply_video_time_metadata(video, args):
    """Imposta i metadata temporali dei frame gia' decodificati.

    Nel ramo "lista di frame" qwen_vl_utils (vision_process.py) legge solo
    `sample_fps`/`raw_fps`, ignorando `fps` e `nframes`, e senza di essi assume
    2 fps. Qwen3VLProcessor inserisce allora nel prompt marker <t seconds> a
    meta' del tempo reale (<0.2>, <1.2>, ... per frame a 1 fps), incoerenti con
    gli indici 0..n-1 delle label. 'seconds' li riallinea ai secondi reali
    (<0.5>, <2.5>, ...); 'legacy' riproduce il comportamento precedente, con cui
    sono stati addestrati i checkpoint senza video_encoding.json.
    """
    if getattr(args, "video_time_mode", "legacy") == "seconds":
        # GRPO non espone --fps: la timeline VideoXum e' sempre a 1 fps.
        fps = getattr(args, "fps", 1.0)
        video["sample_fps"] = fps
        video["raw_fps"] = fps
    return video


def read_video_encoding(checkpoint):
    """Contenuto di video_encoding.json, o None per i checkpoint che lo
    precedono (addestrati con video_time_mode legacy e timeline a frame)."""
    path = Path(checkpoint) / VIDEO_ENCODING_NAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_video_time_mode(checkpoint):
    """Codifica temporale con cui e' stato addestrato un checkpoint, o None se
    il checkpoint precede video_encoding.json (quindi e' 'legacy')."""
    encoding = read_video_encoding(checkpoint)
    return None if encoding is None else encoding["video_time_mode"]


def write_video_encoding(checkpoint, args):
    """Salva nel checkpoint tutto cio' che cambia l'input visto dal modello, cosi'
    fasi successive e inferenza non possono usare una configurazione diversa."""
    timeline_unit = getattr(args, "timeline_unit", "frame")
    (Path(checkpoint) / VIDEO_ENCODING_NAME).write_text(
        json.dumps(
            {
                "video_time_mode": args.video_time_mode,
                "sample_fps": getattr(args, "fps", 1.0),
                "timeline_unit": timeline_unit,
                "prompt_version": PROMPT_VERSIONS[timeline_unit],
                "max_pixels": getattr(args, "max_pixels", None),
                "min_pixels": getattr(args, "min_pixels", None),
                # v2v: solo intervalli (questo script); v2vt: intervalli e summary
                # (sft_train_videoxum_v2vt_trl.py). summary_style e' lo stile del
                # prompt GRPO con cui il checkpoint e' stato addestrato: l'inferenza
                # (infer_grpo_videoxum.py --summary_style) deve usare lo stesso.
                "task": getattr(args, "task", "v2v"),
                "summary_style": getattr(args, "summary_style", None),
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


def resolve_video_time_mode(args, source=None, default="seconds", strict=True):
    """Risolve la codifica video rispetto al checkpoint di partenza.

    `source` e' il checkpoint da cui il job eredita i pesi (resume, init o
    checkpoint valutato).

    - timeline_unit 'auto' adotta quella del checkpoint ('frame' se non ha
      sidecar o se non c'e' checkpoint). Un'unita' diversa da quella del
      checkpoint e' sempre un errore: gli indici non avrebbero lo stesso
      significato.
    - video_time_mode 'auto' adotta quella del checkpoint, 'legacy' se non ha
      sidecar, e `default` se non c'e' un checkpoint di partenza. Con
      timeline_unit='pair' e' sempre 'legacy', l'unica in cui i marker <g.2>
      coincidono con l'indice di coppia. Con strict=True una codifica esplicita
      diversa da quella del checkpoint e' un errore.
    - max_pixels/min_pixels non passati (None) adottano quelli salvati nel
      checkpoint, altrimenti i default; un valore esplicito diverso da quello
      salvato e' sempre un errore.
    """
    encoding = read_video_encoding(source) if source is not None else None
    stored = encoding or {}

    inherited_unit = None if source is None else stored.get("timeline_unit", "frame")
    requested_unit = getattr(args, "timeline_unit", "frame")
    if requested_unit == "auto":
        args.timeline_unit = inherited_unit or "frame"
    else:
        args.timeline_unit = requested_unit
        if inherited_unit is not None and inherited_unit != requested_unit:
            raise SystemExit(
                f"[video] {source} usa timeline_unit={inherited_unit}, ma questo "
                f"job usa {requested_unit}. Usa --timeline_unit auto."
            )
    if args.timeline_unit == "pair":
        default = "legacy"

    inherited = None
    if source is not None:
        inherited = stored.get("video_time_mode", "legacy")
    if args.video_time_mode == "auto":
        args.video_time_mode = inherited or default
    elif inherited is not None and inherited != args.video_time_mode:
        message = (
            f"{source} e' stato addestrato con video_time_mode={inherited}, "
            f"ma questo job usa {args.video_time_mode}"
        )
        if strict:
            raise SystemExit(f"[video] {message}. Usa --video_time_mode auto.")
        print(f"[WARN] {message}: valutazione con codifica diversa dal training.")
    if args.timeline_unit == "pair" and args.video_time_mode != "legacy":
        raise SystemExit(
            "[video] timeline_unit=pair richiede video_time_mode=legacy: con "
            f"{args.video_time_mode} i marker temporali non coincidono con gli "
            "indici di coppia."
        )

    for name, fallback in (
        ("max_pixels", DEFAULT_MAX_PIXELS), ("min_pixels", DEFAULT_MIN_PIXELS),
    ):
        if not hasattr(args, name):
            continue
        saved = stored.get(name)
        if getattr(args, name) is None:
            setattr(args, name, saved if saved is not None else fallback)
        elif saved is not None and saved != getattr(args, name):
            raise SystemExit(
                f"[video] {source} e' stato addestrato con {name}={saved}, ma "
                f"questo job usa {getattr(args, name)}. Ometti --{name} per "
                "ereditarlo."
            )
    if (
        getattr(args, "min_pixels", None) is not None
        and args.min_pixels > args.max_pixels
    ):
        raise SystemExit(
            f"--min_pixels ({args.min_pixels}) non puo' superare "
            f"--max_pixels ({args.max_pixels})"
        )

    print(
        f"[video] video_time_mode={args.video_time_mode} "
        f"timeline_unit={args.timeline_unit} "
        f"max_pixels={getattr(args, 'max_pixels', None)} "
        f"min_pixels={getattr(args, 'min_pixels', None)} "
        f"(checkpoint di partenza={source or 'nessuno'}, codifica ereditata={encoding})"
    )
    return args.video_time_mode


def guard_checkpoint_dir(save_dir, args):
    """Rifiuta un nuovo training in una directory con checkpoint di un'altra
    codifica: sovrascriverebbe checkpoint-best/final di un run diverso (es. il
    baseline legacy) con un modello non confrontabile."""
    for checkpoint in sorted(Path(save_dir).glob("checkpoint-*")):
        if not (checkpoint / "adapter_config.json").is_file():
            continue
        encoding = read_video_encoding(checkpoint) or {}
        existing = encoding.get("video_time_mode", "legacy")
        existing_unit = encoding.get("timeline_unit", "frame")
        if (
            existing != args.video_time_mode
            or existing_unit != getattr(args, "timeline_unit", "frame")
        ):
            raise SystemExit(
                f"[video] {save_dir} contiene {checkpoint.name} con "
                f"video_time_mode={existing}, timeline_unit={existing_unit}; "
                f"questo run usa {args.video_time_mode}, "
                f"{getattr(args, 'timeline_unit', 'frame')}. Usare un --saveroot dedicato."
            )


def build_messages(sample, args, response=None, video_frames=None):
    video = {
        "type": "video",
        "video": video_frames if video_frames is not None else sample["video_path"],
        "max_pixels": args.max_pixels,
        # qwen_vl_utils usa di default VIDEO_MIN_PIXELS=128*28*28=100352 per i
        # video. Se --max_pixels scende sotto quella soglia (es. per risparmiare
        # VRAM), l'assert interno "max_pixels >= min_pixels" fallisce a meno di
        # abbassare esplicitamente anche min_pixels qui.
        "min_pixels": args.min_pixels,
    }
    if video_frames is not None:
        apply_video_time_metadata(video, args)
    elif args.max_frames and sample["sampled_frames"] > args.max_frames:
        nframes = max(2, min(args.max_frames, sample["sampled_frames"]))
        video["nframes"] = nframes - nframes % 2
    else:
        video["fps"] = args.fps
    messages = [
        # Il sample puo' portare il proprio system prompt (SFT V2VT: quello di
        # train_grpo.SYSTEM_PROMPTS), cosi' prepare_completion resta unica.
        {"role": "system", "content": [
            {"type": "text", "text": sample.get("system_prompt", SYSTEM_PROMPT)}
        ]},
        {"role": "user", "content": [
            video, {"type": "text", "text": sample["prompt"]}
        ]},
    ]
    if response is not None:
        messages.append({
            "role": "assistant", "content": [{"type": "text", "text": response}]
        })
    return messages

def process_video_inputs(messages, model_type):
    if model_type == "qwen3_vl":
        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        if videos is not None:
            videos, video_metadata = zip(*videos)
            videos = list(videos)
            video_kwargs["video_metadata"] = list(video_metadata)

        return images, videos, video_kwargs

    images, videos = process_vision_info(
        messages,
        image_patch_size=14,
    )
    return images, videos, {}

def prepare_completion(
    processor, sample, response, args, model_type, video_frames=None
):
    """Decodifica il video una volta e trova l'inizio esatto della completion."""
    prompt_messages = build_messages(sample, args, video_frames=video_frames)
    full_messages = build_messages(sample, args, response, video_frames=video_frames)
    prompt_text = processor.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    full_text = processor.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )
    images, videos, video_kwargs = process_video_inputs(
        prompt_messages, model_type
    )

    common = dict(
    images=images,
    videos=videos,
    padding=False,
    return_tensors="pt",
    cap_pixels_per_frame=True,
    **video_kwargs,
    )
    prompt_inputs = processor(text=[prompt_text], **common)
    full_inputs = processor(text=[full_text], **common)
    prompt_ids = prompt_inputs["input_ids"][0]
    full_ids = full_inputs["input_ids"][0]
    prompt_len = prompt_ids.numel()
    if full_ids.numel() <= prompt_len or not torch.equal(full_ids[:prompt_len], prompt_ids):
        raise RuntimeError(
            "Template prompt/completion incompatibile con questa versione di transformers"
        )
    return full_inputs, prompt_len


def move_inputs(inputs, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }


def prepare_sft_example(
    processor, sample, args, device, model_type, video_frames=None
):
    """Prepara input_ids + labels (prompt mascherato a -100) per la CE loss."""
    frames = video_frames if video_frames is not None else decode_video_frames(sample, args)
    full_inputs, prompt_len = prepare_completion(
        processor, sample, sample["target"], args, model_type, frames
    )
    labels = full_inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100
    full_inputs["labels"] = labels
    completion_tokens = full_inputs["input_ids"].shape[-1] - prompt_len
    return move_inputs(full_inputs, device), completion_tokens


def write_metric(handle, **values):
    """Scrive una metrica come record JSONL immediatamente persistente."""
    handle.write(json.dumps(values, separators=(",", ":"), default=str) + "\n")


def parse_interval_response(response, frame_count):
    """Estrae intervalli JSON interi e crea una selezione frame-level."""
    array_start = re.search(r"\[", response)
    if array_start is None:
        raise ValueError("nessun array JSON trovato nella risposta")
    intervals, _ = json.JSONDecoder().raw_decode(response[array_start.start():])
    if not isinstance(intervals, list):
        raise ValueError("la risposta non e' un array JSON")

    prediction = [0] * frame_count
    previous_end = -1
    for interval in intervals:
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError("ogni intervallo deve essere [start, end]")
        start, end = interval
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
        ):
            raise ValueError("gli estremi degli intervalli devono essere interi")
        if not 0 <= start <= end < frame_count:
            raise ValueError(f"intervallo fuori timeline: [{start}, {end}]")
        if start <= previous_end:
            raise ValueError("gli intervalli devono essere ordinati e non sovrapposti")
        for index in range(start, end + 1):
            prediction[index] = 1
        previous_end = end
    return prediction


def parse_timeline_response(response, sample):
    """Parsa la risposta sulla timeline del modello e la riporta agli N frame."""
    unit = sample.get("timeline_unit", "frame")
    frame_count = sample["sampled_frames"]
    prediction = parse_interval_response(
        response, sample.get("timeline_count", frame_count)
    )
    return segments_to_frames(prediction, frame_count, unit)


def binary_f1(prediction, reference):
    """F1 frame-level equivalente a sklearn per due vettori binari."""
    true_positive = sum(p == 1 and r == 1 for p, r in zip(prediction, reference))
    denominator = sum(prediction) + sum(reference)
    return 2.0 * true_positive / denominator if denominator else 0.0


METRIC_NAMES = ("f1_max", "f1_mean", "kendall", "spearman", "kappa", "lift")


def chance_corrected_metrics(prediction, references):
    """Metriche che non migliorano selezionando piu' frame a caso.

    kappa: Cohen's kappa medio tra predizione e ciascun annotatore (0 = caso,
    anche selezionando tutto il video).
    lift: saliency media dei frame selezionati / saliency media del video, con
    saliency = frazione di annotatori che hanno scelto il frame (1 = caso a
    qualsiasi copertura). Una selezione vuota vale 0.
    """
    frame_count = len(prediction)
    selected = sum(prediction)
    kappas = []
    for reference in references:
        positives = sum(reference)
        true_positive = sum(p and r for p, r in zip(prediction, reference))
        true_negative = frame_count - selected - positives + true_positive
        observed = (true_positive + true_negative) / frame_count
        expected = (
            selected * positives
            + (frame_count - selected) * (frame_count - positives)
        ) / frame_count ** 2
        kappas.append((observed - expected) / (1 - expected) if expected < 1 else 0.0)
    saliency = [sum(column) / len(references) for column in zip(*references)]
    mean_saliency = sum(saliency) / frame_count
    if selected and mean_saliency:
        selected_saliency = sum(s for s, p in zip(saliency, prediction) if p) / selected
        lift = selected_saliency / mean_saliency
    else:
        lift = 0.0
    return {"kappa": sum(kappas) / len(kappas), "lift": lift}


def videoxum_metrics(prediction, references):
    """Protocollo VideoXum adattato a una predizione generativa binaria.

    prediction e references sono sempre sugli N frame originali: una predizione
    a coppie va prima riespansa con segments_to_frames."""
    f1_scores = [binary_f1(prediction, reference) for reference in references]
    prediction_tensor = torch.tensor(prediction, dtype=torch.float32).unsqueeze(0)
    references_tensor = torch.tensor(references, dtype=torch.float32)
    mean_saliency = references_tensor.mean(dim=0, keepdim=True)
    mask = torch.ones_like(prediction_tensor)
    _, _, kendall_mean = compute_kendall(
        prediction_tensor, mean_saliency, mask, num_repeat=1
    )
    _, _, spearman_mean = compute_spearman(
        prediction_tensor, mean_saliency, mask, num_repeat=1
    )
    return {
        "f1_max": max(f1_scores),
        "f1_mean": sum(f1_scores) / len(f1_scores),
        "kendall": kendall_mean.item(),
        "spearman": spearman_mean.item(),
        **chance_corrected_metrics(prediction, references),
    }


def selected_runs(prediction):
    """Intervalli inclusivi [start, end] di una selezione binaria."""
    runs, start = [], None
    for index, value in enumerate(prediction):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(prediction) - 1))
    return runs


def decile_profile(vectors, deciles=10):
    """Densita' media selezionata per decile della timeline."""
    totals = [0.0] * deciles
    for vector in vectors:
        length = len(vector)
        for decile in range(deciles):
            low = decile * length // deciles
            high = max((decile + 1) * length // deciles, low + 1)
            totals[decile] += sum(vector[low:high]) / (high - low)
    return [total / len(vectors) for total in totals]


def collapse_metrics(responses, predictions, references_list, seed=0):
    """Misura quanto l'output dipende dal video invece che dal solo numero di frame.

    Uno SFT collassato emette lo stesso template per video diversi (es. sempre
    [[1,3],[10,12],...]). Le metriche strutturali sono calcolate sulle sole
    predizioni non vuote; `cross_video_iou_null` e' la IoU attesa tra due
    selezioni indipendenti con le stesse densita', da usare come riferimento.
    """
    total = len(predictions)
    if total == 0:
        return {}
    compact = [re.sub(r"\s+", "", response) for response in responses]
    nonempty = [prediction for prediction in predictions if any(prediction)]
    metrics = {
        "distinct_output_ratio": len(set(compact)) / total,
        "top4_output_share": sum(c for _, c in Counter(compact).most_common(4)) / total,
        "empty_prediction_rate": 1.0 - len(nonempty) / total,
    }
    if not nonempty:
        return metrics

    runs = [selected_runs(prediction) for prediction in nonempty]
    prefixes = Counter(tuple(run[:2]) for run in runs)
    starts = Counter(run[0][0] for run in runs)
    lengths = Counter(end - start + 1 for run in runs for start, end in run)
    interval_count = sum(lengths.values())

    by_length = defaultdict(list)
    for position, prediction in enumerate(nonempty):
        by_length[len(prediction)].append(position)
    rng = random.Random(seed)
    ious, nulls = [], []
    for positions in by_length.values():
        if len(positions) < 2:
            continue
        for _ in range(min(50, len(positions) * (len(positions) - 1) // 2)):
            first, second = (nonempty[p] for p in rng.sample(positions, 2))
            union = sum(a | b for a, b in zip(first, second))
            ious.append(sum(a & b for a, b in zip(first, second)) / union)
            density_a = sum(first) / len(first)
            density_b = sum(second) / len(second)
            nulls.append(
                density_a * density_b / (density_a + density_b - density_a * density_b)
            )

    saliency = [
        [sum(column) / len(references) for column in zip(*references)]
        for references in references_list
    ]
    predicted_profile = decile_profile(predictions)
    reference_profile = decile_profile(saliency)
    metrics.update({
        "prefix2_share": prefixes.most_common(1)[0][1] / len(nonempty),
        "first_start_mode_share": starts.most_common(1)[0][1] / len(nonempty),
        "first_start_top5": [[start, count / len(nonempty)] for start, count in starts.most_common(5)],
        "mean_interval_length": sum(k * c for k, c in lengths.items()) / interval_count,
        "modal_interval_length_share": lengths.most_common(1)[0][1] / interval_count,
        "mean_density": sum(sum(p) / len(p) for p in nonempty) / len(nonempty),
        "cross_video_iou": sum(ious) / len(ious) if ious else None,
        "cross_video_iou_null": sum(nulls) / len(nulls) if nulls else None,
        "decile_profile_pred": predicted_profile,
        "decile_profile_gt": reference_profile,
        "decile_profile_l1": sum(abs(a - b) for a, b in zip(predicted_profile, reference_profile)),
    })
    return metrics


def format_collapse_summary(metrics):
    keys = (
        "distinct_output_ratio", "prefix2_share", "first_start_mode_share",
        "modal_interval_length_share", "cross_video_iou", "cross_video_iou_null",
    )
    return " ".join(
        f"{key}={metrics[key]:.3f}" for key in keys if metrics.get(key) is not None
    )


def generate_interval_response(
    model, processor, sample, args, device, frames, model_type
):
    """Genera gli intervalli dal solo video e prompt, senza teacher forcing."""
    messages = build_messages(sample, args, video_frames=frames)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    images, videos, video_kwargs = process_video_inputs(messages, model_type)
    inputs = processor(
        text=[text], images=images, videos=videos,
        padding=False, return_tensors="pt",
        cap_pixels_per_frame=True, **video_kwargs
    )
    inputs = move_inputs(inputs, device)
    prompt_length = inputs["input_ids"].shape[-1]
    generated = model.generate(
        **inputs,
        max_new_tokens=args.eval_max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    completion_ids = generated[:, prompt_length:]
    return processor.batch_decode(
        completion_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def validation_limit(dataset, requested):
    return len(dataset) if requested <= 0 else min(requested, len(dataset))


def format_validation_summary(summary_metrics, limit, invalid_predictions):
    return (
        f"[val] VideoXum F1_MAX={summary_metrics['f1_max']:.4f} "
        f"F1_MEAN={summary_metrics['f1_mean']:.4f} "
        f"Kendall={summary_metrics['kendall']:.4f} "
        f"Spearman={summary_metrics['spearman']:.4f} "
        f"Kappa={summary_metrics['kappa']:.4f} "
        f"Lift={summary_metrics['lift']:.4f} "
        f"valid_predictions={limit - invalid_predictions}/{limit}"
    )


def selection_score(metric_values, args):
    """Punteggio per checkpoint-best, sempre sugli stessi primi video di
    validation: la validation periodica e quella di fine epoca usano quindi un
    riferimento comparabile anche se valutano un numero diverso di video."""
    values = metric_values[args.best_metric]
    if args.eval_steps:
        values = values[:args.eval_steps_samples]
    return sum(values) / len(values)


@torch.no_grad()
def evaluate(
    model, processor, dataset, args, device, metrics_file, run_id, epoch,
    model_type, limit=None, optimizer_step=None,
):
    model.eval()
    limit = validation_limit(dataset, args.eval_samples if limit is None else limit)
    invalid_predictions = 0
    metric_values = {name: [] for name in METRIC_NAMES}
    responses, predictions, references_list = [], [], []
    for index in tqdm(range(limit), desc="validation", unit="video", leave=False):
        sample = dataset[index]
        frames = decode_video_frames(sample, args)
        response = generate_interval_response(
            model, processor, sample, args, device, frames, model_type
        )
        parse_error = None
        try:
            prediction = parse_timeline_response(response, sample)
        except (json.JSONDecodeError, ValueError) as error:
            prediction = [0] * sample["sampled_frames"]
            parse_error = str(error)
            invalid_predictions += 1
        sample_metrics = videoxum_metrics(prediction, sample["references"])
        for name, value in sample_metrics.items():
            metric_values[name].append(value)
        responses.append(response)
        predictions.append(prediction)
        references_list.append(sample["references"])

        write_metric(
            metrics_file,
            run_id=run_id,
            record_type="sample",
            split="validation",
            epoch=epoch,
            optimizer_step=optimizer_step,
            sample_position=index + 1,
            dataset_index=index,
                generated_response=response,
                model_type=model_type,
            prediction_valid=parse_error is None,
            parse_error=parse_error,
            selected_frames=sum(prediction),
            selection_ratio=sum(prediction) / len(prediction),
            **sample_metrics,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summary_metrics = {
        name: sum(values) / len(values)
        for name, values in metric_values.items()
    }
    summary_metrics["selection_score"] = selection_score(metric_values, args)
    collapse = collapse_metrics(responses, predictions, references_list)
    print(format_validation_summary(summary_metrics, limit, invalid_predictions))
    print(f"[val] collasso: {format_collapse_summary(collapse)}")
    write_metric(
        metrics_file,
        run_id=run_id,
        record_type="summary",
        split="validation",
        epoch=epoch,
        optimizer_step=optimizer_step,
        samples=limit,
        invalid_predictions=invalid_predictions,
        metric_protocol="VideoXum aggregation; binary generated saliency",
        video_time_mode=args.video_time_mode,
        timeline_unit=args.timeline_unit,
        best_metric=args.best_metric,
        collapse=collapse,
        **summary_metrics,
    )
    return summary_metrics


@torch.no_grad()
def measure_generation_diversity(
    model, processor, dataset, args, device, metrics_file, run_id, epoch,
    model_type,
):
    """Campiona k generazioni per alcuni video di validation con temperatura
    e misura quanto sono diverse tra loro. Serve come segnale precoce: se la
    SFT ha collassato la policy su un output quasi deterministico, GRPO
    partira' con varianza di reward vicina a zero all'interno dei gruppi
    campionati (stesso sintomo del bug format_reward/std=0 gia' diagnosticato,
    ma con causa diversa: qui la policy stessa, non il validator)."""
    model.eval()
    n = min(args.diversity_check_samples, len(dataset))
    unique_ratios = []
    for index in range(n):
        sample = dataset[index]
        frames = decode_video_frames(sample, args)
        messages = build_messages(sample, args, video_frames=frames)
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos, video_kwargs = process_video_inputs(messages, model_type)
        inputs = processor(
            text=[text], images=images, videos=videos,
            padding=False, return_tensors="pt",
            cap_pixels_per_frame=True, **video_kwargs
        )
        inputs = move_inputs(inputs, device)
        prompt_length = inputs["input_ids"].shape[-1]

        responses = []
        for _ in range(args.diversity_check_k):
            generated = model.generate(
                **inputs,
                max_new_tokens=args.eval_max_new_tokens,
                do_sample=True,
                temperature=args.diversity_temperature,
                top_p=0.95,
                use_cache=True,
            )
            completion_ids = generated[:, prompt_length:]
            responses.append(processor.batch_decode(
                completion_ids, skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip())

        predictions = []
        for response in responses:
            try:
                predictions.append(parse_timeline_response(response, sample))
            except (json.JSONDecodeError, ValueError):
                predictions.append(None)
        valid_predictions = [p for p in predictions if p is not None]
        pairwise_f1 = [
            binary_f1(valid_predictions[i], valid_predictions[j])
            for i in range(len(valid_predictions))
            for j in range(i + 1, len(valid_predictions))
        ]
        unique_ratio = len(set(responses)) / len(responses)
        unique_ratios.append(unique_ratio)

        write_metric(
            metrics_file,
            run_id=run_id,
            record_type="diversity",
            split="validation",
            epoch=epoch,
            dataset_index=index,
            k=args.diversity_check_k,
            temperature=args.diversity_temperature,
            unique_ratio=unique_ratio,
            valid_responses=len(valid_predictions),
            mean_pairwise_f1_among_samples=(
                sum(pairwise_f1) / len(pairwise_f1) if pairwise_f1 else None
            ),
            responses=responses,
        )

    if unique_ratios:
        mean_unique_ratio = sum(unique_ratios) / len(unique_ratios)
        write_metric(
            metrics_file,
            run_id=run_id,
            record_type="diversity_summary",
            split="validation",
            epoch=epoch,
            samples=n,
            mean_unique_ratio=mean_unique_ratio,
        )
        print(
            f"[diversity] mean_unique_ratio={mean_unique_ratio:.2f} su {n} video "
            f"(k={args.diversity_check_k}, T={args.diversity_temperature}). "
            "Valori vicini a 0 indicano possibile mode collapse: rischioso per "
            "la varianza di reward richiesta da GRPO."
        )


@torch.no_grad()
def measure_video_dependence(
    model, processor, dataset, args, device, metrics_file, run_id, epoch,
    model_type,
):
    """Verifica se l'output dipende dai pixel: rigenera con il prompt originale
    (quindi stesso n e stessa struttura di token) ma con i frame di un altro
    video ('swap') o con frame grigi ('blank'). Se la risposta non cambia, il
    modello non sta usando il video. Il donatore e' portato esattamente a n
    frame, altrimenti una risposta diversa potrebbe dipendere dalla lunghezza.
    """
    from PIL import Image

    model.eval()
    count = min(args.video_dependence_samples, len(dataset))
    if count < 2:
        return None
    frames_cache = [decode_video_frames(dataset[index], args) for index in range(count)]
    identical, ious = 0, []
    for index in tqdm(range(count), desc="video dependence", unit="video", leave=False):
        sample = dataset[index]
        original_frames = frames_cache[index]
        if args.video_dependence_mode == "blank":
            substituted = [
                Image.new("RGB", original_frames[0].size, (128, 128, 128))
            ] * len(original_frames)
        else:
            donor = frames_cache[(index + 1) % count]
            substituted = (donor + [donor[-1]] * len(original_frames))[:len(original_frames)]
        original = generate_interval_response(
            model, processor, sample, args, device, original_frames, model_type
        )
        replaced = generate_interval_response(
            model, processor, sample, args, device, substituted, model_type
        )
        same = original == replaced
        identical += same
        iou = None
        try:
            first = parse_timeline_response(original, sample)
            second = parse_timeline_response(replaced, sample)
            union = sum(a | b for a, b in zip(first, second))
            iou = sum(a & b for a, b in zip(first, second)) / union if union else None
        except (json.JSONDecodeError, ValueError):
            pass
        if iou is not None:
            ious.append(iou)
        write_metric(
            metrics_file, run_id=run_id, record_type="video_dependence",
            split="validation", epoch=epoch, dataset_index=index,
            mode=args.video_dependence_mode, original=original,
            substituted=replaced, identical=same, iou=iou,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summary = {
        "samples": count,
        "mode": args.video_dependence_mode,
        "identical_output_rate": identical / count,
        "mean_iou_original_vs_substituted": sum(ious) / len(ious) if ious else None,
    }
    write_metric(
        metrics_file, run_id=run_id, record_type="video_dependence_summary",
        split="validation", epoch=epoch, video_time_mode=args.video_time_mode,
        **summary,
    )
    print(
        f"[video dependence] {args.video_dependence_mode}: "
        f"identical_output_rate={summary['identical_output_rate']:.3f} su {count} video, "
        f"IoU media={summary['mean_iou_original_vs_substituted']}"
    )
    return summary


def load_datasets(args):
    records = load_records(
        args.data_json, require_chosen=True, timeline_unit=args.timeline_unit
    )
    if args.val_json:
        train_records = records
        val_records = load_records(args.val_json, require_chosen=False)
    else:
        train_records, val_records = split_records(records, args.val_ratio, args.seed)
    val_video_dir = args.val_video_dir or args.video_dir
    train = VideoXumSFTDataset(train_records, args.video_dir, args.timeline_unit)
    val = VideoXumSFTDataset(val_records, val_video_dir, args.timeline_unit)
    print(f"[data] train={len(train)}, val={len(val)}, val_video_dir={val_video_dir}")
    return train, val


def load_validation_dataset(args):
    """Carica solo lo split necessario alla validation standalone."""
    if args.val_json:
        val_records = load_records(args.val_json, require_chosen=False)
    else:
        records = load_records(
            args.data_json, require_chosen=True, timeline_unit=args.timeline_unit
        )
        _, val_records = split_records(records, args.val_ratio, args.seed)
    val_video_dir = args.val_video_dir or args.video_dir
    val = VideoXumSFTDataset(val_records, val_video_dir, args.timeline_unit)
    print(f"[data] val={len(val)}, val_video_dir={val_video_dir}")
    return val


def validate_only(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if not args.model.is_dir():
        raise FileNotFoundError(f"Modello base locale non trovato: {args.model}")
    if not args.eval_checkpoint.is_dir():
        raise FileNotFoundError(
            f"Checkpoint adapter non trovato: {args.eval_checkpoint}"
        )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA non disponibile: eseguire su un nodo GPU")
    resolve_video_time_mode(args, args.eval_checkpoint, strict=False)
    resolve_best_metric(args)

    device = torch.device(args.device)
    results_dir = args.resultsroot / args.model.name
    results_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    metrics_path = results_dir / (
        f"validation-{args.eval_checkpoint.name}-{run_id}.jsonl"
    )
    val_dataset = load_validation_dataset(args)

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"[model] caricamento offline del modello base {args.model}")
    model, model_type = load_qwen_vl_model(
        args.model, torch_dtype=dtype, attn_implementation=args.attn_impl
    )
    model = model.to(device)
    print(f"[model] architecture={model_type}")
    print(f"[model] caricamento adapter {args.eval_checkpoint}")
    model = PeftModel.from_pretrained(
        model, args.eval_checkpoint, is_trainable=False, local_files_only=True
    )
    processor = AutoProcessor.from_pretrained(
        args.eval_checkpoint, local_files_only=True
    )
    model.config.use_cache = True

    with metrics_path.open("w", encoding="utf-8", buffering=1) as metrics_file:
        write_metric(
            metrics_file,
            run_id=run_id,
            record_type="config",
            mode="validation_only",
            checkpoint=args.eval_checkpoint,
            arguments=vars(args),
        )
        evaluate(
            model, processor, val_dataset, args, device,
            metrics_file, run_id, epoch=0, model_type=model_type,
        )
        if args.video_dependence_samples > 0:
            measure_video_dependence(
                model, processor, val_dataset, args, device,
                metrics_file, run_id, 0, model_type,
            )
    print(f"[done] validation checkpoint {args.eval_checkpoint}")
    print(f"[done] metriche salvate in {metrics_path}")


def train(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if not args.model.is_dir():
        raise FileNotFoundError(f"Checkpoint locale non trovato: {args.model}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA non disponibile: eseguire su un nodo GPU")
    start_time = time.monotonic()
    resolve_video_time_mode(args, args.resume_from, default="seconds")
    resolve_best_metric(args)
    device = torch.device(args.device)
    save_dir = args.saveroot / args.model.name
    results_dir = args.resultsroot / args.model.name
    if not args.resume_from:
        guard_checkpoint_dir(save_dir, args)
    save_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    metrics_path = results_dir / f"metrics-{run_id}.jsonl"
    metrics_file = metrics_path.open("w", encoding="utf-8", buffering=1)
    write_metric(
        metrics_file,
        run_id=run_id,
        record_type="config",
        arguments=vars(args),
    )
    print(f"[metrics] log JSONL: {metrics_path}")
    train_dataset, val_dataset = load_datasets(args)

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"[model] caricamento offline di {args.model}")
    model, model_type = load_qwen_vl_model(
        args.model, torch_dtype=dtype, attn_implementation=args.attn_impl
    )
    model = model.to(device)
    print(f"[model] architecture={model_type}")
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    if args.resume_from:
        model = PeftModel.from_pretrained(
            model, args.resume_from, is_trainable=True, local_files_only=True
        )
    else:
        lora_targets = [name.strip() for name in args.lora_target_modules.split(",") if name.strip()]
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r,
            lora_alpha=2 * args.lora_r,
            lora_dropout=args.lora_dropout,
            target_modules=lora_targets,
            bias="none",
            task_type="CAUSAL_LM",
        ))
        print(f"[lora] target_modules={lora_targets}")

    if args.tune_visual_merger:
        merger_params = [
            (name, parameter) for name, parameter in model.named_parameters()
            if "merger" in name.lower()
        ]
        if not merger_params:
            print(
                "[WARN] --tune_visual_merger attivo ma nessun modulo con "
                "'merger' nel nome e' stato trovato: controllare i nomi reali "
                "con model.named_modules() per questa architettura."
            )
        else:
            for name, parameter in merger_params:
                parameter.requires_grad_(True)
            print(
                f"[lora] sbloccati {len(merger_params)} tensori del connettore "
                "vision-language (full fine-tune, fuori da LoRA): "
                + ", ".join(name for name, _ in merger_params[:5])
                + (" ..." if len(merger_params) > 5 else "")
            )

    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    n_samples = min(args.debug_samples, len(train_dataset)) if args.debug_samples else len(train_dataset)
    total_steps = math.ceil(n_samples / args.accum_steps) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, round(total_steps * args.warmup_ratio), total_steps
    )
    print(f"[train] samples={n_samples}, steps={total_steps}")

    best_score, global_step = -math.inf, 0
    start_epoch, samples_seen_in_epoch = 0, 0
    if args.resume_from:
        state_path = args.resume_from / TRAINING_STATE_NAME
        if state_path.is_file():
            state = torch.load(state_path, map_location=device, weights_only=False)
            saved_samples = state.get("n_samples")
            if saved_samples is not None and saved_samples != n_samples:
                raise SystemExit(
                    f"[resume] {args.resume_from} e' stato addestrato con "
                    f"{saved_samples} campioni per epoca, questo job ne usa "
                    f"{n_samples}: passare lo stesso --debug_samples, altrimenti "
                    "la posizione nel dataset e lo scheduler non corrispondono."
                )
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            global_step = int(state["global_step"])
            start_epoch = int(state["epoch"])
            samples_seen_in_epoch = int(state["samples_seen_in_epoch"])
            # Gli stati precedenti salvano il best in best_f1_mean e selezionano
            # sempre su f1_mean: il valore e' riusato solo se il criterio coincide.
            if state.get("best_metric", "f1_mean") == args.best_metric:
                best_score = float(state.get("best_f1_mean", -math.inf))
            random.setstate(state["python_rng_state"])
            # torch.load qui sopra usa map_location=device, quindi anche gli stati
            # RNG tornano su GPU, mentre set_rng_state* pretende ByteTensor su CPU
            # (stessa normalizzazione di train_dpo.py).
            cpu_rng_state = torch.as_tensor(
                state["torch_rng_state"], dtype=torch.uint8, device="cpu"
            ).contiguous()
            torch.set_rng_state(cpu_rng_state)
            if device.type == "cuda" and state.get("cuda_rng_state") is not None:
                saved_cuda_rng = state["cuda_rng_state"]
                # I checkpoint possono contenere una lista di ByteTensor, oppure
                # un tensore/lista serializzato su CPU/GPU.
                if isinstance(saved_cuda_rng, torch.Tensor):
                    saved_cuda_rng = [saved_cuda_rng]
                cuda_rng_states = [
                    torch.as_tensor(
                        rng_state, dtype=torch.uint8, device="cpu"
                    ).contiguous()
                    for rng_state in saved_cuda_rng
                ]
                torch.cuda.set_rng_state_all(cuda_rng_states)
            print(
                f"[resume] stato completo da {state_path}: step={global_step}, "
                f"epoch={start_epoch + 1}, campioni_gia_visti={samples_seen_in_epoch}"
            )
        else:
            match = re.fullmatch(r"checkpoint-(\d+)", args.resume_from.name)
            if match is None:
                raise ValueError(
                    f"{args.resume_from} non contiene {TRAINING_STATE_NAME} e il nome "
                    "non permette di ricavare lo step (atteso checkpoint-N)"
                )
            global_step = int(match.group(1))
            steps_per_epoch = math.ceil(n_samples / args.accum_steps)
            start_epoch, step_in_epoch = divmod(global_step, steps_per_epoch)
            samples_seen_in_epoch = min(step_in_epoch * args.accum_steps, n_samples)
            for _ in range(global_step):
                optimizer.step()
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            print(
                f"[resume] checkpoint legacy: riparto da step={global_step}, "
                f"epoch={start_epoch + 1}, campioni_gia_visti={samples_seen_in_epoch}. "
                "Optimizer non disponibile: viene reinizializzato; scheduler allineato."
            )

    def save_checkpoint(path, next_epoch, next_samples_seen):
        path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(path)
        processor.save_pretrained(path)
        write_video_encoding(path, args)
        state = {
            "global_step": global_step,
            "epoch": next_epoch,
            "samples_seen_in_epoch": next_samples_seen,
            "n_samples": n_samples,
            "best_metric": args.best_metric,
            "best_f1_mean": best_score,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "python_rng_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        }
        temporary_state = path / f"{TRAINING_STATE_NAME}.tmp"
        torch.save(state, temporary_state)
        temporary_state.replace(path / TRAINING_STATE_NAME)

    def update_best(video_metrics, next_epoch, next_samples_seen):
        nonlocal best_score
        if video_metrics["selection_score"] > best_score:
            best_score = video_metrics["selection_score"]
            save_checkpoint(save_dir / "checkpoint-best", next_epoch, next_samples_seen)
            print(f"[best] checkpoint-best aggiornato: {args.best_metric}={best_score:.4f}")

    def stop_if_out_of_time(next_epoch, next_samples_seen):
        """Salva uno stato riprendibile ed esce prima del limite SLURM."""
        elapsed_hours = (time.monotonic() - start_time) / 3600
        if not args.time_budget_hours or elapsed_hours < args.time_budget_hours:
            return
        checkpoint = save_dir / f"checkpoint-{global_step}"
        save_checkpoint(checkpoint, next_epoch, next_samples_seen)
        (save_dir / RESUME_POINTER_NAME).write_text(f"{checkpoint}\n", encoding="utf-8")
        write_metric(
            metrics_file, run_id=run_id, record_type="time_budget_exit",
            elapsed_hours=elapsed_hours, optimizer_step=global_step,
            epoch=next_epoch + 1, samples_seen_in_epoch=next_samples_seen,
            resume_from=checkpoint,
        )
        metrics_file.close()
        print(
            f"[time budget] {elapsed_hours:.2f}h >= {args.time_budget_hours}h: "
            f"riprendere da {checkpoint}"
        )
        sys.exit(TIME_BUDGET_EXIT_CODE)

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        indices = list(range(len(train_dataset)))
        random.Random(args.seed + epoch).shuffle(indices)
        indices = indices[:n_samples]
        skip_samples = samples_seen_in_epoch if epoch == start_epoch else 0
        if skip_samples > len(indices):
            raise ValueError(
                f"Resume non valido: {skip_samples} campioni gia visti, "
                f"ma l'epoca ne contiene {len(indices)}"
            )
        accumulated_loss = 0.0
        epoch_loss_sum, epoch_loss_count = 0.0, 0
        progress = tqdm(
            indices[skip_samples:], desc=f"epoch {epoch + 1}/{args.epochs}", unit="video"
        )
        for position, index in enumerate(progress, start=skip_samples + 1):
            try:
                inputs, completion_tokens = prepare_sft_example(
                    processor, train_dataset[index], args, device, model_type
                )
                outputs = model(**inputs)
                raw_loss = outputs.loss
                (raw_loss / args.accum_steps).backward()
            except torch.cuda.OutOfMemoryError as error:
                # Video anomalo (troppi frame/token): si scarta il sample invece
                # di far crashare l'intero job. Si libera subito la memoria
                # residua e si azzerano eventuali gradienti parziali accumulati
                # per questo sample prima di proseguire.
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                write_metric(
                    metrics_file,
                    run_id=run_id,
                    record_type="sample",
                    split="train",
                    epoch=epoch + 1,
                    sample_position=position,
                    dataset_index=index,
                    optimizer_step=global_step + 1,
                    skipped_oom=True,
                    error=str(error),
                )
                print(
                    f"[WARN] OOM sul sample {index} (video_id="
                    f"{train_dataset.records[index]['video_id']}): saltato.",
                    file=__import__("sys").stderr,
                )
                continue

            write_metric(
                metrics_file,
                run_id=run_id,
                record_type="sample",
                split="train",
                epoch=epoch + 1,
                sample_position=position,
                dataset_index=index,
                optimizer_step=global_step + 1,
                completion_tokens=completion_tokens,
                sft_loss=raw_loss.detach().item(),
                learning_rate=optimizer.param_groups[0]["lr"],
            )
            accumulated_loss += raw_loss.item()
            epoch_loss_sum += raw_loss.item()
            epoch_loss_count += 1

            if position % args.accum_steps == 0 or position == len(indices):
                torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                divisor = args.accum_steps if position % args.accum_steps == 0 else position % args.accum_steps
                shown_loss, accumulated_loss = accumulated_loss / divisor, 0.0
                progress.set_postfix(step=global_step, loss=f"{shown_loss:.4f}")
                if global_step % args.save_steps == 0:
                    checkpoint = save_dir / f"checkpoint-{global_step}"
                    save_checkpoint(checkpoint, epoch, position)
                    print(f"[checkpoint] salvato {checkpoint}")
                if (
                    args.eval_steps
                    and global_step % args.eval_steps == 0
                    and position < len(indices)
                ):
                    # I logits dell'ultimo forward occupano diversi GB: liberarli
                    # prima della generazione di validation.
                    inputs = outputs = raw_loss = None
                    stop_if_out_of_time(epoch, position)
                    step_metrics = evaluate(
                        model, processor, val_dataset, args, device,
                        metrics_file, run_id, epoch + 1, model_type,
                        limit=args.eval_steps_samples, optimizer_step=global_step,
                    )
                    update_best(step_metrics, epoch, position)
                    model.train()
                stop_if_out_of_time(epoch, position)
            del inputs, outputs, raw_loss
            if device.type == "cuda":
                torch.cuda.empty_cache()

        write_metric(
            metrics_file,
            run_id=run_id,
            record_type="summary",
            split="train",
            epoch=epoch + 1,
            samples=epoch_loss_count,
            train_loss_mean=(epoch_loss_sum / epoch_loss_count) if epoch_loss_count else None,
        )
        # La validation di fine epoca parte solo se c'e' ancora budget: un job
        # ripreso da questo checkpoint salta tutti i campioni e valida subito.
        stop_if_out_of_time(epoch, len(indices))
        video_metrics = evaluate(
            model, processor, val_dataset, args, device,
            metrics_file, run_id, epoch + 1, model_type,
            optimizer_step=global_step,
        )
        checkpoint = save_dir / f"checkpoint-epoch{epoch + 1}"
        save_checkpoint(checkpoint, epoch + 1, 0)
        print(f"[epoch {epoch + 1}] F1_MEAN={video_metrics['f1_mean']:.4f}")
        update_best(video_metrics, epoch + 1, 0)

    final_path = save_dir / "final"
    save_checkpoint(final_path, args.epochs, 0)
    if args.diversity_check_samples > 0:
        measure_generation_diversity(
            model, processor, val_dataset, args, device,
            metrics_file, run_id, args.epochs, model_type,
        )
    if args.video_dependence_samples > 0:
        measure_video_dependence(
            model, processor, val_dataset, args, device,
            metrics_file, run_id, args.epochs, model_type,
        )
    metrics_file.close()
    print(f"[done] adapter finale salvato in {final_path}")
    print(f"[done] metriche salvate in {metrics_path}")


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("deve essere un intero positivo")
    return parsed


def add_video_time_argument(parser):
    parser.add_argument(
        "--video_time_mode", choices=("auto", "legacy", "seconds"), default="auto",
        help="Marker temporali <t seconds> di Qwen3-VL per i frame decodificati. "
             "seconds: coincidono con i secondi reali; legacy: comportamento "
             "precedente (meta' del tempo reale). auto: eredita la codifica dal "
             "checkpoint di partenza (legacy se non ha video_encoding.json), "
             "seconds se si parte dal modello base; sempre legacy con "
             "--timeline_unit pair.",
    )
    parser.add_argument(
        "--timeline_unit", choices=("auto",) + TIMELINE_UNITS, default="auto",
        help="Indici predetti dal modello. frame: uno per frame a 1 fps; pair: "
             "uno per coppia di frame, come le posizioni temporali di Qwen3-VL "
             "(richiede label a coppie). auto: eredita dal checkpoint di "
             "partenza, frame se si parte dal modello base.",
    )


def add_pixel_arguments(parser):
    parser.add_argument(
        "--max_pixels", type=positive_int, default=None,
        help=f"pixel massimi per frame; se omesso eredita il valore del "
             f"checkpoint di partenza, altrimenti {DEFAULT_MAX_PIXELS}",
    )
    parser.add_argument(
        "--min_pixels", type=positive_int, default=None,
        help=f"pixel minimi per frame (<= --max_pixels); se omesso eredita il "
             f"valore del checkpoint di partenza, altrimenti {DEFAULT_MIN_PIXELS}",
    )


def resolve_best_metric(args):
    """kappa per le timeline a coppie: f1_mean premia chi seleziona piu' frame."""
    if args.best_metric is None:
        args.best_metric = "kappa" if args.timeline_unit == "pair" else "f1_mean"
    print(f"[best] checkpoint-best selezionato con {args.best_metric}")


def add_training_schedule_arguments(parser):
    parser.add_argument(
        "--eval_steps", type=int, default=0,
        help="se >0 valida ogni N step su --eval_steps_samples video e aggiorna "
             "checkpoint-best; 0 valida solo a fine epoca",
    )
    parser.add_argument("--eval_steps_samples", type=positive_int, default=200)
    parser.add_argument(
        "--best_metric", choices=METRIC_NAMES, default=None,
        help="metrica di validation per checkpoint-best; default kappa con "
             "--timeline_unit pair, f1_mean altrimenti",
    )
    parser.add_argument(
        "--time_budget_hours", type=float, default=0.0,
        help="se >0, dopo questo tempo salva un checkpoint riprendibile ed esce "
             f"con codice {TIME_BUDGET_EXIT_CODE}; lasciare margine per la "
             "validation di fine epoca",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="SFT Qwen2.5-VL su video VideoXum usando le risposte 'chosen'"
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data_json", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--video_dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--val_json", type=Path, default=DEFAULT_VAL_DATA)
    parser.add_argument(
        "--val_video_dir", type=Path, default=DEFAULT_VAL_VIDEO_DIR
    )
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--saveroot", type=Path, default=DEFAULT_SAVEROOT)
    parser.add_argument("--resultsroot", type=Path, default=DEFAULT_RESULTSROOT)
    parser.add_argument("--resume_from", type=Path)
    parser.add_argument(
        "--eval_only", action="store_true",
        help="esegue soltanto la validation generativa di un adapter esistente",
    )
    parser.add_argument(
        "--eval_checkpoint", type=Path,
        help="directory del checkpoint PEFT/LoRA da usare con --eval_only",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn_impl", choices=("sdpa", "flash_attention_2"), default="sdpa")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=positive_int, default=3)
    parser.add_argument("--accum_steps", type=positive_int, default=8)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--fps", type=float, choices=(1.0,), default=1.0,
        help="campionamento fisso VideoXum: un frame per secondo",
    )
    parser.add_argument(
        "--max_frames", type=int, default=0,
        help="limite di sicurezza; 0 non tronca la timeline a 1 FPS",
    )
    add_video_time_argument(parser)
    add_pixel_arguments(parser)
    parser.add_argument("--lora_r", type=positive_int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules", type=str,
        default="q_proj,k_proj,v_proj,o_proj",
        help="Moduli LoRA separati da virgola. Il default q/v-only dei paper "
             "generici e' spesso troppo stretto per un task di grounding "
             "posizionale (indici di frame); q,k,v,o e' un compromesso "
             "ragionevole. Aggiungere gate_proj,up_proj,down_proj se la VRAM lo "
             "consente.",
    )
    parser.add_argument(
        "--tune_visual_merger", action="store_true",
        help="Include nei parametri allenabili (fuori da LoRA, full fine-tune) "
             "i moduli del connettore vision-language (nomi contenenti "
             "'merger' o 'visual.merger'), se presenti nel modello. Se questi "
             "moduli restano congelati, la SFT adatta solo il pesare "
             "linguistico dell'informazione visiva, non la proiezione delle "
             "feature visive stesse: per un task di riconoscimento di frame "
             "salienti puo' essere il fattore limitante.",
    )
    parser.add_argument("--save_steps", type=positive_int, default=100)
    parser.add_argument("--eval_samples", type=int, default=200,
                        help="numero di video di validation; <=0 usa tutto")
    parser.add_argument(
        "--eval_max_new_tokens", type=positive_int, default=256,
        help="token massimi per gli intervalli generati durante la validation",
    )
    add_training_schedule_arguments(parser)
    parser.add_argument(
        "--video_dependence_samples", type=int, default=0,
        help="se >0, rigenera N video di validation con frame sostituiti per "
             "verificare che l'output dipenda dal video (measure_video_dependence)",
    )
    parser.add_argument(
        "--video_dependence_mode", choices=("swap", "blank"), default="swap",
        help="swap: frame di un altro video; blank: frame grigi",
    )
    parser.add_argument("--debug_samples", type=int, default=0,
                        help="se >0 limita il training a N video")
    parser.add_argument(
        "--diversity_check_samples", type=int, default=0,
        help="se >0, a fine training campiona N video di validation con "
             "temperatura per misurare il rischio di mode collapse prima di "
             "passare a GRPO (vedi measure_generation_diversity)",
    )
    parser.add_argument("--diversity_check_k", type=positive_int, default=4,
                        help="numero di generazioni campionate per video nel diversity check")
    parser.add_argument("--diversity_temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.eval_steps and cli_args.eval_samples > 0 and cli_args.eval_samples < cli_args.eval_steps_samples:
        raise SystemExit(
            "--eval_samples deve essere >= --eval_steps_samples: checkpoint-best "
            "confronta tutte le validation sui primi --eval_steps_samples video"
        )
    if cli_args.eval_only:
        if cli_args.eval_checkpoint is None:
            raise SystemExit("--eval_only richiede --eval_checkpoint CHECKPOINT")
        validate_only(cli_args)
    elif cli_args.eval_checkpoint is not None:
        raise SystemExit("--eval_checkpoint richiede --eval_only")
    else:
        train(cli_args)
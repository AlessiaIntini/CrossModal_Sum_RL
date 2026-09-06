#!/usr/bin/env python3
"""LoRA-DPO Qwen2.5-VL su video e frame chosen/rejected VideoXum."""

import argparse
import json
import math
import os
import random
import re
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
from utils import compute_kendall, compute_spearman
from vl_model_loader import load_qwen_vl_model

PROJECT_ROOT = Path(__file__).resolve().parent
VIDEOXUM_ROOT = PROJECT_ROOT / "dataset" / "aintini" / "datasets" / "videoxum"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "Qwen2.5-VL-7B-Instruct"
DEFAULT_DATA = (
    VIDEOXUM_ROOT / "metadata" / "derived" / "train_videoxum_dpo.json"
)
DEFAULT_VIDEO_DIR = VIDEOXUM_ROOT / "train"
DEFAULT_VAL_DATA = VIDEOXUM_ROOT / "metadata" / "val_videoxum.json"
DEFAULT_VAL_VIDEO_DIR = VIDEOXUM_ROOT / "val"
DEFAULT_SAVEROOT = PROJECT_ROOT / "outputs" / "dpo" / "checkpoints"
DEFAULT_RESULTSROOT = PROJECT_ROOT / "outputs" / "dpo" / "results"
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v")
TRAINING_STATE_NAME = "training_state.pt"
SYSTEM_PROMPT = (
    "You are a video summarization assistant. Select the frames that best "
    "summarize the important events in the video."
)


def format_response(onehot):
    """Converte una label one-hot in intervalli JSON inclusivi [inizio, fine]."""
    if not isinstance(onehot, list) or not onehot:
        raise ValueError("label chosen/rejected vuota o non valida")
    if any(value not in (0, 1, False, True) for value in onehot):
        raise ValueError("le label chosen/rejected devono contenere solo 0 e 1")

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


def load_records(path, require_preferences=True):
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    items = payload.items() if isinstance(payload, dict) else enumerate(payload)
    records = []
    for key, record in items:
        required = {"sampled_frames", "vsum_onehot"}
        preference_fields = {"vsum_onehot_chosen", "vsum_onehot_rejected"}
        if require_preferences:
            required |= preference_fields
        if missing := required - set(record):
            raise ValueError(f"Record {key!r}: campi mancanti {sorted(missing)}")
        present_preferences = preference_fields & set(record)
        if present_preferences and present_preferences != preference_fields:
            raise ValueError(
                f"Record {key!r}: coppia chosen/rejected incompleta"
            )
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
        if present_preferences:
            chosen = record["vsum_onehot_chosen"]
            rejected = record["vsum_onehot_rejected"]
            if len(chosen) != frame_count or len(rejected) != frame_count:
                raise ValueError(
                    f"Record {key!r}: sampled_frames={frame_count}, "
                    f"len(chosen)={len(chosen)}, len(rejected)={len(rejected)}"
                )
            if chosen == rejected:
                raise ValueError(f"Record {key!r}: chosen e rejected coincidono")
            item["chosen"] = chosen
            item["rejected"] = rejected
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


class VideoXumDPODataset(Dataset):
    def __init__(self, records, video_dir):
        self.records = records
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
        sample = {
            "video_path": str(self.paths[index]),
            "sampled_frames": count,
            # Prompt precedente:
            # "The timeline has {count} candidate frames, numbered from 0 to "
            # "{count - 1}, sampled at one frame per second. Return only a JSON "
            # "array of inclusive [start, end] frame-index intervals that form "
            # "the best extractive video summary, in chronological order. "
            # "Represent an isolated frame i as [i, i]."
            "prompt": (
                f"The timeline contains {count} candidate frames, numbered from 0 "
                f"to {count - 1}, sampled at one frame per second. Select the "
                "frames that form the best extractive video summary, in chronological "
                "order. The numbers are frame indices, NOT timestamps or seconds. "
                "Return exactly one valid JSON array of inclusive integer pairs in "
                "the form [[start_frame,end_frame],...]. Example: [[0,2],[5,6]]. "
                "Use only integers. Represent an isolated frame i as [i,i]. Do not "
                "use objects, decimal numbers, numeric strings, Markdown, code "
                "fences, or explanatory text."
            ),
            "references": record["references"],
        }
        if "chosen" in record:
            sample["chosen"] = format_response(record["chosen"])
            sample["rejected"] = format_response(record["rejected"])
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

    # if len(frames) != expected_frames:
    #     raise ValueError(
    #         f"A 1 FPS sono stati estratti {len(frames)} frame da "
    #         f"{sample['video_path']}, ma le label ne richiedono "
    #         f"{expected_frames}."
    #     )
    return frames


def build_messages(sample, args, response=None, video_frames=None):
    video = {
        "type": "video",
        "video": video_frames if video_frames is not None else sample["video_path"],
        "max_pixels": args.max_pixels,
        # qwen_vl_utils usa 128*28*28 come minimo predefinito per i video.
        # Rendendolo configurabile possiamo ridurre max_pixels senza ottenere
        # l'assert "max_pixels >= min_pixels" durante il preprocessing.
        "min_pixels": args.min_pixels,
    }
    if video_frames is not None:
        video["fps"] = args.fps
        video["nframes"] = len(video_frames)
    elif args.max_frames and sample["sampled_frames"] > args.max_frames:
        nframes = max(2, min(args.max_frames, sample["sampled_frames"]))
        video["nframes"] = nframes - nframes % 2
    else:
        video["fps"] = args.fps
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
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
    
        
    images, videos, video_kwargs = process_video_inputs(prompt_messages, model_type)

    common = {
    "images": images,
    "videos": videos,
    "padding": False,
    "return_tensors": "pt",
    "cap_pixels_per_frame": True,
    **video_kwargs,
    }
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


def completion_log_prob(model, inputs, prompt_len):
    input_ids = inputs["input_ids"][0]
    with torch.autocast(
        device_type=input_ids.device.type,
        dtype=torch.bfloat16,
        enabled=input_ids.device.type == "cuda",
    ):
        outputs = model(**inputs)
    logits = outputs.logits[0, prompt_len - 1:input_ids.numel() - 1].float()
    labels = input_ids[prompt_len:]
    del outputs
    return F.log_softmax(logits, dim=-1).gather(1, labels[:, None]).sum()


def prepare_pair(processor, sample, args, device, model_type, video_frames=None):
    frames = video_frames if video_frames is not None else decode_video_frames(sample, args)
    chosen, chosen_start = prepare_completion(
        processor, sample, sample["chosen"], args, model_type, frames
    )
    rejected, rejected_start = prepare_completion(
        processor, sample, sample["rejected"], args, model_type, frames
    )
    return (
        move_inputs(chosen, device), chosen_start,
        move_inputs(rejected, device), rejected_start,
    )


def dpo_loss(chosen_pol, rejected_pol, chosen_ref, rejected_ref, beta):
    policy_margin = chosen_pol - rejected_pol
    reference_margin = chosen_ref.detach() - rejected_ref.detach()
    return -F.logsigmoid(beta * (policy_margin - reference_margin))


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


def binary_f1(prediction, reference):
    """F1 frame-level equivalente a sklearn per due vettori binari."""
    true_positive = sum(p == 1 and r == 1 for p, r in zip(prediction, reference))
    denominator = sum(prediction) + sum(reference)
    return 2.0 * true_positive / denominator if denominator else 0.0


def videoxum_metrics(prediction, references):
    """Protocollo VideoXum adattato a una predizione generativa binaria."""
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
    }


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


@torch.no_grad()
def evaluate(
    model, processor, dataset, args, device, metrics_file, run_id, epoch,
    model_type,
):
    model.eval()
    limit = len(dataset) if args.eval_samples <= 0 else min(args.eval_samples, len(dataset))
    correct, margins, invalid_predictions = 0, [], 0
    metric_values = {
        name: [] for name in ("f1_max", "f1_mean", "kendall", "spearman")
    }
    for index in tqdm(range(limit), desc="validation", unit="video", leave=False):
        sample = dataset[index]
        frames = decode_video_frames(sample, args)
        chosen_tokens = rejected_tokens = margin = None
        if "chosen" in sample:
            chosen, chosen_start, rejected, rejected_start = prepare_pair(
                processor, sample, args, device, model_type, video_frames=frames
            )
            chosen_lp = completion_log_prob(model, chosen, chosen_start)
            rejected_lp = completion_log_prob(model, rejected, rejected_start)
            margin = (chosen_lp - rejected_lp).item()
            chosen_tokens = chosen["input_ids"].shape[-1] - chosen_start
            rejected_tokens = rejected["input_ids"].shape[-1] - rejected_start
            correct += margin > 0
            margins.append(margin)
            del chosen, rejected, chosen_lp, rejected_lp

        response = generate_interval_response(
            model, processor, sample, args, device, frames, model_type
        )
        parse_error = None
        try:
            prediction = parse_interval_response(
                response, sample["sampled_frames"]
            )
        except (json.JSONDecodeError, ValueError) as error:
            prediction = [0] * sample["sampled_frames"]
            parse_error = str(error)
            invalid_predictions += 1
        sample_metrics = videoxum_metrics(prediction, sample["references"])
        for name, value in sample_metrics.items():
            metric_values[name].append(value)

        write_metric(
            metrics_file,
            run_id=run_id,
            record_type="sample",
            split="validation",
            epoch=epoch,
            sample_position=index + 1,
            dataset_index=index,
            chosen_completion_tokens=chosen_tokens,
            rejected_completion_tokens=rejected_tokens,
            policy_margin=margin,
            generated_response=response,
            prediction_valid=parse_error is None,
            parse_error=parse_error,
            selected_frames=sum(prediction),
            selection_ratio=sum(prediction) / len(prediction),
            **sample_metrics,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if margins:
        accuracy = correct / len(margins)
        mean_margin = sum(margins) / len(margins)
    else:
        accuracy = mean_margin = None
    summary_metrics = {
        name: sum(values) / len(values)
        for name, values in metric_values.items()
    }
    if margins:
        print(
            f"[val] pairwise_accuracy={accuracy:.4f} ({correct}/{len(margins)}) "
            f"mean_logprob_margin={mean_margin:.4f}"
        )
    else:
        print("[val] pairwise DPO non disponibile: validation senza chosen/rejected")
    print(
        f"[val] VideoXum F1_MAX={summary_metrics['f1_max']:.4f} "
        f"F1_MEAN={summary_metrics['f1_mean']:.4f} "
        f"Kendall={summary_metrics['kendall']:.4f} "
        f"Spearman={summary_metrics['spearman']:.4f} "
        f"valid_predictions={limit - invalid_predictions}/{limit}"
    )
    write_metric(
        metrics_file,
        run_id=run_id,
        record_type="summary",
        split="validation",
        epoch=epoch,
        samples=limit,
        pairwise_samples=len(margins),
        pairwise_accuracy=accuracy,
        mean_policy_margin=mean_margin,
        invalid_predictions=invalid_predictions,
        metric_protocol="VideoXum aggregation; binary generated saliency",
        **summary_metrics,
    )
    return accuracy, mean_margin, summary_metrics


def load_datasets(args):
    records = load_records(args.data_json, require_preferences=True)
    if args.val_json:
        train_records = records
        val_records = load_records(args.val_json, require_preferences=False)
    else:
        train_records, val_records = split_records(records, args.val_ratio, args.seed)
    val_video_dir = args.val_video_dir or args.video_dir
    train = VideoXumDPODataset(train_records, args.video_dir)
    val = VideoXumDPODataset(val_records, val_video_dir)
    print(f"[data] train={len(train)}, val={len(val)}, val_video_dir={val_video_dir}")
    return train, val


def load_validation_dataset(args):
    """Carica solo lo split necessario alla validation standalone."""
    if args.val_json:
        val_records = load_records(args.val_json, require_preferences=False)
    else:
        records = load_records(args.data_json, require_preferences=True)
        _, val_records = split_records(records, args.val_ratio, args.seed)
    val_video_dir = args.val_video_dir or args.video_dir
    val = VideoXumDPODataset(val_records, val_video_dir)
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
    print(f"[done] validation checkpoint {args.eval_checkpoint}")
    print(f"[done] metriche salvate in {metrics_path}")


def train(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if not args.model.is_dir():
        raise FileNotFoundError(f"Checkpoint locale non trovato: {args.model}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA non disponibile: eseguire su un nodo GPU")
    device = torch.device(args.device)
    save_dir = args.saveroot / args.model.name
    results_dir = args.resultsroot / args.model.name
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
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r,
            lora_alpha=2 * args.lora_r,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        ))
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

    best_f1_mean, global_step = -1.0, 0
    start_epoch, samples_seen_in_epoch = 0, 0
    if args.resume_from:
        state_path = args.resume_from / TRAINING_STATE_NAME
        if state_path.is_file():
            state = torch.load(state_path, map_location=device, weights_only=False)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            global_step = int(state["global_step"])
            start_epoch = int(state["epoch"])
            samples_seen_in_epoch = int(state["samples_seen_in_epoch"])
            best_f1_mean = float(state.get("best_f1_mean", -1.0))
            random.setstate(state["python_rng_state"])
            torch.set_rng_state(state["torch_rng_state"].cpu())
            if device.type == "cuda" and state.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state_all(state["cuda_rng_state"])
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
            # Il vecchio formato contiene solo l'adapter: allineiamo almeno il LR.
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
        state = {
            "global_step": global_step,
            "epoch": next_epoch,
            "samples_seen_in_epoch": next_samples_seen,
            "best_f1_mean": best_f1_mean,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "python_rng_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        }
        temporary_state = path / f"{TRAINING_STATE_NAME}.tmp"
        torch.save(state, temporary_state)
        temporary_state.replace(path / TRAINING_STATE_NAME)

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
        progress = tqdm(
            indices[skip_samples:], desc=f"epoch {epoch + 1}/{args.epochs}", unit="video"
        )
        for position, index in enumerate(progress, start=skip_samples + 1):
            try:
                chosen, chosen_start, rejected, rejected_start = prepare_pair(
                    processor, train_dataset[index], args, device, model_type
                )
                with model.disable_adapter(), torch.no_grad():
                    chosen_ref = completion_log_prob(model, chosen, chosen_start)
                    rejected_ref = completion_log_prob(model, rejected, rejected_start)
                # Conservare contemporaneamente i grafi autograd di chosen e rejected
                # raddoppia quasi il picco di memoria sui video lunghi. Manteniamo il
                # grafo chosen, misuriamo rejected senza grafo per ricavare il peso
                # scalare della loss DPO, quindi eseguiamo i due backward in sequenza.
                # Lo stato RNG viene ripristinato prima di ricalcolare rejected, così
                # anche il dropout LoRA usa la stessa maschera del forward di misura.
                chosen_pol = completion_log_prob(model, chosen, chosen_start)
                cpu_rng_before_rejected = torch.get_rng_state()
                cuda_rng_before_rejected = (
                    torch.cuda.get_rng_state(device) if device.type == "cuda" else None
                )
                with torch.no_grad():
                    rejected_pol_value = completion_log_prob(
                        model, rejected, rejected_start
                    )
                raw_loss = dpo_loss(
                    chosen_pol.detach(), rejected_pol_value,
                    chosen_ref, rejected_ref, args.beta,
                )
                chosen_tokens = chosen["input_ids"].shape[-1] - chosen_start
                rejected_tokens = rejected["input_ids"].shape[-1] - rejected_start
                policy_margin_tensor = chosen_pol.detach() - rejected_pol_value
                policy_margin = policy_margin_tensor.item()
                reference_margin = (chosen_ref - rejected_ref).item()
                margin_delta = policy_margin - reference_margin
                write_metric(
                    metrics_file,
                    run_id=run_id,
                    record_type="sample",
                    split="train",
                    epoch=epoch + 1,
                    sample_position=position,
                    dataset_index=index,
                    optimizer_step=global_step + 1,
                    chosen_completion_tokens=chosen_tokens,
                    rejected_completion_tokens=rejected_tokens,
                    policy_margin=policy_margin,
                    reference_margin=reference_margin,
                    preference_margin_delta=margin_delta,
                    dpo_logit=args.beta * margin_delta,
                    dpo_loss=raw_loss.detach().item(),
                    learning_rate=optimizer.param_groups[0]["lr"],
                )
                # Per L=-logsigmoid(beta * margin_delta):
                # dL/d(chosen)= -beta*sigmoid(-beta*margin_delta)
                # dL/d(rejected)=+beta*sigmoid(-beta*margin_delta)
                backward_scale = (
                    args.beta
                    * torch.sigmoid(-args.beta * (
                        policy_margin_tensor
                        - (chosen_ref.detach() - rejected_ref.detach())
                    ))
                    / args.accum_steps
                ).detach()
                chosen_pol.backward(gradient=-backward_scale)
                del chosen_pol

                torch.set_rng_state(cpu_rng_before_rejected)
                if cuda_rng_before_rejected is not None:
                    torch.cuda.set_rng_state(cuda_rng_before_rejected, device)
                rejected_pol = completion_log_prob(model, rejected, rejected_start)
                rejected_pol.backward(gradient=backward_scale)
                accumulated_loss += raw_loss.item()

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
                del (
                    chosen, rejected, chosen_ref, rejected_ref, rejected_pol,
                    rejected_pol_value, raw_loss, policy_margin_tensor,
                    backward_scale, cpu_rng_before_rejected,
                    cuda_rng_before_rejected,
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError as error:
                # Un video puo' richiedere piu' VRAM della GPU. I gradienti del
                # campione parziale non sono utilizzabili: li scartiamo e
                # proseguiamo, come avviene nella SFT.
                optimizer.zero_grad(set_to_none=True)
                accumulated_loss = 0.0
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

        accuracy, margin, video_metrics = evaluate(
            model, processor, val_dataset, args, device,
            metrics_file, run_id, epoch + 1, model_type,
        )
        checkpoint = save_dir / f"checkpoint-epoch{epoch + 1}"
        save_checkpoint(checkpoint, epoch + 1, 0)
        pairwise_summary = (
            f"val_accuracy={accuracy:.4f} val_margin={margin:.4f}"
            if accuracy is not None
            else "pairwise_unavailable"
        )
        print(
            f"[epoch {epoch + 1}] {pairwise_summary} "
            f"F1_MEAN={video_metrics['f1_mean']:.4f}"
        )
        if video_metrics["f1_mean"] > best_f1_mean:
            best_f1_mean = video_metrics["f1_mean"]
            best = save_dir / "checkpoint-best"
            save_checkpoint(best, epoch + 1, 0)

    final_path = save_dir / "final"
    save_checkpoint(final_path, args.epochs, 0)
    metrics_file.close()
    print(f"[done] adapter finale salvato in {final_path}")
    print(f"[done] metriche salvate in {metrics_path}")


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("deve essere un intero positivo")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description="DPO Qwen2.5-VL su video e coppie di frame VideoXum"
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
    parser.add_argument("--beta", type=float, default=0.1)
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
    parser.add_argument("--max_pixels", type=positive_int, default=256 * 28 * 28)
    parser.add_argument(
        "--min_pixels", type=positive_int, default=4 * 28 * 28,
        help="numero minimo di pixel per frame; deve essere <= --max_pixels",
    )
    parser.add_argument("--lora_r", type=positive_int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--save_steps", type=positive_int, default=100)
    parser.add_argument("--eval_samples", type=int, default=200,
                        help="numero di coppie di validation; <=0 usa tutto")
    parser.add_argument(
        "--eval_max_new_tokens", type=positive_int, default=256,
        help="token massimi per gli intervalli generati durante la validation",
    )
    parser.add_argument("--debug_samples", type=int, default=0,
                        help="se >0 limita il training a N video")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.min_pixels > cli_args.max_pixels:
        raise SystemExit(
            f"--min_pixels ({cli_args.min_pixels}) non puo' superare "
            f"--max_pixels ({cli_args.max_pixels})"
        )
    if cli_args.eval_only:
        if cli_args.eval_checkpoint is None:
            raise SystemExit("--eval_only richiede --eval_checkpoint CHECKPOINT")
        validate_only(cli_args)
    elif cli_args.eval_checkpoint is not None:
        raise SystemExit("--eval_checkpoint richiede --eval_only")
    else:
        train(cli_args)

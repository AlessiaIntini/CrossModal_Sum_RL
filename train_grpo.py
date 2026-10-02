#!/usr/bin/env python3
"""GRPO custom Qwen2.5-VL per frame discreti e summary VideoXum congiunti."""
from __future__ import annotations
import traceback
import argparse, json, math, os, random
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import av
import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import AutoProcessor
from transformers import get_cosine_schedule_with_warmup
from grpo_rewards import (DEFAULT_REWARD_WEIGHTS, F_BETA_AGGREGATIONS,
                          LENGTH_BUDGET_SIGMA, REFERENCE_COSINE_FLOOR,
                          REFERENCE_COSINE_SCALE, SUMMARY_LENGTH_SIGMA,
                          SentenceTransformerEncoder, VideoSummaryReward)
from sft_train_videoxum import (
    add_pixel_arguments,
    add_video_time_argument,
    apply_video_time_metadata,
    build_interval_rules,
    build_selection_requirements,
    build_timeline_intro,
    pad_decoded_frames,
    resolve_video_time_mode,
    timeline_length,
    write_video_encoding,
)
from vl_model_loader import load_qwen_vl_model


DEFAULT_MODEL = Path("/home/aintini/RL_VideoSum/GRPOVideo/models/Qwen2.5-VL-7B-Instruct")
DEFAULT_DATA = Path("/home/aintini/RL_VideoSum/GRPOVideo/dataset/aintini/datasets/videoxum/metadata/derived/train_videoxum_dpo.json")
DEFAULT_CHECKPOINT = Path("/home/aintini/RL_VideoSum/GRPOVideo/checkpoints/grpo-dpo")
DEFAULT_VIDEO_DIR = Path("/home/aintini/RL_VideoSum/GRPOVideo/dataset/aintini/datasets/videoxum/train")
DEFAULT_FEATURE_DIR = Path("/home/aintini/RL_VideoSum/GRPOVideo/dataset/aintini/datasets/videoxum/feat/blip")
SYSTEM_PROMPT = "You are a video summarization assistant. Select representative frames and write one concise summary."
# "concise" riproduce i prompt con cui sono stati addestrati i checkpoint GRPO
# fino al 2026-09-17 (una frase, ~25 parole). "reference" chiede la lunghezza dei
# reference VideoXum: train mediana 44 parole (p10-p90 26-84), 3.7 frasi.
SUMMARY_STYLES = ("reference", "concise")
# Centrato su ~45 parole, la media geometrica delle lunghezze dei reference e
# quindi l'ottimo di una policy che non puo' conoscere il bersaglio per-video.
# Con il vecchio (35, 55) il modello si piazzava sul bordo basso del range (37.3
# parole di media): prompt e reward chiedevano due cose diverse.
SUMMARY_WORD_RANGE = (42, 50)
SYSTEM_PROMPTS = {
    "concise": SYSTEM_PROMPT,
    "reference": "You are a video summarization assistant. Select representative frames and write a summary of the video.",
}


def summary_requirement(summary_style, word_range=None):
    """(istruzione sul summary, esempio illustrativo) per lo stile richiesto.

    ``word_range`` sostituisce SUMMARY_WORD_RANGE solo in inferenza: il training
    usa sempre il default, che e' anche quello del termine summary_length."""
    if summary_style == "concise":
        return ("Write exactly one concise summary of the video",
                "A short one-sentence description of the video.")
    low, high = word_range or SUMMARY_WORD_RANGE
    return (f"Write a summary of 3 to 4 sentences, about {low} to {high} words in "
            "total, describing the main events of the video in the order they happen",
            "First sentence describing the setting and the people. Second sentence "
            "describing the main action. Third sentence describing how the video ends.")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v")
# Adapter PEFT fuso nel modello base prima di addestrare il LoRA GRPO. Il LoRA
# salvato e' un delta rispetto a quel modello fuso: senza questo sidecar
# l'inferenza lo applicherebbe al Qwen base, perdendo SFT/DPO.
GRPO_INIT_NAME = "grpo_init.json"


def write_grpo_init(checkpoint, init_from):
    (Path(checkpoint) / GRPO_INIT_NAME).write_text(
        json.dumps({"init_from": str(init_from) if init_from else None}, indent=2) + "\n",
        encoding="utf-8",
    )


def read_grpo_init(checkpoint):
    """Adapter di partenza registrato nel checkpoint GRPO.

    Restituisce (trovato, init_from): trovato=False per i checkpoint che
    precedono il sidecar, per i quali l'adapter di partenza non e' ricostruibile.
    """
    path = Path(checkpoint) / GRPO_INIT_NAME
    if not path.is_file():
        return False, None
    init_from = json.loads(path.read_text(encoding="utf-8"))["init_from"]
    return True, Path(init_from) if init_from else None


def build_grpo_prompt(count, index_phrase=" indexed from", summary_style="concise", unit="frame",
                      word_range=None):
    """Prompt principale GRPO: selezione dei frame e summary.

    Riusa i blocchi condivisi con SFT e DPO (intro, selection requirements,
    interval rules) e sostituisce solo la sezione di output, perche' qui serve
    anche un summary testuale. Le regole di selezione devono restare identiche a
    quelle delle fasi precedenti: il warm-start arriva da un checkpoint DPO, e
    un prompt riformulato gli farebbe reimparare un compito solo apparentemente
    diverso. Il budget non viene espresso in numero di frame: quello della reward
    dipende dalla lunghezza del reference, che il modello non puo' osservare.
    ``index_phrase`` segue il checkpoint di partenza: " indexed from" per DPO
    (default), ", numbered from" per un warm-start SFT; con unit='pair' il testo
    della timeline e' unico e index_phrase e' ignorato. ``count`` e' la lunghezza
    della timeline nell'unita' ``unit``. Il campo JSON resta "selected_frames"
    anche con le coppie, perche' il parser della reward lo richiede.
    ``summary_style`` cambia solo le righe sul summary (vedi SUMMARY_STYLES).
    """
    requirement, example = summary_requirement(summary_style, word_range)
    item = "segment" if unit == "pair" else "frame"
    return (
        build_timeline_intro(count, index_phrase, unit)
        + build_selection_requirements(unit)

        + "Output requirements:\n"
        '- Return exactly one valid JSON object in the form:\n'
        f'  {{"selected_frames":[[start_{item},end_{item}],...],"summary":"..."}}\n'
        + build_interval_rules(count, unit)
        + f'- {requirement} in the "summary" field.\n'
        "- Do not return timestamps, seconds, decimal numbers, numeric strings, "
        f"per-{item} captions, Markdown, code fences, explanations, or additional "
        "fields.\n"
        "- Output raw JSON only, with no surrounding text.\n\n"

        f"Selecting substantially more {item}s than the target share above is "
        "progressively penalized, the more so the further you go beyond it. Do "
        "not select the whole timeline.\n\n"

        f'Valid example: {{"selected_frames":[[2,3],[10,14]],"summary":"{example}"}}\n'
        'Invalid example: {"selected_frames":[[0,2],[2,4]],"summary":"..."} '
        "because the intervals overlap.\n"
        'Invalid example: {"selected_frames":[[0,2],[3,4]],"summary":"..."} '
        "because consecutive intervals must be merged into [[0,4]].\n\n"

        "Return only the JSON object."
    )


def build_summary_only_prompt(count, summary_style="concise", unit="frame", word_range=None):
    """Prompt alternativo per l'inferenza che richiede soltanto il summary."""
    requirement, example = summary_requirement(summary_style, word_range)
    if summary_style == "concise":
        requirement, example = "Write exactly one grammatical sentence", "A short one-sentence description of the video."
    timeline = (
        f"The timeline contains {count} candidate segments, numbered from 0 to "
        f"{count - 1}; each segment covers two consecutive seconds of the video. "
        if unit == "pair" else
        f"The timeline contains {count} candidate frames, numbered from 0 "
        f"to {count - 1}, sampled at one frame per second. "
    )
    return (
        timeline +
        f"Only a {'concise ' if summary_style == 'concise' else ''}final summary is required, not the selected frames. "
        "The summary should capture the essential events and content of the "
        "video accurately. Do not include frame indices, timestamps, or any "
        f"other details. {requirement}. Do not use "
        "additional text, Markdown, code fences, or extra JSON fields. "
        'Return exactly one valid JSON object in the form {"summary":"..."}. '
        f'Example (illustrative only): {{"summary":"{example}"}}'
    )

def resolve_video(root, video_id):
    direct = root / video_id
    if direct.is_file(): return direct
    for ext in VIDEO_EXTENSIONS:
        path = root / f"{video_id}{ext}"
        if path.is_file(): return path
    return root / f"{video_id}.mp4"


def load_records(path):
    with path.open(encoding="utf-8") as handle: payload = json.load(handle)
    records = []
    for key, row in (payload.items() if isinstance(payload, dict) else enumerate(payload)):
        required = {"sampled_frames", "vsum_onehot", "vsum", "tsum"}
        if missing := required - set(row):
            raise ValueError(f"Record {key!r}: campi mancanti {sorted(missing)}")
        count, captions = int(row["sampled_frames"]), row["tsum"]
        if count <= 0 or np.asarray(row["vsum_onehot"]).shape != (10, count):
            raise ValueError(f"Record {key!r}: vsum_onehot non e' [10, {count}]")
        if not captions or np.asarray(row["vsum"]).shape != (len(captions), 10, 2):
            raise ValueError(f"Record {key!r}: vsum/tsum non coerenti")
        records.append(dict(video_id=str(row.get("video_id", key)), count=count,
                            vsum_onehot=row["vsum_onehot"], vsum=row["vsum"], tsum=captions))
    return records


class VideoXumGRPODataset(Dataset):
    def __init__(self, records, video_dir, feature_dir, split, summary_style="concise",
                 timeline_unit="frame"):
        self.records, self.split, self.summary_style = records, split, summary_style
        self.timeline_unit = timeline_unit
        self.videos = [resolve_video(video_dir, x["video_id"]) for x in records]
        self.features = [feature_dir / f'{x["video_id"]}.npz' for x in records]
        missing = [str(x) for x in self.videos + self.features if not x.is_file()]
        if missing: raise FileNotFoundError("File mancanti:\n  " + "\n  ".join(missing[:10]))

    def __len__(self): return len(self.records)

    def __getitem__(self, index):
        row, video, feature = self.records[index], self.videos[index], self.features[index]
        count = row["count"]
        prompt = build_grpo_prompt(
            timeline_length(count, self.timeline_unit),
            summary_style=self.summary_style, unit=self.timeline_unit,
        )
        return {**row, "video_path": str(video), "feature_path": str(feature),
                "cache_key": (self.split, row["video_id"]), "prompt": prompt}


def load_datasets(args):
    train = load_records(args.data_json)
    if args.val_json: val = load_records(args.val_json)
    else:
        if not 0 < args.val_ratio < 1:
            raise ValueError("--val_ratio deve essere compreso tra 0 e 1")
        ids = list(range(len(train))); random.Random(args.seed).shuffle(ids)
        n = max(1, round(len(train) * args.val_ratio)); val_ids = set(ids[:n])
        train, val = ([x for i, x in enumerate(train) if i not in val_ids],
                      [x for i, x in enumerate(train) if i in val_ids])
    return (VideoXumGRPODataset(train, args.video_dir, args.feature_dir, "train", args.summary_style,
                                args.timeline_unit),
            VideoXumGRPODataset(val, args.val_video_dir or args.video_dir,
                                args.val_feature_dir or args.feature_dir, "val", args.summary_style,
                                args.timeline_unit))


def ensure_reward_cache(engine, sample):
    if engine.has_video(sample["cache_key"]): return
    with np.load(sample["feature_path"]) as archive:
        features = archive["features"] if "features" in archive else archive[archive.files[0]]
    # L'estrattore OpenCV puo' includere il frame finale del file, mentre il
    # metadata VideoXum conta i frame sulla griglia [0, durata) a 1 fps. I file
    # BLIP attuali hanno quindi talvolta esattamente una feature in piu'. Essa
    # non appartiene ad alcun indice della timeline supervisionata e va rimossa.
    if features.ndim == 2 and features.shape[0] == sample["count"] + 1:
        features = features[:sample["count"]]
    engine.prepare_video(sample["cache_key"], count=sample["count"],
        vsum_onehot=sample["vsum_onehot"], vsum=sample["vsum"], tsum=sample["tsum"],
        frame_features=features)


def decode_frames(sample, max_frames):
    expected = sample["count"]
    if max_frames and max_frames < expected: raise ValueError("--max_frames troncherebbe la timeline")
    frames, first, next_second = [], None, 0
    with av.open(sample["video_path"]) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if frame.pts is None: continue
            ts = float(frame.pts * frame.time_base)
            if first is None: first = ts
            if ts - first + 1e-6 >= next_second:
                frames.append(frame.to_image()); next_second += 1
                if len(frames) == expected: break
    # Stessa politica di SFT/DPO e dell'estrattore BLIP: la timeline e le
    # annotazioni hanno lunghezza `expected`, quindi si ripete l'ultimo frame.
    return pad_decoded_frames(frames, {**sample, "sampled_frames": expected})


def prompt_inputs(processor, sample, args, device, frames):
    video = apply_video_time_metadata(
        {"type":"video","video":frames,"max_pixels":args.max_pixels,
         "min_pixels":args.min_pixels}, args
    )
    system_prompt = SYSTEM_PROMPTS[getattr(args, "summary_style", "concise")]
    messages = [{"role":"system","content":[{"type":"text","text":system_prompt}]},
                {"role":"user","content":[video, {"type":"text","text":sample["prompt"]}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Stesso preprocessing di sft_train_videoxum.process_video_inputs: patch 16
    # di Qwen3-VL (il default 14 ridimensionerebbe a multipli di 28) e nessun
    # secondo resize nel processor.
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        [messages], image_patch_size=16, return_video_kwargs=True, return_video_metadata=True
    )
    if video_inputs is not None:
        video_inputs, video_metadatas = zip(*video_inputs)
        video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
    else:
        video_metadatas = None
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                   video_metadata=video_metadatas, **video_kwargs,
                   cap_pixels_per_frame=True, return_tensors="pt")
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}


@torch.no_grad()
def generate(model, inputs, processor, args, training=True):
    model.config.use_cache = True; prompt_len = inputs["input_ids"].shape[1]
    kwargs = dict(max_new_tokens=args.max_completion_length, do_sample=training,
                  num_return_sequences=args.num_generations if training else 1, use_cache=True)
    if training: kwargs.update(temperature=args.temperature, top_p=args.top_p)
    # Alcune versioni di Transformers estendono attention_mask durante
    # generate(). Conserviamo quindi gli input del prompt immutati: gli stessi
    # tensor sono riutilizzati da token_logps per il backward GRPO.
    def generate_from_prompt():
        generation_inputs = {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
        return model.generate(**generation_inputs, **kwargs)
    try:
        try:
            sequences = generate_from_prompt()
            ids = [row[prompt_len:].detach() for row in sequences]
        except torch.OutOfMemoryError:
            if not training or args.num_generations == 1:
                raise
            print("[warn] rollout batch in OOM; uso generazioni singole")
            torch.cuda.empty_cache()
            kwargs["num_return_sequences"] = 1
            ids = []
            for _ in range(args.num_generations):
                sequences = generate_from_prompt()
                ids.append(sequences[0, prompt_len:].detach())
                del sequences
                torch.cuda.empty_cache()
    finally: model.config.use_cache = False
    texts = processor.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return ids, [x.strip() for x in texts]


def token_logps(model, inputs, completion):
    device, p = inputs["input_ids"].device, inputs["input_ids"].shape[1]
    completion = completion.to(device).unsqueeze(0)
    ids = torch.cat((inputs["input_ids"], completion), 1)
    attention = torch.cat((inputs["attention_mask"], torch.ones_like(completion)), 1)
    mm = {}
    for k, v in inputs.items():
        if k in ("input_ids", "attention_mask"):
            continue
        if isinstance(v, torch.Tensor) and v.ndim >= 2 and v.shape[1] == p:
            pad = torch.zeros((v.shape[0], completion.shape[1], *v.shape[2:]), dtype=v.dtype, device=v.device)
            mm[k] = torch.cat((v, pad), dim=1)
        else:
            mm[k] = v
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type=="cuda"):
        output = model(input_ids=ids, attention_mask=attention, **mm)
    logits, labels = output.logits[0, p-1:ids.shape[1]-1].float(), ids[0, p:]
    del output
    return F.log_softmax(logits, -1).gather(1, labels[:,None]).squeeze(1)

def token_mask(ids, eos, device):
    mask = torch.ones(len(ids), device=device)
    for i, token in enumerate(ids.tolist()):
        if token in eos: mask[i+1:] = 0; break
    return mask


def write_metric(handle, **values):
    handle.write(json.dumps(values, separators=(",",":"), default=str) + "\n")


def subset_indices(size, limit, seed):
    """Seleziona un sottoinsieme fisso e riproducibile; 0 usa tutti i record."""
    indices = list(range(size))
    if limit > 0 and limit < size:
        random.Random(seed).shuffle(indices)
        indices = indices[:limit]
    return indices


@torch.no_grad()
def evaluate(model, processor, dataset, reward, args, device, log, run_id, epoch):
    model.eval(); totals = {}; evaluated = skipped = 0
    validation_indices = subset_indices(len(dataset), args.val_samples, args.seed)
    for position, index in enumerate(tqdm(validation_indices, desc="validation", leave=False), 1):
        sample = dataset[index]
        # Come in train: ~1.4% dei video val ha feature BLIP con un numero di frame
        # incompatibile con il metadata (oltre il caso +1 gestito in
        # ensure_reward_cache). Senza skip un solo video fa crashare la validation.
        try:
            ensure_reward_cache(reward, sample)
            frames = decode_frames(sample, args.max_frames)
            inputs = prompt_inputs(processor, sample, args, device, frames)
            _, texts = generate(model, inputs, processor, args, False)
            row = reward.score(texts[0], sample["cache_key"])
        except Exception as error:
            print(f"[warn] skip validation {sample['video_id']}: {type(error).__name__}: {error}")
            skipped += 1; continue
        evaluated += 1
        for key, value in row.items():
            if isinstance(value, (int,float,bool)): totals[key] = totals.get(key,0.) + float(value)
        write_metric(log, run_id=run_id, record_type="sample", split="validation", epoch=epoch,
                     sample_position=position, dataset_index=index,
                     video_id=sample["video_id"], completion=texts[0], **row)
    means = {k:v/max(evaluated,1) for k,v in totals.items()}
    write_metric(log, run_id=run_id, record_type="summary", split="validation", epoch=epoch,
                 samples=evaluated, skipped=skipped, **means)
    print(f"[val] reward={means.get('total_reward',0):.4f} reward_default={means.get('total_reward_default',0):.4f} valid={means.get('valid',0)*100:.1f}% F_beta={means.get('f_beta_reward',0):.4f} summary_words={means.get('summary_words',0):.1f} summary_length={means.get('summary_length_reward',0):.4f} samples={evaluated} skipped={skipped}")
    return means


def train(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    if not args.model.is_dir(): raise FileNotFoundError(f"modello base non trovato: {args.model}")
    if args.init_from is not None and not args.init_from.is_dir():
        raise FileNotFoundError(f"checkpoint PEFT iniziale non trovato: {args.init_from}")
    if args.device.startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("CUDA non disponibile")
    if args.resume_from is not None:
        # Il LoRA da riprendere e' un delta sul modello con l'adapter iniziale
        # fuso: va ricostruito lo stesso modello, altrimenti il resume parte da
        # pesi diversi da quelli del checkpoint.
        found, saved_init = read_grpo_init(args.resume_from)
        if found and args.init_from is None:
            args.init_from = saved_init
        elif found and saved_init is not None and Path(args.init_from) != saved_init:
            raise SystemExit(
                f"[model] {args.resume_from} parte da {saved_init}, non da {args.init_from}"
            )
        elif not found and args.init_from is None:
            print(f"[WARN] {args.resume_from} non registra l'adapter di partenza: "
                  "se GRPO era partito da SFT/DPO passare lo stesso --init_from")
    # Il resume eredita l'asse temporale del proprio checkpoint, il warm-start
    # quello del checkpoint SFT/DPO di partenza.
    resolve_video_time_mode(args, args.resume_from or args.init_from, default="seconds")
    device = torch.device(args.device)
    train_set, val_set = load_datasets(args)
    reward = VideoSummaryReward(
        SentenceTransformerEncoder(args.semantic_model, args.semantic_device),
        f_beta=args.reward_f_beta,
        f_beta_aggregation=args.f_beta_aggregation,
        weights={name: getattr(args, f"w_{name}") for name in DEFAULT_REWARD_WEIGHTS},
        use_length_penalty=args.length_penalty,
        summary_length_sigma=args.summary_length_sigma,
        summary_target_words=args.summary_target_words,
        length_budget_sigma=args.length_budget_sigma,
        reference_cosine_floor=args.reference_cosine_floor,
        reference_cosine_scale=args.reference_cosine_scale,
        timeline_unit=args.timeline_unit,
    )
    print(f"[reward] pesi={reward.weights} length_penalty={reward.use_length_penalty} "
          f"f_beta={args.reward_f_beta} f_beta_aggregation={reward.f_beta_aggregation} "
          f"summary_style={args.summary_style} summary_length_sigma={reward.summary_length_sigma} "
          f"summary_target_words={reward.summary_target_words or 'per-video'} "
          f"summary_word_range={SUMMARY_WORD_RANGE} "
          f"length_budget_sigma={reward.length_budget_sigma} "
          f"reference_cosine=clamp((cos-{reward.reference_cosine_floor})"
          f"/{reward.reference_cosine_scale}) "
          f"timeline_unit={reward.timeline_unit}")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    save_dir, result_dir = args.saveroot/args.model.name, args.resultsroot/args.model.name
    save_dir.mkdir(parents=True,exist_ok=True); result_dir.mkdir(parents=True,exist_ok=True)
    log = (result_dir/f"metrics-{run_id}.jsonl").open("w",encoding="utf-8",buffering=1)
    write_metric(log, run_id=run_id, record_type="config", arguments=vars(args))
    dtype = torch.bfloat16 if device.type=="cuda" else torch.float32
    model, model_type = load_qwen_vl_model(
        args.model, torch_dtype=dtype, attn_implementation=args.attn_impl
    )
    model = model.to(device)
    print(f"[model] architecture={model_type}")
    processor = AutoProcessor.from_pretrained(args.model,local_files_only=True)
    if args.init_from is not None:
        model = PeftModel.from_pretrained(
            model, args.init_from, is_trainable=False, local_files_only=True
        ).merge_and_unload()
        print(
            f"[model] warm-start e reference KL dal checkpoint PEFT: "
            f"{args.init_from}"
        )
    else:
        print("[model] nessun checkpoint PEFT: warm-start e reference KL dal modello base")
    if args.resume_from:
        model = PeftModel.from_pretrained(model,args.resume_from,is_trainable=True,local_files_only=True)
    else:
        model = get_peft_model(model,LoraConfig(r=args.lora_r,lora_alpha=2*args.lora_r,lora_dropout=0,
            target_modules=["q_proj","v_proj"],bias="none",task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    model.enable_input_require_grads(); model.config.use_cache=False; model.print_trainable_parameters()
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=args.lr,weight_decay=.01)
    requested_train_samples = args.train_samples or args.debug_samples
    train_indices = subset_indices(len(train_set), requested_train_samples, args.seed)
    val_indices = subset_indices(len(val_set), args.val_samples or args.debug_samples, args.seed)
    n = len(train_indices)
    print(f"[data] training={n}/{len(train_set)}, validation={min(args.val_samples, len(val_set)) if args.val_samples else len(val_set)}/{len(val_set)}")
    steps=math.ceil(n/args.accum_steps)*args.epochs
    scheduler=get_cosine_schedule_with_warmup(optimizer,round(.1*steps),steps)
    eos=model.generation_config.eos_token_id or processor.tokenizer.eos_token_id
    eos={eos} if isinstance(eos,int) else set(eos); global_step=0; best=-float("inf")

    def save(path):
        path.mkdir(parents=True,exist_ok=True); model.save_pretrained(path); processor.save_pretrained(path)
        write_video_encoding(path, args); write_grpo_init(path, args.init_from)
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        order = train_indices.copy(); random.Random(args.seed+epoch).shuffle(order)
        for position,index in enumerate(tqdm(order,desc=f"epoch {epoch+1}/{args.epochs}"),1):
            sample=train_set[index]
            try:
                ensure_reward_cache(reward,sample); frames=decode_frames(sample,args.max_frames)
                inputs=prompt_inputs(processor,sample,args,device,frames); model.eval()
                completions,texts=generate(model,inputs,processor,args,True)
                rows=[reward.score(text,sample["cache_key"]) for text in texts]
                scores=torch.tensor([x["total_reward"] for x in rows]); adv=(scores-scores.mean())/(scores.std(unbiased=False)+1e-4)
                model.train(); losses=[]; kls=[]
                for i,completion in enumerate(completions):
                    pol=token_logps(model,inputs,completion)
                    with model.disable_adapter(),torch.no_grad(): ref=token_logps(model,inputs,completion)
                    mask=token_mask(completion,eos,device); ratio=ref.detach()-pol; kl=torch.exp(ratio)-ratio-1
                    loss=(-(adv[i].to(device)*pol-args.beta*kl)*mask).sum()/mask.sum().clamp(min=1)
                    (loss/len(completions)/args.accum_steps).backward(); losses.append(loss.item()); kls.append(kl.mean().item())
                write_metric(log,run_id=run_id,record_type="sample",split="train",epoch=epoch+1,
                    video_id=sample["video_id"],count=sample["count"],completions=texts,reward_components=rows,
                    total_reward=float(scores.mean()),loss=float(np.mean(losses)),kl=float(np.mean(kls)),
                    # Picco dall'inizio del job: serve a scegliere max_pixels/num_generations.
                    max_memory_allocated_gb=(torch.cuda.max_memory_allocated(device)/1024**3
                                             if device.type=="cuda" else None))
            except Exception as error:
                print(f"[warn] skip {sample['video_id']}: {type(error).__name__}: {error}")
                traceback.print_exc()
                optimizer.zero_grad(set_to_none=True); continue
            if position%args.accum_steps==0 or position==len(order):
                torch.nn.utils.clip_grad_norm_(parameters,1.0); optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                global_step+=1
                if global_step%args.save_steps==0: save(save_dir/f"checkpoint-{global_step}")
            if device.type=="cuda": torch.cuda.empty_cache()
        # Salva prima della validation: se SLURM termina il job per time limit
        # durante evaluate(), resta comunque disponibile il modello completo
        # dell'epoca appena addestrata.
        prevalidation_checkpoint = save_dir/f"checkpoint-epoch{epoch+1}-preval"
        save(prevalidation_checkpoint)
        print(f"[checkpoint] salvato prima della validation: {prevalidation_checkpoint}")
        metrics=evaluate(model,processor,val_set,reward,args,device,log,run_id,epoch+1); save(save_dir/f"checkpoint-epoch{epoch+1}")
        # Reward con pesi di default: il best resta confrontabile tra ablation.
        if metrics.get("total_reward_default",-1)>best: best=metrics["total_reward_default"]; save(save_dir/"checkpoint-best")
    save(save_dir/"final"); log.close(); print(f"[done] {save_dir/'final'}")


def positive_int(value):
    value=int(value)
    if value<=0: raise argparse.ArgumentTypeError("richiesto intero positivo")
    return value


def parse_args():
    p=argparse.ArgumentParser(description="GRPO VideoXum: selected_frames + summary")
    p.add_argument("--model",type=Path,default=DEFAULT_MODEL)
    p.add_argument("--init_from",type=Path,
        help="adapter PEFT SFT/DPO opzionale; se omesso GRPO parte dal modello base")
    p.add_argument("--resume_from",type=Path); p.add_argument("--data_json",type=Path,default=DEFAULT_DATA)
    p.add_argument("--video_dir",type=Path,default=DEFAULT_VIDEO_DIR); p.add_argument("--feature_dir",type=Path,default=DEFAULT_FEATURE_DIR)
    p.add_argument("--val_json",type=Path); p.add_argument("--val_video_dir",type=Path); p.add_argument("--val_feature_dir",type=Path)
    p.add_argument("--val_ratio",type=float,default=.05); p.add_argument("--saveroot",type=Path,default=Path("checkpoints-grpo"))
    p.add_argument("--resultsroot",type=Path,default=Path("results/grpo")); p.add_argument("--semantic_model",default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--semantic_device",default="cpu"); p.add_argument("--device",default="cuda:0")
    p.add_argument("--attn_impl",choices=("sdpa","flash_attention_2"),default="sdpa"); p.add_argument("--lr",type=float,default=1e-6)
    p.add_argument("--epochs",type=positive_int,default=2); p.add_argument("--accum_steps",type=positive_int,default=8)
    p.add_argument("--beta",type=float,default=.04); p.add_argument("--num_generations",type=positive_int,default=4)
    p.add_argument("--reward_f_beta",type=float,default=.5,
        help="beta della F-beta nella reward (1=F1; <1 pesa piu' la precisione, penalizzando selezioni ampie e imprecise)")
    p.add_argument("--f_beta_aggregation", choices=F_BETA_AGGREGATIONS, default="max",
        help="aggregazione della F-beta sui 10 annotatori: max e' l'aggregazione della "
             "metrica di validation (F1_MAX) ed e' piu' discriminante fra rollout; "
             "mean riproduce il comportamento precedente")
    for name, weight in DEFAULT_REWARD_WEIGHTS.items():
        p.add_argument(f"--w_{name}", type=float, default=weight,
            help=f"peso del termine {name} nella reward di training (default {weight}; 0 lo spegne)")
    p.add_argument("--length_penalty", action=argparse.BooleanOptionalAction, default=False,
        help="riapplica la penalita' MOLTIPLICATIVA legacy sul budget di frame, sostituita "
             "dalla componente additiva length_budget; serve solo a riprodurre i run "
             "precedenti, insieme a --w_length_budget 0")
    p.add_argument("--length_budget_sigma", type=float, default=LENGTH_BUDGET_SIGMA,
        help="sigma (log-rapporto frame selezionati/budget) del termine length_budget della reward")
    p.add_argument("--summary_style", choices=SUMMARY_STYLES, default="reference",
        help=f"reference: summary di 3-4 frasi (~{SUMMARY_WORD_RANGE[0]}-{SUMMARY_WORD_RANGE[1]} "
             "parole) come i reference VideoXum; "
             "concise: prompt a una frase dei checkpoint precedenti")
    p.add_argument("--summary_length_sigma", type=float, default=SUMMARY_LENGTH_SIGMA,
        help="sigma (log-rapporto parole) del termine summary_length della reward")
    p.add_argument("--summary_target_words", type=positive_int, default=None,
        help="bersaglio FISSO di parole per il termine summary_length, al posto della "
             "lunghezza del reference del singolo video (default: per-video). Quella "
             "e' imparabile solo in media (std 28 parole, R^2 0.078 sulla durata) e i "
             "gradienti dei video corti e lunghi si cancellano; 45 e' la media "
             "geometrica dei reference. Da abbinare a un --summary_length_sigma piu' "
             "stretto (es. 0.20), che col bersaglio per-video peggiorerebbe invece la pendenza")
    p.add_argument("--reference_cosine_floor", type=float, default=REFERENCE_COSINE_FLOOR,
        help="floor della riscalatura clamp((cos-floor)/scale) del termine reference "
             f"(default {REFERENCE_COSINE_FLOOR}, cioe' la vecchia (cos+1)/2 che satura; "
             "0.30 con scale 0.50 porta la std intra-gruppo da 0.029 a 0.114)")
    p.add_argument("--reference_cosine_scale", type=float, default=REFERENCE_COSINE_SCALE,
        help=f"scale della riscalatura del termine reference (default {REFERENCE_COSINE_SCALE})")
    # 384 e non 256: con summary da ~50 parole e molti intervalli la vecchia
    # soglia troncava il JSON (426/4000 risposte invalide sul test grpo-dpo-v2)
    p.add_argument("--max_completion_length",type=positive_int,default=384); p.add_argument("--temperature",type=float,default=1.)
    p.add_argument("--top_p",type=float,default=.95); p.add_argument("--max_frames",type=int,default=0)
    add_video_time_argument(p)
    add_pixel_arguments(p); p.add_argument("--lora_r",type=positive_int,default=16)
    p.add_argument("--save_steps",type=positive_int,default=100)
    p.add_argument("--train_samples",type=int,default=0,
        help="numero di video train da usare; 0 usa tutto il training set")
    p.add_argument("--val_samples","--eval_samples",dest="val_samples",type=int,default=200,
        help="numero di video validation da usare; 0 usa tutto il validation set")
    p.add_argument("--debug_samples",type=int,default=0,
        help="alias deprecato di --train_samples")
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args()
    if args.num_generations<2: p.error("--num_generations deve essere >=2")
    if args.train_samples < 0 or args.val_samples < 0 or args.debug_samples < 0:
        p.error("--train_samples e --val_samples devono essere >= 0")
    if args.train_samples and args.debug_samples:
        p.error("usare --train_samples oppure --debug_samples, non entrambi")
    return args


if __name__=="__main__": train(parse_args())

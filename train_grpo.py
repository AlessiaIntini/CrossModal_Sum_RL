#!/usr/bin/env python3
"""GRPO custom Qwen2.5-VL per frame discreti e summary VideoXum congiunti."""
from __future__ import annotations

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
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers import get_cosine_schedule_with_warmup
from grpo_rewards import SentenceTransformerEncoder, VideoSummaryReward

DEFAULT_MODEL = Path("/leonardo_work/IscrC_MSMU/aintini0/models/Qwen2.5-VL-3B-Instruct")
DEFAULT_DATA = Path("/leonardo_work/IscrC_MSMU/aintini0/datasets/videoxum/metadata/derived/train_videoxum_dpo.json")
DEFAULT_VIDEO_DIR = Path("/leonardo_work/IscrC_MSMU/aintini0/datasets/videoxum/train")
DEFAULT_FEATURE_DIR = Path("/leonardo_work/IscrC_MSMU/aintini0/datasets/videoxum/feat/blip")
SYSTEM_PROMPT = "You are a video summarization assistant. Select representative frames and write one concise summary."
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v")


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
    def __init__(self, records, video_dir, feature_dir, split):
        self.records, self.split = records, split
        self.videos = [resolve_video(video_dir, x["video_id"]) for x in records]
        self.features = [feature_dir / f'{x["video_id"]}.npz' for x in records]
        missing = [str(x) for x in self.videos + self.features if not x.is_file()]
        if missing: raise FileNotFoundError("File mancanti:\n  " + "\n  ".join(missing[:10]))

    def __len__(self): return len(self.records)

    def __getitem__(self, index):
        row, video, feature = self.records[index], self.videos[index], self.features[index]
        count, budget = row["count"], max(1, math.ceil(.20 * row["count"]))
        prompt = (
            f"The timeline contains {count} frames numbered 0 through {count-1}, sampled at one frame per second. "
            f"Select at most {budget} unique frames and write one final summary. Numbers are frame indices, not timestamps. "
            'Return exactly one JSON object: {"selected_frames":[[start,end],...],"summary":"..."}. '
            "Pairs are inclusive, chronological, non-overlapping and non-adjacent (merge adjacent pairs). "
            "Do not emit per-frame captions, Markdown, explanations, or additional fields."
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
    return (VideoXumGRPODataset(train, args.video_dir, args.feature_dir, "train"),
            VideoXumGRPODataset(val, args.val_video_dir or args.video_dir,
                                args.val_feature_dir or args.feature_dir, "val"))


def ensure_reward_cache(engine, sample):
    if engine.has_video(sample["cache_key"]): return
    with np.load(sample["feature_path"]) as archive:
        features = archive["features"] if "features" in archive else archive[archive.files[0]]
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
    if len(frames) != expected: raise ValueError(f"frame estratti={len(frames)}, attesi={expected}")
    return frames


def prompt_inputs(processor, sample, args, device, frames):
    messages = [{"role":"system","content":[{"type":"text","text":SYSTEM_PROMPT}]},
                {"role":"user","content":[{"type":"video","video":frames,"max_pixels":args.max_pixels},
                                              {"type":"text","text":sample["prompt"]}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    inputs = processor(text=[text], images=images, videos=videos, padding=False,
                       return_tensors="pt", cap_pixels_per_frame=True)
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}


@torch.no_grad()
def generate(model, inputs, processor, args, training=True):
    model.config.use_cache = True; prompt_len = inputs["input_ids"].shape[1]
    kwargs = dict(max_new_tokens=args.max_completion_length, do_sample=training,
                  num_return_sequences=args.num_generations if training else 1, use_cache=True)
    if training: kwargs.update(temperature=args.temperature, top_p=args.top_p)
    try:
        try:
            sequences = model.generate(**inputs, **kwargs)
            ids = [row[prompt_len:].detach() for row in sequences]
        except torch.OutOfMemoryError:
            if not training or args.num_generations == 1:
                raise
            print("[warn] rollout batch in OOM; uso generazioni singole")
            torch.cuda.empty_cache()
            kwargs["num_return_sequences"] = 1
            ids = []
            for _ in range(args.num_generations):
                sequences = model.generate(**inputs, **kwargs)
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
    mm = {k:v for k,v in inputs.items() if k not in ("input_ids", "attention_mask")}
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
    model.eval(); totals = {}
    validation_indices = subset_indices(len(dataset), args.val_samples, args.seed)
    for position, index in enumerate(tqdm(validation_indices, desc="validation", leave=False), 1):
        sample = dataset[index]; ensure_reward_cache(reward, sample)
        frames = decode_frames(sample, args.max_frames)
        inputs = prompt_inputs(processor, sample, args, device, frames)
        _, texts = generate(model, inputs, processor, args, False)
        row = reward.score(texts[0], sample["cache_key"])
        for key, value in row.items():
            if isinstance(value, (int,float,bool)): totals[key] = totals.get(key,0.) + float(value)
        write_metric(log, run_id=run_id, record_type="sample", split="validation", epoch=epoch,
                     sample_position=position, dataset_index=index,
                     video_id=sample["video_id"], completion=texts[0], **row)
    means = {k:v/max(len(validation_indices),1) for k,v in totals.items()}
    write_metric(log, run_id=run_id, record_type="summary", split="validation", epoch=epoch,
                 samples=len(validation_indices), **means)
    print(f"[val] reward={means.get('total_reward',0):.4f} valid={means.get('valid',0)*100:.1f}% F1={means.get('f1_reward',0):.4f}")
    return means


def train(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    if not args.model.is_dir(): raise FileNotFoundError(f"modello base non trovato: {args.model}")
    if args.init_from is not None and not args.init_from.is_dir():
        raise FileNotFoundError(f"checkpoint DPO non trovato: {args.init_from}")
    if args.device.startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("CUDA non disponibile")
    device = torch.device(args.device)
    train_set, val_set = load_datasets(args)
    reward = VideoSummaryReward(SentenceTransformerEncoder(args.semantic_model, args.semantic_device))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    save_dir, result_dir = args.saveroot/args.model.name, args.resultsroot/args.model.name
    save_dir.mkdir(parents=True,exist_ok=True); result_dir.mkdir(parents=True,exist_ok=True)
    log = (result_dir/f"metrics-{run_id}.jsonl").open("w",encoding="utf-8",buffering=1)
    write_metric(log, run_id=run_id, record_type="config", arguments=vars(args))
    dtype = torch.bfloat16 if device.type=="cuda" else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model,local_files_only=True,
        torch_dtype=dtype,attn_implementation=args.attn_impl).to(device)
    processor = AutoProcessor.from_pretrained(args.model,local_files_only=True)
    if args.init_from is not None:
        model = PeftModel.from_pretrained(
            model, args.init_from, is_trainable=False, local_files_only=True
        ).merge_and_unload()
        print(f"[model] warm-start e reference KL dal checkpoint DPO: {args.init_from}")
    else:
        print("[model] nessun checkpoint DPO: warm-start e reference KL dal modello base")
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
    n = len(train_indices)
    print(f"[data] training={n}/{len(train_set)}, validation={min(args.val_samples, len(val_set)) if args.val_samples else len(val_set)}/{len(val_set)}")
    steps=math.ceil(n/args.accum_steps)*args.epochs
    scheduler=get_cosine_schedule_with_warmup(optimizer,round(.1*steps),steps)
    eos=model.generation_config.eos_token_id or processor.tokenizer.eos_token_id
    eos={eos} if isinstance(eos,int) else set(eos); global_step=0; best=-float("inf")

    def save(path): path.mkdir(parents=True,exist_ok=True); model.save_pretrained(path); processor.save_pretrained(path)
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
                    video_id=sample["video_id"],completions=texts,reward_components=rows,
                    total_reward=float(scores.mean()),loss=float(np.mean(losses)),kl=float(np.mean(kls)))
            except Exception as error:
                print(f"[warn] skip {sample['video_id']}: {type(error).__name__}: {error}")
                optimizer.zero_grad(set_to_none=True); continue
            if position%args.accum_steps==0 or position==len(order):
                torch.nn.utils.clip_grad_norm_(parameters,1.0); optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                global_step+=1
                if global_step%args.save_steps==0: save(save_dir/f"checkpoint-{global_step}")
            if device.type=="cuda": torch.cuda.empty_cache()
        metrics=evaluate(model,processor,val_set,reward,args,device,log,run_id,epoch+1); save(save_dir/f"checkpoint-epoch{epoch+1}")
        if metrics.get("total_reward",-1)>best: best=metrics["total_reward"]; save(save_dir/"checkpoint-best")
    save(save_dir/"final"); log.close(); print(f"[done] {save_dir/'final'}")


def positive_int(value):
    value=int(value)
    if value<=0: raise argparse.ArgumentTypeError("richiesto intero positivo")
    return value


def parse_args():
    p=argparse.ArgumentParser(description="GRPO VideoXum: selected_frames + summary")
    p.add_argument("--model",type=Path,default=DEFAULT_MODEL)
    p.add_argument("--init_from",type=Path,
        help="checkpoint DPO opzionale; se omesso GRPO parte dal modello base")
    p.add_argument("--resume_from",type=Path); p.add_argument("--data_json",type=Path,default=DEFAULT_DATA)
    p.add_argument("--video_dir",type=Path,default=DEFAULT_VIDEO_DIR); p.add_argument("--feature_dir",type=Path,default=DEFAULT_FEATURE_DIR)
    p.add_argument("--val_json",type=Path); p.add_argument("--val_video_dir",type=Path); p.add_argument("--val_feature_dir",type=Path)
    p.add_argument("--val_ratio",type=float,default=.05); p.add_argument("--saveroot",type=Path,default=Path("checkpoints-grpo"))
    p.add_argument("--resultsroot",type=Path,default=Path("results/grpo")); p.add_argument("--semantic_model",default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--semantic_device",default="cpu"); p.add_argument("--device",default="cuda:0")
    p.add_argument("--attn_impl",choices=("sdpa","flash_attention_2"),default="sdpa"); p.add_argument("--lr",type=float,default=1e-6)
    p.add_argument("--epochs",type=positive_int,default=2); p.add_argument("--accum_steps",type=positive_int,default=8)
    p.add_argument("--beta",type=float,default=.04); p.add_argument("--num_generations",type=positive_int,default=4)
    p.add_argument("--max_completion_length",type=positive_int,default=256); p.add_argument("--temperature",type=float,default=1.)
    p.add_argument("--top_p",type=float,default=.95); p.add_argument("--max_frames",type=int,default=0)
    p.add_argument("--max_pixels",type=positive_int,default=256*28*28); p.add_argument("--lora_r",type=positive_int,default=16)
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

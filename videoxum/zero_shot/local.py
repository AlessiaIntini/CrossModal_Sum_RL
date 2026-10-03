"""Zero-shot backend for local base models: Qwen2.5-VL, Qwen3-VL, VideoLLaMA3 and InternVL3.5.

Qwen3-VL and VideoLLaMA3 receive the 1 fps frames with VideoMetadata(fps=1) and
do_sample_frames=False, so their processors neither resample the frames nor mislabel their
timestamps. InternVL uses a fixed number of tokens per frame, so at most --max_model_frames
uniformly sampled frames are shown, each as an image preceded by "Second {s}:".
"""

import random
from pathlib import Path

import numpy as np
import torch
from qwen_vl_utils import process_vision_info
from tqdm.auto import tqdm
from transformers import AutoProcessor
from transformers.video_utils import VideoMetadata

from ..inputs import move_to_device
from ..models import ZERO_SHOT_MODEL_TYPES, get_model_type, load_model
from ..video import decode_frames
from .common import (ZeroShotRun, build_task_prompt, default_run_name, load_task_data, task_system_prompt,
                     TEXT_TASKS)


def check_arguments(args):
    # Local paths, pixel budgets and the default output length of the task.
    args.model = Path(args.model)
    if args.max_model_frames is None:
        args.max_model_frames = 64
    if args.max_model_frames <= 0:
        raise SystemExit("--max_model_frames must be > 0 with the local backend")
    if args.eval_max_new_tokens is None:
        args.eval_max_new_tokens = 384 if args.task == "v2vt" else 256
    for path, description in ((args.model, "base model"), (args.test_video_dir, "test video directory")):
        if not path.is_dir():
            raise FileNotFoundError(f"{description} not found: {path}")
    if not args.test_json.is_file():
        raise FileNotFoundError(f"test metadata not found: {args.test_json}")
    if args.min_pixels > args.max_pixels:
        raise ValueError(f"--min_pixels ({args.min_pixels}) exceeds --max_pixels ({args.max_pixels})")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available: run on a GPU node")


def shown_frame_indices(count, model_type, args):
    # Seconds of the frames shown to the model: all of them, except for InternVL, which uses a fixed
    # number of tokens per frame and sees at most --max_model_frames uniformly sampled frames.
    if model_type != "internvl" or count <= args.max_model_frames:
        return list(range(count))
    indices = np.linspace(0, count - 1, args.max_model_frames).round().astype(int)
    return sorted(set(indices.tolist()))


def video_size(args, count, model_type):
    # Pixel budget of the video processor, where longest_edge is on the whole video. Qwen3-VL bounds
    # both edges on the whole video; VideoLLaMA3 divides longest_edge by the number of frames and
    # uses shortest_edge per frame.
    longest_edge = min(args.max_pixels * count, args.max_video_pixels)
    if model_type == "video_llama_3":
        return {"shortest_edge": args.min_pixels, "longest_edge": longest_edge}
    return {"shortest_edge": min(args.min_pixels * count, longest_edge), "longest_edge": longest_edge}


def build_inputs(processor, model_type, frames, prompt, args, device):
    count = len(frames)
    seconds = shown_frame_indices(count, model_type, args)
    # InternVL: every shown frame is an image preceded by "Second {s}:".
    if model_type == "internvl":
        content = []
        for second in seconds:
            content += [{"type": "text", "text": f"Second {second}: "}, {"type": "image"}]
        user_content = content + [{"type": "text", "text": prompt}]
    else:
        video = {"type": "video", "video": frames}
        if model_type == "qwen2_5_vl":
            video["max_pixels"] = args.max_pixels
            video["min_pixels"] = args.min_pixels
        user_content = [video, {"type": "text", "text": prompt}]
    messages = [
        {"role": "system", "content": [{"type": "text", "text": task_system_prompt(args)}]},
        {"role": "user", "content": user_content},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Qwen3-VL and VideoLLaMA3 get the 1 fps frames with VideoMetadata(fps=1) and
    # do_sample_frames=False, so their processors neither resample the frames nor mislabel their
    # timestamps.
    if model_type in ("qwen3_vl", "video_llama_3"):
        metadata = VideoMetadata(total_num_frames=count, fps=1.0, duration=float(count),
                                 frames_indices=list(range(count)))
        inputs = processor(text=[text], videos=[np.stack([np.asarray(frame.convert("RGB")) for frame in frames])],
                           video_metadata=[metadata], do_sample_frames=False,
                           size=video_size(args, count, model_type), padding=False, return_tensors="pt")
    elif model_type == "internvl":
        inputs = processor(text=[text], images=[frames[second].convert("RGB") for second in seconds],
                           crop_to_patches=False, padding=False, return_tensors="pt")
    else:
        # Qwen2.5-VL: the qwen_vl_utils preprocessing used in training.
        images, videos = process_vision_info(messages)
        inputs = processor(text=[text], images=images, videos=videos, padding=False, return_tensors="pt",
                           cap_pixels_per_frame=True)
    return move_to_device(inputs, device)


def generate(model, processor, inputs, args):
    # (greedy response, prompt tokens).
    prompt_length = inputs["input_ids"].shape[-1]
    generated = model.generate(**inputs, max_new_tokens=args.eval_max_new_tokens, do_sample=False, use_cache=True)
    response = processor.batch_decode(generated[:, prompt_length:], skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)[0].strip()
    return response, prompt_length


@torch.inference_mode()
def run(args):
    # Zero-shot inference of a local base model on the test subset.
    check_arguments(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    all_records, records, stop, dataset = load_task_data(args)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model_type = get_model_type(args.model, ZERO_SHOT_MODEL_TYPES)
    print(f"[data] test={len(all_records)}, subset=[{args.start_index}, {stop})")
    print(f"[task] {args.task}" + (f" summary_style={args.summary_style}" if args.task in TEXT_TASKS else ""))
    print(f"[model] base model={args.model} architecture={model_type} attention={args.attn_impl} dtype={dtype}")
    model, _ = load_model(args.model, dtype=dtype, attn_implementation=args.attn_impl,
                          supported_types=ZERO_SHOT_MODEL_TYPES)
    model = model.to(device)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model.eval()
    model.config.use_cache = True

    run_name = args.run_name or default_run_name(args.model.name, args.task)
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    first_count = records[0]["sampled_frames"]
    prompt_example = build_task_prompt(args, first_count, len(shown_frame_indices(first_count, model_type, args)))
    backend_config = dict(model_type=model_type,
                          max_model_frames=args.max_model_frames if model_type == "internvl" else None)
    with ZeroShotRun(args, output_dir, stop, len(dataset)) as zero_shot:
        zero_shot.write_config(run_name, backend_config, prompt_example)
        for local_index in tqdm(range(len(dataset)), desc="zero-shot", unit="video"):
            record, sample = records[local_index], dataset[local_index]
            position = zero_shot.position(local_index, record)
            count, response = sample["sampled_frames"], None
            try:
                frames = decode_frames(sample["video_path"], count, args.max_frames)
                # Frame i must be second i of the video, as stated in the prompt.
                if len(frames) != count:
                    raise ValueError(f"decoded {len(frames)} frames, expected {count}: "
                                     "the index-to-second mapping does not hold")
                shown_frames = len(shown_frame_indices(count, model_type, args))
                inputs = build_inputs(processor, model_type, frames, build_task_prompt(args, count, shown_frames),
                                      args, device)
                response, prompt_length = generate(model, processor, inputs, args)
                zero_shot.add(position, sample, count, response,
                              dict(shown_frames=shown_frames, prompt_tokens=prompt_length))
            except Exception as error:
                zero_shot.add_error(position, count, response, error)
            finally:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        zero_shot.finish()
    zero_shot.print_paths()

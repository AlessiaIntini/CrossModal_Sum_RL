"""Chat messages, processor inputs and generation for Qwen2.5-VL and Qwen3-VL.

Decoded frames are passed as a list, so the processor does not resample them and every
timeline index stays aligned with one second of video.
"""

import torch
from qwen_vl_utils import process_vision_info

from .prompts import SELECTION_SYSTEM_PROMPT
from .video import apply_video_time_metadata


def build_messages(sample, args, response=None, video_frames=None):
    # The system prompt, the video and the task prompt, plus the assistant response when given
    # (training targets).
    video = {
        "type": "video",
        "video": video_frames if video_frames is not None else sample["video_path"],
        "max_pixels": args.max_pixels,
        "min_pixels": args.min_pixels,
    }
    # Decoded frames are passed as a list, so the processor does not resample them; only their frame
    # rate is set for the temporal markers.
    if video_frames is not None:
        apply_video_time_metadata(video, args)
    # With a video path qwen_vl_utils decodes it: at most --max_frames frames (an even number),
    # otherwise at args.fps.
    elif args.max_frames and sample["sampled_frames"] > args.max_frames:
        nframes = max(2, min(args.max_frames, sample["sampled_frames"]))
        video["nframes"] = nframes - nframes % 2
    else:
        video["fps"] = args.fps
    messages = [
        {"role": "system", "content": [
            {"type": "text", "text": sample.get("system_prompt", SELECTION_SYSTEM_PROMPT)}
        ]},
        {"role": "user", "content": [video, {"type": "text", "text": sample["prompt"]}]},
    ]
    if response is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
    return messages


def process_video_inputs(messages, model_type):
    # Qwen3-VL uses 16-pixel patches and also needs the video metadata (frame rate, timestamps);
    # Qwen2.5-VL uses 14-pixel patches.
    if model_type == "qwen3_vl":
        images, videos, video_kwargs = process_vision_info(
            messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True,
        )
        if videos is not None:
            videos, video_metadata = zip(*videos)
            videos = list(videos)
            video_kwargs["video_metadata"] = list(video_metadata)
        return images, videos, video_kwargs
    images, videos = process_vision_info(messages, image_patch_size=14)
    return images, videos, {}


def move_to_device(inputs, device):
    # Only tensors are moved; the other processor outputs stay as they are.
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()}


def tokenize_messages(processor, messages, model_type, add_generation_prompt):
    # Processor output of a single conversation, without padding.
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    images, videos, video_kwargs = process_video_inputs(messages, model_type)
    return processor(text=[text], images=images, videos=videos, padding=False, return_tensors="pt",
                     cap_pixels_per_frame=True, **video_kwargs)


def encode_prompt(processor, sample, args, model_type, frames, device):
    # Model inputs of the prompt alone, ready for generation.
    messages = build_messages(sample, args, video_frames=frames)
    return move_to_device(tokenize_messages(processor, messages, model_type, True), device)


def prepare_completion(processor, sample, response, args, model_type, video_frames=None):
    # Inputs of prompt + response, plus the index where the response starts (used to score only the
    # response tokens).
    prompt_messages = build_messages(sample, args, video_frames=video_frames)
    full_messages = build_messages(sample, args, response, video_frames=video_frames)
    prompt_text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    full_text = processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
    images, videos, video_kwargs = process_video_inputs(prompt_messages, model_type)
    common = dict(images=images, videos=videos, padding=False, return_tensors="pt",
                  cap_pixels_per_frame=True, **video_kwargs)
    prompt_ids = processor(text=[prompt_text], **common)["input_ids"][0]
    full_inputs = processor(text=[full_text], **common)
    full_ids = full_inputs["input_ids"][0]
    prompt_len = prompt_ids.numel()
    # The response tokens start at prompt_len only if the template keeps the prompt as a prefix.
    if full_ids.numel() <= prompt_len or not torch.equal(full_ids[:prompt_len], prompt_ids):
        raise RuntimeError("the chat template does not keep the prompt as a prefix of prompt + response")
    return full_inputs, prompt_len


def decode_completions(processor, completion_ids):
    # Text of the generated token ids, stripped.
    texts = processor.batch_decode(completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return [text.strip() for text in texts]


def generate_response(model, processor, sample, args, device, frames, model_type):
    # Greedy response to the sample prompt, up to args.eval_max_new_tokens tokens.
    inputs = encode_prompt(processor, sample, args, model_type, frames, device)
    prompt_length = inputs["input_ids"].shape[-1]
    generated = model.generate(**inputs, max_new_tokens=args.eval_max_new_tokens, do_sample=False, use_cache=True)
    return decode_completions(processor, generated[:, prompt_length:])[0]


@torch.no_grad()
def generate_completions(model, inputs, processor, max_new_tokens, sampling=None):
    # Greedy completion, or sampling["num_samples"] sampled ones; returns (token ids, texts). The KV
    # cache is enabled only while generating (training keeps it off).
    model.config.use_cache = True
    prompt_len = inputs["input_ids"].shape[1]
    kwargs = dict(max_new_tokens=max_new_tokens, do_sample=sampling is not None,
                  num_return_sequences=sampling["num_samples"] if sampling else 1, use_cache=True)
    if sampling:
        kwargs.update(temperature=sampling["temperature"], top_p=sampling["top_p"])

    # Some Transformers versions extend attention_mask inside generate(): every call gets a copy,
    # so the prompt inputs stay unchanged and can be reused for other generations.
    def generate_once():
        cloned = {key: value.clone() if isinstance(value, torch.Tensor) else value
                  for key, value in inputs.items()}
        return model.generate(**cloned, **kwargs)

    try:
        try:
            ids = [row[prompt_len:].detach() for row in generate_once()]
        except torch.OutOfMemoryError:
            # A batch of sampled completions that runs out of memory is generated one sample at a
            # time; a single completion has nothing to split.
            if not sampling or sampling["num_samples"] == 1:
                raise
            print("[warning] rollout batch out of memory; generating one sample at a time")
            torch.cuda.empty_cache()
            kwargs["num_return_sequences"] = 1
            ids = []
            for _ in range(sampling["num_samples"]):
                sequences = generate_once()
                ids.append(sequences[0, prompt_len:].detach())
                del sequences
                torch.cuda.empty_cache()
    finally:
        model.config.use_cache = False
    return ids, decode_completions(processor, ids)

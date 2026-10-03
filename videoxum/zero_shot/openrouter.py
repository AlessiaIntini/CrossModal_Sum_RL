"""Zero-shot backend for API models through OpenRouter (Gemini), without a GPU.

The video is sent either as JPEG frames preceded by "Second {s}:" (--input frames) or as an
H.264 MP4 at 1 fps (--input video, base64 accepted only by the Google Vertex provider).
Successful responses are appended to api_cache.jsonl in the run directory, keyed by model,
prompt and request options, so an interrupted run can be relaunched without paying twice.
Videos rejected by the provider's content filters count as empty responses.
"""

import base64
import hashlib
import io
import json
import math
import os
import random
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import PurePosixPath

import av
import numpy as np
import requests
from PIL import Image
from tqdm.auto import tqdm

from ..jsonl import write_record
from ..prompts import add_mmss_note
from ..video import decode_frames
from .common import ZeroShotRun, build_task_prompt, default_run_name, load_task_data, task_system_prompt, TEXT_TASKS

DEFAULT_API_BASE = "https://openrouter.ai/api/v1"
INPUT_MODES = ("frames", "video")
IMAGE_DETAILS = ("low", "high", "auto")
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high")
VIDEO_PROVIDERS = ["google-vertex"]
CACHE_NAME = "api_cache.jsonl"
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}
FATAL_STATUS = {401, 402, 403}
BLOCK_CODES = {"PROHIBITED_CONTENT", "SAFETY", "BLOCKLIST", "SPII", "IMAGE_SAFETY"}


class FatalAPIError(RuntimeError):
    # Error that stops the whole run (invalid key, no credits, permissions).
    pass


class ContentBlockedError(RuntimeError):
    def __init__(self, error):
        # Video rejected by the provider's filters: an empty prediction, and the run continues.
        super().__init__(f"content blocked by the provider: {error}")
        self.provider_code = (error.get("metadata") or {}).get("provider_code")


def is_content_block(error):
    # True for provider errors that reject the content itself.
    if not isinstance(error, dict):
        return False
    metadata = error.get("metadata") or {}
    return metadata.get("error_type") == "content_policy_violation" or metadata.get("provider_code") in BLOCK_CODES


def check_arguments(args):
    # API defaults and checks of the OpenRouter backend.
    if args.max_model_frames is None:
        args.max_model_frames = 256
    if not args.test_video_dir.is_dir():
        raise FileNotFoundError(f"test video directory not found: {args.test_video_dir}")
    if not args.test_json.is_file():
        raise FileNotFoundError(f"test metadata not found: {args.test_json}")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg_quality must be between 1 and 100")


def shown_seconds(count, max_model_frames):
    # Seconds shown to the model: all, or max_model_frames uniformly sampled ones.
    if not max_model_frames or count <= max_model_frames:
        return list(range(count))
    indices = np.linspace(0, count - 1, max_model_frames).round().astype(int)
    return sorted(set(indices.tolist()))


def resize_frame(frame, max_pixels, even=False):
    # Downscale to <= max_pixels keeping the aspect ratio; H.264 yuv420p needs even sides.
    image = frame.convert("RGB")
    scale = min(1.0, math.sqrt(max_pixels / (image.width * image.height)))
    step = 2 if even else 1
    size = tuple(max(step, int(side * scale) // step * step) for side in image.size)
    return image if size == image.size else image.resize(size, Image.BICUBIC)


def data_url(data, mime):
    # Base64 data URL.
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def jpeg_bytes(image, quality):
    # JPEG encoding of an image.
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def encode_mp4(images):
    # H.264 MP4 at 1 fps: frame i has timestamp i seconds, as in the labels.
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=1)
        stream.width, stream.height = images[0].size
        stream.pix_fmt = "yuv420p"
        for second, image in enumerate(images):
            frame = av.VideoFrame.from_image(image)
            frame.pts = second
            frame.time_base = Fraction(1, 1)
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    return buffer.getvalue()


def build_user_content(frames, prompt, args):
    # --input video: one MP4 (base64 is accepted only by the Google Vertex provider).
    if args.input == "video":
        images = [resize_frame(frame, args.max_pixels, even=True) for frame in frames]
        video = {"type": "video_url", "video_url": {"url": data_url(encode_mp4(images), "video/mp4")}}
        return [video, {"type": "text", "text": prompt}]
    # --input frames: JPEG frames, each preceded by "Second {s}:", then the prompt.
    content = []
    for second in shown_seconds(len(frames), args.max_model_frames):
        image = resize_frame(frames[second], args.max_pixels)
        content += [
            {"type": "text", "text": f"Second {second}: "},
            {"type": "image_url", "image_url": dict(
                url=data_url(jpeg_bytes(image, args.jpeg_quality), "image/jpeg"),
                **({"detail": args.image_detail} if args.image_detail else {}),
            )},
        ]
    return content + [{"type": "text", "text": prompt}]


def request_options(args):
    # Request parameters shared by every video; they are part of the cache key.
    options = dict(model=args.model, max_tokens=args.max_tokens, temperature=args.temperature, seed=args.seed)
    if args.reasoning_effort:
        options["reasoning"] = {"effort": args.reasoning_effort}
    elif args.reasoning_max_tokens:
        options["reasoning"] = {"max_tokens": args.reasoning_max_tokens}
    providers = args.provider or (VIDEO_PROVIDERS if args.input == "video" else [])
    if providers:
        options["provider"] = {"only": providers}
    return options


def cache_key(options, system_prompt, prompt, video_id, args):
    # SHA-256 of everything that determines the response of a video.
    identity = dict(options, system_prompt=system_prompt, prompt=prompt, video_id=video_id, input=args.input,
                    max_pixels=args.max_pixels,
                    max_model_frames=args.max_model_frames if args.input == "frames" else None,
                    jpeg_quality=args.jpeg_quality)
    if args.image_detail and args.input == "frames":
        identity["image_detail"] = args.image_detail
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def load_cache(path):
    # Cached responses by key; truncated lines of an interrupted run are skipped.
    cache = {}
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cache[entry["key"]] = entry
    return cache


def retry_after_seconds(response):
    # Retry-After header in seconds, if any.
    try:
        return max(0.0, float(response.headers.get("Retry-After", "")))
    except ValueError:
        return None


def response_error(body):
    # Error reported inside an HTTP 200 response (downstream provider error).
    choices = body.get("choices") or []
    if "error" in body:
        return body["error"]
    if not choices:
        return {"message": "response without choices"}
    if "error" in choices[0] or choices[0].get("finish_reason") == "error":
        return choices[0].get("error") or {"message": "finish_reason=error"}
    return None


def post_chat(payload, args, api_key):
    # POST to /chat/completions with exponential backoff on transient errors.
    url = f"{args.api_base.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
               "X-Title": "VideoXum zero-shot"}
    for attempt in range(args.max_retries + 1):
        delay = None
        try:
            response = requests.post(url, headers=headers, data=payload, timeout=args.timeout)
        except requests.RequestException as error:
            problem = f"{type(error).__name__}: {error}"
        else:
            status = response.status_code
            # Authentication, credit and permission errors stop the run, unless they reject the
            # content.
            if status in FATAL_STATUS:
                try:
                    error = response.json().get("error")
                except (ValueError, AttributeError):
                    error = None
                if is_content_block(error):
                    raise ContentBlockedError(error)
                raise FatalAPIError(f"HTTP {status}: {response.text[:500]}")
            if status == 200:
                try:
                    body = response.json()
                except ValueError:
                    body = {"error": {"message": "non-JSON response"}}
                error = response_error(body)
                if error is None:
                    return body
                if is_content_block(error):
                    raise ContentBlockedError(error)
                code = error.get("code") if isinstance(error, dict) else None
                if code in FATAL_STATUS:
                    raise FatalAPIError(f"provider error: {error}")
                # Other client errors of the provider are errors of this video only.
                if isinstance(code, int) and 400 <= code < 500 and code not in RETRY_STATUS:
                    raise RuntimeError(f"provider error: {error}")
                problem = f"provider error: {error}"
            elif status in RETRY_STATUS:
                problem = f"HTTP {status}: {response.text[:300]}"
                delay = retry_after_seconds(response)
            else:
                raise RuntimeError(f"HTTP {status}: {response.text[:500]}")
        if attempt == args.max_retries:
            raise RuntimeError(f"{problem} (after {attempt + 1} attempts)")
        time.sleep(delay if delay is not None else min(60.0, 2.0 ** attempt) + random.random())


def completion_fields(body):
    # Response text and accounting (tokens, cost, provider) of a chat completion.
    choice = body["choices"][0]
    content = (choice.get("message") or {}).get("content") or ""
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    usage = body.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    return dict(response=content.strip(), finish_reason=choice.get("finish_reason"),
                native_finish_reason=choice.get("native_finish_reason"), api_model=body.get("model"),
                api_provider=body.get("provider"), prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"), reasoning_tokens=details.get("reasoning_tokens"),
                cost=usage.get("cost"))


def blocked_fields(error):
    # Completion of a rejected video: an empty response, cached as any other.
    return dict(response="", finish_reason="content_filter", native_finish_reason=error.provider_code,
                api_model=None, api_provider=None, prompt_tokens=None, completion_tokens=None,
                reasoning_tokens=None, cost=None)


def build_jobs(records, dataset, options, args):
    # Prompt and cache key of every video, without decoding it.
    system_prompt = task_system_prompt(args)
    jobs = []
    for local_index, record in enumerate(records):
        sample = dataset[local_index]
        count = sample["sampled_frames"]
        shown_frames = count if args.input == "video" else len(shown_seconds(count, args.max_model_frames))
        prompt = build_task_prompt(args, count, shown_frames)
        # Models that see an MP4 read the timestamps as MM:SS: the selection prompts ask them to
        # convert to seconds.
        if args.input == "video" and args.task in ("v2v", "v2vt"):
            prompt = add_mmss_note(prompt)
        jobs.append(dict(local_index=local_index, record=record, sample=sample, count=count,
                         shown_frames=shown_frames, prompt=prompt,
                         key=cache_key(options, system_prompt, prompt, record["video_id"], args)))
    return jobs


def query_job(job, context):
    # Cached responses are reused: an interrupted run can be relaunched without paying twice.
    cached = context["cache"].get(job["key"])
    if cached is not None:
        return dict(cached["completion"], request_mb=cached.get("request_mb"), from_cache=True)
    if context["stop"].is_set():
        raise FatalAPIError("run stopped by a previous fatal error")
    args = context["args"]
    frames = decode_frames(job["sample"]["video_path"], job["count"], args.max_frames)
    # Frame i must be second i of the video, as stated in the prompt.
    if len(frames) != job["count"]:
        raise ValueError(f"decoded {len(frames)} frames, expected {job['count']}: "
                         "the index-to-second mapping does not hold")
    messages = [{"role": "system", "content": context["system_prompt"]},
                {"role": "user", "content": build_user_content(frames, job["prompt"], args)}]
    del frames
    payload = json.dumps(dict(context["options"], messages=messages)).encode()
    try:
        completion = completion_fields(post_chat(payload, args, context["api_key"]))
    except FatalAPIError:
        context["stop"].set()
        raise
    except ContentBlockedError as error:
        tqdm.write(f"[blocked] video={job['record']['video_id']}: {error}")
        completion = blocked_fields(error)
    request_mb = len(payload) / 2 ** 20
    # New responses are appended to the cache at once.
    with context["cache_lock"]:
        write_record(context["cache_file"], key=job["key"], video_id=job["record"]["video_id"], model=args.model,
                     request_mb=request_mb, completion=completion)
    return dict(completion, request_mb=request_mb, from_cache=False)


def run_job(job, context):
    # (job, completion, error): errors of a single video do not stop the run, fatal ones do.
    try:
        return job, query_job(job, context), None
    except FatalAPIError:
        raise
    except Exception as error:
        return job, None, error


def run(args):
    # Zero-shot inference of an OpenRouter model on the test subset, reusing cached responses.
    check_arguments(args)
    random.seed(args.seed)
    all_records, records, stop, dataset = load_task_data(args)
    run_name = args.run_name or (default_run_name(PurePosixPath(args.model).name, args.task)
                                 + ("-video" if args.input == "video" else ""))
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / CACHE_NAME
    cache = load_cache(cache_path)
    options = request_options(args)
    jobs = build_jobs(records, dataset, options, args)
    to_request = sum(job["key"] not in cache for job in jobs)
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    # The API key is needed only for responses that are not cached.
    if to_request and not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    print(f"[data] test={len(all_records)}, subset=[{args.start_index}, {stop})")
    print(f"[task] {args.task}" + (f" summary_style={args.summary_style}" if args.task in TEXT_TASKS else ""))
    print(f"[model] {args.model} via OpenRouter, input={args.input}")
    print(f"[request] {json.dumps(options)}")
    print(f"[cache] {cache_path}: {len(jobs) - to_request}/{len(jobs)} responses already cached")

    finish_reasons, api_totals = Counter(), Counter()
    executor = ThreadPoolExecutor(max_workers=args.num_workers)
    backend_config = dict(model_type="openrouter", input_mode=args.input, request_options=options,
                          max_model_frames=args.max_model_frames if args.input == "frames" else None)
    with ZeroShotRun(args, output_dir, stop, len(dataset)) as zero_shot, \
            cache_path.open("a", encoding="utf-8", buffering=1) as cache_file:
        context = dict(args=args, api_key=api_key, cache=cache, cache_file=cache_file, cache_lock=threading.Lock(),
                       options=options, system_prompt=task_system_prompt(args), stop=threading.Event())
        zero_shot.write_config(run_name, backend_config, jobs[0]["prompt"])
        progress = tqdm(total=len(jobs), desc="zero-shot openrouter", unit="video")
        try:
            # Requests run in parallel; results are scored in test order.
            for job, result, error in executor.map(lambda job: run_job(job, context), jobs):
                position = zero_shot.position(job["local_index"], job["record"])
                response = result["response"] if result else None
                try:
                    if error is not None:
                        raise error
                    for name in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "cost"):
                        api_totals[name] += result[name] or 0
                    api_totals["cached_responses"] += result["from_cache"]
                    finish_reasons[str(result["finish_reason"])] += 1
                    fields = {name: result[name] for name in (
                        "prompt_tokens", "completion_tokens", "reasoning_tokens", "cost", "finish_reason",
                        "native_finish_reason", "api_model", "api_provider", "request_mb", "from_cache")}
                    zero_shot.add(position, job["sample"], job["count"], response,
                                  dict(shown_frames=job["shown_frames"], **fields))
                except Exception as sample_error:
                    zero_shot.add_error(position, job["count"], response, sample_error)
                progress.update()
                progress.set_postfix(cost=f"${api_totals['cost']:.2f}")
        except FatalAPIError as error:
            raise SystemExit(f"[fatal] {error}\nResponses received so far are in {cache_path}: "
                             "relaunch the same command to resume.") from error
        finally:
            progress.close()
            executor.shutdown(wait=True, cancel_futures=True)
        zero_shot.finish(dict(finish_reasons=dict(finish_reasons), api_cost=api_totals["cost"],
                              api_prompt_tokens=api_totals["prompt_tokens"],
                              api_completion_tokens=api_totals["completion_tokens"],
                              api_reasoning_tokens=api_totals["reasoning_tokens"],
                              cached_responses=api_totals["cached_responses"]))
    print(f"[api] cost=${api_totals['cost']:.4f} finish_reasons={dict(finish_reasons)}")
    zero_shot.print_paths()

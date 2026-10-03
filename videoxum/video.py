"""Frame decoding at 1 fps and the video encoding stored next to every checkpoint.

The encoding (temporal markers, timeline unit, pixel budget, task) changes the input seen by
the model, so later stages and inference inherit it from the checkpoint they start from.
"""

import json
from pathlib import Path

import av

VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v")
VIDEO_ENCODING_NAME = "video_encoding.json"
PROMPT_VERSIONS = {"frame": 1, "pair": 2}
DEFAULT_MAX_PIXELS = 256 * 28 * 28
DEFAULT_MIN_PIXELS = 4 * 28 * 28


def resolve_video(video_dir, video_id):
    # Metadata ids may already include the extension; otherwise every known extension is tried.
    direct = video_dir / video_id
    if direct.is_file():
        return direct
    for extension in VIDEO_EXTENSIONS:
        candidate = video_dir / f"{video_id}{extension}"
        if candidate.is_file():
            return candidate
    # Missing videos get the .mp4 path, so the error names the expected file.
    return video_dir / f"{video_id}.mp4"


def decode_frames(video_path, frame_count, max_frames=0):
    if max_frames and max_frames < frame_count:
        raise ValueError(
            f"the video needs {frame_count} frames at 1 fps but --max_frames={max_frames}; "
            "use 0 (no limit) or a value >= sampled_frames"
        )
    # One frame per second (0 s, 1 s, ...): the first frame at or after every whole second, measured
    # from the first timestamp of the stream.
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
                if len(frames) == frame_count:
                    break
    decoded = len(frames)
    if decoded == 0:
        raise ValueError(f"no decodable frame in {video_path} (expected {frame_count})")
    # Videos shorter than sampled_frames are padded with their last frame, so labels and frames stay
    # aligned.
    if decoded < frame_count:
        frames = frames + [frames[-1]] * (frame_count - decoded)
        print(f"[warning] {video_path}: decoded {decoded} frames, expected {frame_count}; "
              f"padded {frame_count - decoded}")
    return frames


def apply_video_time_metadata(video, args):
    # Without sample_fps qwen_vl_utils assumes 2 fps and the <t seconds> markers are half the real
    # time ("legacy", used by the first checkpoints); "seconds" aligns them with real seconds.
    if getattr(args, "video_time_mode", "legacy") == "seconds":
        fps = getattr(args, "fps", 1.0)
        video["sample_fps"] = fps
        video["raw_fps"] = fps
    return video


def read_video_encoding(checkpoint):
    path = Path(checkpoint) / VIDEO_ENCODING_NAME
    # Checkpoints saved before video_encoding.json existed have no encoding.
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_video_encoding(checkpoint, args):
    timeline_unit = getattr(args, "timeline_unit", "frame")
    # Everything that changes the model input is stored next to the checkpoint.
    (Path(checkpoint) / VIDEO_ENCODING_NAME).write_text(
        json.dumps(
            {
                "video_time_mode": args.video_time_mode,
                "sample_fps": getattr(args, "fps", 1.0),
                "timeline_unit": timeline_unit,
                "prompt_version": PROMPT_VERSIONS[timeline_unit],
                "max_pixels": getattr(args, "max_pixels", None),
                "min_pixels": getattr(args, "min_pixels", None),
                "task": getattr(args, "task", "v2v"),
                "summary_style": getattr(args, "summary_style", None),
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


def resolve_video_encoding(args, source=None, default="seconds", strict=True):
    # Values left to "auto" (or omitted pixel budgets) are inherited from the starting checkpoint
    # `source`; explicit values that differ from it are errors.
    encoding = read_video_encoding(source) if source is not None else None
    stored = encoding or {}

    inherited_unit = None if source is None else stored.get("timeline_unit", "frame")
    requested_unit = getattr(args, "timeline_unit", "frame")
    if requested_unit == "auto":
        args.timeline_unit = inherited_unit or "frame"
    else:
        args.timeline_unit = requested_unit
        if inherited_unit is not None and inherited_unit != requested_unit:
            raise SystemExit(f"[video] {source} uses timeline_unit={inherited_unit}, this job uses "
                             f"{requested_unit}. Use --timeline_unit auto.")
    # The pair timeline always uses legacy markers, the only ones aligned with pair indices.
    if args.timeline_unit == "pair":
        default = "legacy"

    inherited = None if source is None else stored.get("video_time_mode", "legacy")
    if args.video_time_mode == "auto":
        args.video_time_mode = inherited or default
    # A different video_time_mode is only a warning when evaluating (strict=False).
    elif inherited is not None and inherited != args.video_time_mode:
        message = (f"{source} was trained with video_time_mode={inherited}, this job uses "
                   f"{args.video_time_mode}")
        if strict:
            raise SystemExit(f"[video] {message}. Use --video_time_mode auto.")
        print(f"[warning] {message}: evaluating with a different encoding than training.")
    if args.timeline_unit == "pair" and args.video_time_mode != "legacy":
        raise SystemExit("[video] timeline_unit=pair requires video_time_mode=legacy: with "
                         f"{args.video_time_mode} the markers do not match the pair indices.")

    # Omitted pixel budgets come from the checkpoint, or from the defaults for a base model.
    for name, fallback in (("max_pixels", DEFAULT_MAX_PIXELS), ("min_pixels", DEFAULT_MIN_PIXELS)):
        if not hasattr(args, name):
            continue
        saved = stored.get(name)
        if getattr(args, name) is None:
            setattr(args, name, saved if saved is not None else fallback)
        elif saved is not None and saved != getattr(args, name):
            raise SystemExit(f"[video] {source} was trained with {name}={saved}, this job uses "
                             f"{getattr(args, name)}. Omit --{name} to inherit it.")
    if getattr(args, "min_pixels", None) is not None and args.min_pixels > args.max_pixels:
        raise SystemExit(f"--min_pixels ({args.min_pixels}) exceeds --max_pixels ({args.max_pixels})")

    print(f"[video] video_time_mode={args.video_time_mode} timeline_unit={args.timeline_unit} "
          f"max_pixels={getattr(args, 'max_pixels', None)} "
          f"min_pixels={getattr(args, 'min_pixels', None)} "
          f"(starting checkpoint={source or 'none'}, inherited encoding={encoding})")
    return args.video_time_mode


def guard_checkpoint_dir(save_dir, args, task=None):
    # Refuse to train into a directory that holds checkpoints of another encoding or task, which
    # would mix incompatible checkpoint-best candidates.
    for checkpoint in sorted(Path(save_dir).glob("checkpoint-*")):
        if not (checkpoint / "adapter_config.json").is_file():
            continue
        encoding = read_video_encoding(checkpoint) or {}
        existing_mode = encoding.get("video_time_mode", "legacy")
        existing_unit = encoding.get("timeline_unit", "frame")
        if existing_mode != args.video_time_mode or existing_unit != getattr(args, "timeline_unit", "frame"):
            raise SystemExit(
                f"[video] {save_dir} contains {checkpoint.name} with video_time_mode={existing_mode}, "
                f"timeline_unit={existing_unit}; this run uses {args.video_time_mode}, "
                f"{getattr(args, 'timeline_unit', 'frame')}. Use a dedicated --saveroot."
            )
        existing_task = encoding.get("task", "v2v")
        if task is not None and existing_task != task:
            raise SystemExit(f"[task] {save_dir} contains {checkpoint.name} with task={existing_task}; "
                             f"this run is {task}. Use a dedicated --saveroot.")

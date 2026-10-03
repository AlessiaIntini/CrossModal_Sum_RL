"""VideoXum records and datasets of the four training tasks.

Every video has `sampled_frames` frames at 1 fps and 10 binary annotations (vsum_onehot) on
those frames. Training labels (chosen/rejected) are on the model timeline instead, whose length
depends on the timeline unit.
"""

import json
import random

import numpy as np
from torch.utils.data import Dataset

from .parsing import CompletionValidationError, joint_budget, parse_and_validate_completion
from .prompts import (
    INDEX_PHRASES, SELECTION_SYSTEM_PROMPT, SYSTEM_PROMPTS, build_joint_prompt,
    build_selection_prompt, build_summary_prompt,
)
from .timeline import intervals_json, timeline_length
from .video import resolve_video


def load_items(path):
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    # Metadata files are either a JSON object keyed by video id or a list of records.
    return payload.items() if isinstance(payload, dict) else enumerate(payload)


def join_tsum(tsum):
    # Reference summary: the VideoXum caption sentences joined by spaces.
    return " ".join(str(sentence).strip() for sentence in tsum)


def check_captions(key, captions):
    # tsum must be a non-empty list of non-empty sentences.
    if not isinstance(captions, list) or not captions or any(not str(s).strip() for s in captions):
        raise ValueError(f"record {key!r}: tsum must contain non-empty sentences")


def selection_item(key, record, required=()):
    # video_id, sampled_frames and the 10 frame annotations of a record.
    if missing := ({"sampled_frames", "vsum_onehot"} | set(required)) - set(record):
        raise ValueError(f"record {key!r}: missing fields {sorted(missing)}")
    references = record["vsum_onehot"]
    frame_count = int(record["sampled_frames"])
    if len(references) != 10 or any(len(reference) != frame_count for reference in references):
        raise ValueError(f"record {key!r}: vsum_onehot must contain 10 annotations of {frame_count} frames")
    return {"video_id": str(record.get("video_id", key)), "sampled_frames": frame_count,
            "references": references}


def timeline_label(key, record, field, frame_count, timeline_unit):
    label = record[field]
    # Labels are on the model timeline: their length depends on the timeline unit.
    expected = timeline_length(frame_count, timeline_unit)
    if len(label) != expected:
        raise ValueError(f"record {key!r}: sampled_frames={frame_count}, timeline_unit={timeline_unit} "
                         f"requires len({field})={expected}, found {len(label)}")
    return label


def load_selection_records(path, with_target=True, timeline_unit="frame"):
    records = []
    for key, record in load_items(path):
        item = selection_item(key, record, ("vsum_onehot_chosen",) if with_target else ())
        # V2V records; with_target adds the chosen intervals as SFT target.
        if with_target:
            chosen = timeline_label(key, record, "vsum_onehot_chosen", item["sampled_frames"], timeline_unit)
            item["target"] = intervals_json(chosen)
        records.append(item)
    if not records:
        raise ValueError(f"no record in {path}")
    return records


def load_preference_records(path, require_pairs=True, timeline_unit="frame"):
    fields = {"vsum_onehot_chosen", "vsum_onehot_rejected"}
    records = []
    for key, record in load_items(path):
        item = selection_item(key, record, fields if require_pairs else ())
        # Validation records may have no pair (require_pairs=False), but never half of one.
        present = fields & set(record)
        if present and present != fields:
            raise ValueError(f"record {key!r}: incomplete chosen/rejected pair")
        if present:
            chosen = timeline_label(key, record, "vsum_onehot_chosen", item["sampled_frames"], timeline_unit)
            rejected = timeline_label(key, record, "vsum_onehot_rejected", item["sampled_frames"], timeline_unit)
            if chosen == rejected:
                raise ValueError(f"record {key!r}: chosen and rejected are identical")
            item["chosen"], item["rejected"] = chosen, rejected
        records.append(item)
    if not records:
        raise ValueError(f"no record in {path}")
    return records


def load_summary_records(path):
    records = []
    for key, record in load_items(path):
        if missing := {"sampled_frames", "tsum"} - set(record):
            raise ValueError(f"record {key!r}: missing fields {sorted(missing)}")
        # V2T records: the target is {"summary": reference}.
        check_captions(key, record["tsum"])
        reference = join_tsum(record["tsum"])
        records.append({
            "video_id": str(record.get("video_id", key)),
            "sampled_frames": int(record["sampled_frames"]),
            "reference_summary": reference,
            "target": json.dumps({"summary": reference}, ensure_ascii=False),
        })
    if not records:
        raise ValueError(f"no usable record in {path}")
    return records


def joint_target(chosen, tsum):
    # V2VT target: chosen intervals and reference summary in the GRPO output format.
    return json.dumps({"selected_frames": json.loads(intervals_json(chosen)), "summary": join_tsum(tsum)},
                      separators=(",", ":"), ensure_ascii=False)


def load_joint_records(path, with_target=True, timeline_unit="frame"):
    records, invalid_targets = [], 0
    for key, record in load_items(path):
        required = ("tsum", "vsum_onehot_chosen") if with_target else ("tsum",)
        item = selection_item(key, record, required)
        check_captions(key, record["tsum"])
        item["reference_summary"] = join_tsum(record["tsum"])
        if with_target:
            chosen = timeline_label(key, record, "vsum_onehot_chosen", item["sampled_frames"], timeline_unit)
            target = joint_target(chosen, record["tsum"])
            # Targets that the GRPO parser would reject (empty selection) are dropped.
            try:
                parse_and_validate_completion(target, len(chosen), joint_budget(len(chosen)))
            except CompletionValidationError:
                invalid_targets += 1
                continue
            item["target"] = target
        records.append(item)
    if not records:
        raise ValueError(f"no usable record in {path}")
    if invalid_targets:
        print(f"[data] {path.name}: dropped {invalid_targets} records whose label is not a valid V2VT target")
    return records


def load_grpo_records(path):
    records = []
    for key, row in load_items(path):
        if missing := {"sampled_frames", "vsum_onehot", "vsum", "tsum"} - set(row):
            raise ValueError(f"record {key!r}: missing fields {sorted(missing)}")
        # GRPO records keep the frame annotations, the event intervals (vsum) and the captions
        # (tsum) used by the reward.
        count, captions = int(row["sampled_frames"]), row["tsum"]
        if count <= 0 or np.asarray(row["vsum_onehot"]).shape != (10, count):
            raise ValueError(f"record {key!r}: vsum_onehot is not [10, {count}]")
        if not captions or np.asarray(row["vsum"]).shape != (len(captions), 10, 2):
            raise ValueError(f"record {key!r}: vsum and tsum are inconsistent")
        records.append(dict(video_id=str(row.get("video_id", key)), count=count,
                            vsum_onehot=row["vsum_onehot"], vsum=row["vsum"], tsum=captions))
    return records


def split_records(records, val_ratio, seed):
    if not 0 < val_ratio < 1:
        raise ValueError("--val_ratio must be between 0 and 1")
    # Deterministic train/validation split, with at least one validation record.
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    val_indices = set(indices[:max(1, round(len(records) * val_ratio))])
    return ([item for i, item in enumerate(records) if i not in val_indices],
            [item for i, item in enumerate(records) if i in val_indices])


class VideoDataset(Dataset):
    def __init__(self, records, video_dir, timeline_unit="frame"):
        self.records = records
        self.timeline_unit = timeline_unit
        # Records are paired with their video files, which must all exist.
        self.paths = [resolve_video(video_dir, item["video_id"]) for item in records]
        missing = [str(path) for path in self.paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} videos missing in {video_dir}. First files:\n  "
                                    + "\n  ".join(missing[:10]))

    def __len__(self):
        return len(self.records)

    def timeline(self, index):
        # (record, frame count, timeline length) of an item.
        record = self.records[index]
        count = record["sampled_frames"]
        return record, count, timeline_length(count, self.timeline_unit)


class SelectionDataset(VideoDataset):
    def __init__(self, records, video_dir, timeline_unit="frame", index_phrase="numbered"):
        super().__init__(records, video_dir, timeline_unit)
        self.index_phrase = index_phrase

    def __getitem__(self, index):
        record, count, timeline_count = self.timeline(index)
        sample = {
            "video_path": str(self.paths[index]),
            "sampled_frames": count,
            "timeline_unit": self.timeline_unit,
            "timeline_count": timeline_count,
            "system_prompt": SELECTION_SYSTEM_PROMPT,
            "prompt": build_selection_prompt(timeline_count, INDEX_PHRASES[self.index_phrase],
                                             self.timeline_unit),
            "references": record["references"],
        }
        # V2V samples: `target` for SFT, `chosen`/`rejected` intervals for DPO.
        if "target" in record:
            sample["target"] = record["target"]
        if "chosen" in record:
            sample["chosen"] = intervals_json(record["chosen"])
            sample["rejected"] = intervals_json(record["rejected"])
        return sample


class SummaryDataset(VideoDataset):
    def __init__(self, records, video_dir, timeline_unit="frame", summary_style="reference"):
        super().__init__(records, video_dir, timeline_unit)
        self.summary_style = summary_style

    def __getitem__(self, index):
        record, count, timeline_count = self.timeline(index)
        # V2T samples with the summary-only prompt.
        return {
            "video_id": record["video_id"],
            "video_path": str(self.paths[index]),
            "sampled_frames": count,
            "timeline_unit": self.timeline_unit,
            "timeline_count": timeline_count,
            "system_prompt": SYSTEM_PROMPTS[self.summary_style],
            "prompt": build_summary_prompt(timeline_count, self.summary_style, self.timeline_unit),
            "reference_summary": record["reference_summary"],
            "target": record["target"],
        }


class JointDataset(VideoDataset):
    def __init__(self, records, video_dir, timeline_unit="frame", summary_style="reference",
                 index_phrase="indexed"):
        super().__init__(records, video_dir, timeline_unit)
        self.summary_style, self.index_phrase = summary_style, index_phrase

    def __getitem__(self, index):
        record, count, timeline_count = self.timeline(index)
        # V2VT samples with the joint selection-and-summary prompt of GRPO.
        sample = {
            "video_id": record["video_id"],
            "video_path": str(self.paths[index]),
            "sampled_frames": count,
            "timeline_unit": self.timeline_unit,
            "timeline_count": timeline_count,
            "system_prompt": SYSTEM_PROMPTS[self.summary_style],
            "prompt": build_joint_prompt(timeline_count, index_phrase=INDEX_PHRASES[self.index_phrase],
                                         summary_style=self.summary_style, unit=self.timeline_unit),
            "references": record["references"],
            "reference_summary": record["reference_summary"],
        }
        if "target" in record:
            sample["target"] = record["target"]
        return sample


class GRPODataset(VideoDataset):
    def __init__(self, records, video_dir, feature_dir, split, summary_style="concise", timeline_unit="frame"):
        records = [{**record, "sampled_frames": record["count"]} for record in records]
        super().__init__(records, video_dir, timeline_unit)
        self.split, self.summary_style = split, summary_style
        # BLIP frame features of every video, used by the GRPO reward.
        self.features = [feature_dir / f'{record["video_id"]}.npz' for record in records]
        missing = [str(path) for path in self.features if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing files:\n  " + "\n  ".join(missing[:10]))

    def __getitem__(self, index):
        record, count, timeline_count = self.timeline(index)
        row = {key: value for key, value in record.items() if key != "sampled_frames"}
        # GRPO samples: the record, the video, its features and the joint prompt.
        return {**row, "video_path": str(self.paths[index]), "feature_path": str(self.features[index]),
                "cache_key": (self.split, record["video_id"]), "sampled_frames": count,
                "timeline_unit": self.timeline_unit, "timeline_count": timeline_count,
                "system_prompt": SYSTEM_PROMPTS[self.summary_style],
                "prompt": build_joint_prompt(timeline_count, summary_style=self.summary_style,
                                             unit=self.timeline_unit)}

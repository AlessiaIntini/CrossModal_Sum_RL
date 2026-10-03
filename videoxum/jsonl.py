"""JSONL logs shared by training, inference and evaluation."""

import json
from datetime import datetime, timezone
from pathlib import Path


def write_record(handle, **values):
    # Handles are opened line-buffered, so every record is persisted as soon as it is written.
    handle.write(json.dumps(values, separators=(",", ":"), default=str) + "\n")


def new_run_id():
    # UTC timestamp that identifies a run in file names and records.
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def mean_or_none(values):
    # None instead of a division by zero for an empty sequence.
    return sum(values) / len(values) if values else None


def read_records(path, record_type=None):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            # Without record_type every record is kept.
            if record_type is None or row.get("record_type") == record_type:
                rows.append(row)
    return rows


def last_summary(path):
    # A file can hold several summaries (e.g. resumed runs): the last one is the final result.
    summaries = read_records(path, "summary")
    return summaries[-1] if summaries else None

"""Tasks, prompts, lenient scoring and output files shared by the zero-shot backends.

Base models often wrap the JSON in code fences, add text or extra fields, so responses are
parsed leniently (first JSON value); strict_format reports whether they would also pass the
strict parser used for the trained models.
"""

import json
import re
from collections import Counter

from tqdm.auto import tqdm

from ..data import (JointDataset, SelectionDataset, SummaryDataset, load_joint_records, load_selection_records,
                    load_summary_records)
from ..jsonl import new_run_id, write_record
from ..metrics import METRIC_NAMES, TEXT_METRIC_NAMES, corpus_text_metrics, summary_metrics, videoxum_metrics
from ..parsing import (CompletionValidationError, joint_budget, parse_and_validate_completion,
                       parse_interval_response, parse_lenient_selection, parse_lenient_summary,
                       parse_summary_response)
from ..prompts import (SELECTION_SYSTEM_PROMPT, SYSTEM_PROMPTS, build_zero_shot_joint_prompt,
                       build_zero_shot_selection_prompt, build_zero_shot_summary_prompt)
from ..timeline import selected_runs

TASKS = ("v2v", "v2t", "v2vt")
TEXT_TASKS = ("v2t", "v2vt")
TASK_METRIC_NAMES = {"v2v": METRIC_NAMES, "v2t": TEXT_METRIC_NAMES, "v2vt": METRIC_NAMES + TEXT_METRIC_NAMES}
SELECTION_PROTOCOL = "VideoXum aggregation; binary generated saliency"
TEXT_PROTOCOL = ("summary vs joined tsum (split() tokenization); lenient JSON parse; "
                 "COCO metrics with evaluation/eval_tsum_coco.py")
TASK_PROTOCOLS = {"v2v": SELECTION_PROTOCOL, "v2t": TEXT_PROTOCOL, "v2vt": f"{SELECTION_PROTOCOL} | {TEXT_PROTOCOL}"}


def build_task_prompt(args, count, shown_frames):
    # Zero-shot prompt of the task for a video of `count` frames, `shown_frames` of which are shown.
    if args.task == "v2t":
        return build_zero_shot_summary_prompt(count, args.summary_style, shown_frames)
    if args.task == "v2vt":
        return build_zero_shot_joint_prompt(count, args.selection_ratio, shown_frames, args.summary_style)
    return build_zero_shot_selection_prompt(count, args.selection_ratio, shown_frames)


def task_system_prompt(args):
    # System prompt of the selection models for V2V, of the summary models for V2T and V2VT.
    return SYSTEM_PROMPTS[args.summary_style] if args.task in TEXT_TASKS else SELECTION_SYSTEM_PROMPT


def load_task_data(args):
    # (all test records, selected records, stop index, dataset) of the task.
    if args.task == "v2t":
        all_records = load_summary_records(args.test_json)
    elif args.task == "v2vt":
        all_records = load_joint_records(args.test_json, with_target=False)
    else:
        all_records = load_selection_records(args.test_json, with_target=False)
    if args.start_index >= len(all_records):
        raise ValueError(f"--start_index={args.start_index} is outside the test set ({len(all_records)} videos)")
    # num_samples=0 means all the remaining videos.
    stop = len(all_records) if args.num_samples == 0 else min(len(all_records), args.start_index + args.num_samples)
    records = all_records[args.start_index:stop]
    if args.task == "v2t":
        dataset = SummaryDataset(records, args.test_video_dir, summary_style=args.summary_style)
    elif args.task == "v2vt":
        dataset = JointDataset(records, args.test_video_dir, summary_style=args.summary_style)
    else:
        dataset = SelectionDataset(records, args.test_video_dir)
    return all_records, records, stop, dataset


def default_run_name(model_name, task):
    # zero-shot[-<task>]-<model>; V2V keeps the shorter name of the first runs.
    prefix = "zero-shot" if task == "v2v" else f"zero-shot-{task}"
    return re.sub(r"[^A-Za-z0-9._-]+", "-", f"{prefix}-{model_name}").strip("-")


def is_strict_joint(response, count):
    # True if the response also passes the strict parser of the GRPO reward.
    try:
        parse_and_validate_completion(response, count, joint_budget(count))
    except (CompletionValidationError, json.JSONDecodeError, ValueError):
        return False
    return True


def is_strict_summary(response):
    # True if the response also passes the strict V2T parser of the trained models.
    try:
        parse_summary_response(response)
    except (json.JSONDecodeError, ValueError):
        return False
    return True


def selection_fields(prediction):
    # Prediction fields of a binary selection.
    return dict(
        predicted_intervals=[list(run) for run in selected_runs(prediction)],
        selected_frame_indices=[index for index, selected in enumerate(prediction) if selected],
        selected_frames=sum(prediction),
        selection_ratio=sum(prediction) / len(prediction),
    )


def score_v2v(response, sample, count):
    # Selection responses use the strict interval parser; an invalid response counts as an empty
    # selection.
    parse_error = None
    try:
        prediction = parse_interval_response(response, count)
    except (json.JSONDecodeError, ValueError) as error:
        prediction, parse_error = [0] * count, str(error)
    return parse_error, selection_fields(prediction), videoxum_metrics(prediction, sample["references"])


def score_v2t(response, sample):
    # Summary responses are parsed leniently; an invalid response counts as an empty summary.
    parse_error = None
    try:
        summary = parse_lenient_summary(response)
    except (json.JSONDecodeError, ValueError) as error:
        summary, parse_error = "", str(error)
    fields = dict(summary=summary, strict_format=is_strict_summary(response))
    return parse_error, fields, summary_metrics(summary, sample["reference_summary"])


def score_v2vt(response, sample, count):
    # Selection and summary are scored separately: an invalid field counts as an empty prediction
    # only for its own metrics.
    errors = {}
    try:
        prediction = parse_lenient_selection(response, count)
    except (json.JSONDecodeError, ValueError) as error:
        prediction, errors["selection"] = [0] * count, str(error)
    try:
        summary = parse_lenient_summary(response)
    except (json.JSONDecodeError, ValueError) as error:
        summary, errors["summary"] = "", str(error)
    parse_error = "; ".join(f"{name}: {error}" for name, error in errors.items()) or None
    fields = dict(selection_fields(prediction), summary=summary, selection_valid="selection" not in errors,
                  summary_valid="summary" not in errors, strict_format=is_strict_joint(response, count))
    metrics = dict(videoxum_metrics(prediction, sample["references"]),
                   **summary_metrics(summary, sample["reference_summary"]))
    return parse_error, fields, metrics


def score_task(args, response, sample, count):
    # (parse error, prediction fields, metrics) of a response to the task prompt.
    if args.task == "v2t":
        return score_v2t(response, sample)
    if args.task == "v2vt":
        return score_v2vt(response, sample, count)
    return score_v2v(response, sample, count)


class ZeroShotRun:
    def __init__(self, args, output_dir, stop, samples):
        # Predictions and metrics JSONL files of a zero-shot run, with per-video scoring.
        self.args, self.samples = args, samples
        self.text = args.task in TEXT_TASKS
        self.run_id = new_run_id()
        suffix = f"test-{args.start_index}-{stop}-{self.run_id}.jsonl"
        self.predictions_path = output_dir / f"predictions-{suffix}"
        self.metrics_path = output_dir / f"metrics-{suffix}"
        self.metric_values = {name: [] for name in TASK_METRIC_NAMES[args.task]}
        self.candidates, self.references = [], []
        self.invalid_predictions = self.strict_format_valid = self.processing_errors = 0
        self.invalid_parts = Counter()

    def __enter__(self):
        self.predictions_file = self.predictions_path.open("w", encoding="utf-8", buffering=1)
        self.metrics_file = self.metrics_path.open("w", encoding="utf-8", buffering=1)
        return self

    def __exit__(self, *exc):
        self.predictions_file.close()
        self.metrics_file.close()

    def position(self, local_index, record):
        # Fields that identify a test video in every record.
        return dict(run_id=self.run_id, split="test", sample_position=local_index + 1,
                    dataset_index=self.args.start_index + local_index, video_id=record["video_id"])

    def write_config(self, run_name, backend_config, prompt_example):
        # Config records; the prompt example is the one of the first video of the subset.
        common = dict(run_id=self.run_id, record_type="config", split="test", mode="zero_shot",
                      task=self.args.task, summary_style=self.args.summary_style if self.text else None,
                      run_name=run_name, model=self.args.model, **backend_config, test_json=self.args.test_json,
                      start_index=self.args.start_index, stop_index=self.args.start_index + self.samples,
                      samples=self.samples, system_prompt=task_system_prompt(self.args), prompt_example=prompt_example)
        write_record(self.predictions_file, **common, test_video_dir=self.args.test_video_dir, arguments=vars(self.args))
        write_record(self.metrics_file, **common, metric_protocol=TASK_PROTOCOLS[self.args.task])

    def add(self, position, sample, count, response, fields):
        # Score a response and write its prediction and metrics records.
        parse_error, prediction_fields, sample_metrics = score_task(self.args, response, sample, count)
        self.invalid_predictions += parse_error is not None
        if "selection_valid" in prediction_fields:
            self.invalid_parts["invalid_selections"] += not prediction_fields["selection_valid"]
            self.invalid_parts["invalid_summaries"] += not prediction_fields["summary_valid"]
        if self.text:
            self.strict_format_valid += prediction_fields["strict_format"]
            self.candidates.append(prediction_fields["summary"])
            self.references.append(sample["reference_summary"])
        for name, value in sample_metrics.items():
            self.metric_values[name].append(value)
        write_record(self.predictions_file, record_type="sample", **position, sampled_frames=count, **fields,
                     generated_response=response, prediction_valid=parse_error is None, parse_error=parse_error,
                     **prediction_fields)
        write_record(self.metrics_file, record_type="sample", **position, prediction_valid=parse_error is None,
                     **sample_metrics)

    def add_error(self, position, count, response, error):
        # A failing video gets a sample_error record in both files.
        self.processing_errors += 1
        for handle in (self.predictions_file, self.metrics_file):
            write_record(handle, record_type="sample_error", **position, sampled_frames=count,
                         generated_response=response, error_type=type(error).__name__, error=str(error))
        tqdm.write(f"[warning] test[{position['dataset_index']}] video={position['video_id']}: "
                   f"{type(error).__name__}: {error}")

    def finish(self, extra_summary=None):
        # Summary record: mean metrics, plus corpus text metrics for the text tasks.
        values = {name: (sum(items) / len(items) if items else None) for name, items in self.metric_values.items()}
        if self.text:
            values.update(corpus_text_metrics(self.candidates, self.references))
            values["strict_format_valid"] = self.strict_format_valid
        first_metric = TASK_METRIC_NAMES[self.args.task][0]
        write_record(self.metrics_file, run_id=self.run_id, record_type="summary", split="test", mode="zero_shot",
                     task=self.args.task, samples_requested=self.samples,
                     samples_evaluated=len(self.metric_values[first_metric]),
                     invalid_predictions=self.invalid_predictions, **self.invalid_parts,
                     processing_errors=self.processing_errors, metric_protocol=TASK_PROTOCOLS[self.args.task],
                     **(extra_summary or {}), **values)
        valid = self.samples - self.invalid_predictions - self.processing_errors
        print("[test] " + " ".join(f"{name.upper()}={value:.4f}" for name, value in values.items()
                                   if value is not None) + f" valid={valid}/{self.samples}")
        return values

    def print_paths(self):
        print(f"[done] predictions: {self.predictions_path}")
        print(f"[done] metrics: {self.metrics_path}")

#!/usr/bin/env python3
"""SFT V2VT VideoXum su TRL: selezione dei frame e summary testuale insieme.

sft_train_videoxum_trl.py insegna solo la selezione (V2V): il target e' l'array
di intervalli costruito da vsum_onehot_chosen. GRPO ottimizza invece il task
congiunto, cioe' l'oggetto {"selected_frames":[[a,b],...],"summary":"..."} del
prompt di train_grpo.py. Qui il target e' quell'oggetto: intervalli dal label
consensus e summary di riferimento VideoXum (tsum unito in una stringa, lo
stesso reference usato dalla reward GRPO e da evaluation/eval_coco.py). Il
checkpoint prodotto e' quindi un warm-start che GRPO puo' usare (--init_from)
senza dover prima imparare formato e scrittura del summary.

Loop, collator, trainer, checkpoint e resume restano quelli di
sft_train_videoxum_trl.py: cambiano solo dati/prompt/target e la validation.
"""
from __future__ import annotations
import argparse, json, math, os, sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from tqdm.auto import tqdm

import sft_train_videoxum_trl
from grpo_rewards import (CompletionValidationError, expand_pair_completion,
                          parse_and_validate_completion)
from sft_train_videoxum import (METRIC_NAMES, VideoXumSFTDataset, collapse_metrics,
                                decode_video_frames, format_collapse_summary, format_response,
                                generate_interval_response as generate_response, positive_int,
                                read_video_encoding, split_records, timeline_length,
                                validation_limit, videoxum_metrics, write_metric)
from sft_train_videoxum_trl import SFTSchedule, save_named
from train_grpo import SUMMARY_STYLES, SYSTEM_PROMPTS, build_grpo_prompt

# Metriche testuali per-sample del summary generato contro " ".join(tsum).
TEXT_METRIC_NAMES = ("rouge1", "rougeL", "bleu4_sentence", "summary_words")
# CIDEr e BLEU di corpus restano fuori: sono corpus-level e non confrontabili fra
# validation di dimensione diversa (vedi evaluation/eval.compute_cider_score).
BEST_METRIC_CHOICES = METRIC_NAMES + ("rouge1", "rougeL", "bleu4_sentence", "v2vt")
INDEX_PHRASES = {"indexed": " indexed from", "numbered": ", numbered from"}
_TEXT_TOOLS = None


def text_tools():
    """Import pigro: evaluation/eval.py carica rouge_score, nltk e sentence_transformers."""
    global _TEXT_TOOLS
    if _TEXT_TOOLS is None:
        from rouge_score import rouge_scorer
        from evaluation.eval import (compute_cider_score, compute_corpus_bleu,
                                     compute_sentence_bleu_scores)
        _TEXT_TOOLS = SimpleNamespace(
            # Stessa configurazione di evaluation/eval.py: gli stessi summary danno
            # qui e nella valutazione offline lo stesso ROUGE.
            rouge=rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True),
            sentence_bleu=compute_sentence_bleu_scores,
            corpus_bleu=compute_corpus_bleu, cider=compute_cider_score,
        )
    return _TEXT_TOOLS


def finite(value):
    """None al posto di NaN/inf, cosi' il JSONL delle metriche resta JSON valido."""
    value = float(value)
    return value if math.isfinite(value) else None


def max_selected(timeline_count):
    """Budget passato al validatore, lo stesso di infer_grpo_videoxum.py."""
    return max(1, math.floor(0.50 * timeline_count))


def build_v2vt_target(chosen, tsum):
    """Target congiunto nel formato che la reward GRPO accetta.

    Gli intervalli sono quelli della SFT V2V (format_response); il summary e' il
    reference VideoXum, cioe' le frasi di tsum unite, come in
    grpo_rewards.VideoSummaryReward e in evaluation/eval_coco.py."""
    intervals = json.loads(format_response(chosen))
    summary = " ".join(str(sentence).strip() for sentence in tsum)
    return json.dumps({"selected_frames": intervals, "summary": summary},
                      separators=(",", ":"), ensure_ascii=False)


def load_v2vt_records(path, require_chosen=True, timeline_unit="frame"):
    """Come sft_train_videoxum.load_records, ma carica anche tsum e compone il
    target V2VT; una sola passata sul JSON (i file consensus sono da 170 MB)."""
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    items = payload.items() if isinstance(payload, dict) else enumerate(payload)
    records, invalid_targets = [], 0
    for key, record in items:
        required = {"sampled_frames", "vsum_onehot", "tsum"}
        if require_chosen:
            required |= {"vsum_onehot_chosen"}
        if missing := required - set(record):
            raise ValueError(f"Record {key!r}: campi mancanti {sorted(missing)}")
        references = record["vsum_onehot"]
        frame_count = int(record["sampled_frames"])
        if len(references) != 10 or any(len(reference) != frame_count for reference in references):
            raise ValueError(
                f"Record {key!r}: vsum_onehot deve contenere 10 annotazioni da "
                f"{frame_count} frame"
            )
        captions = record["tsum"]
        if not isinstance(captions, list) or not captions or any(
            not str(sentence).strip() for sentence in captions
        ):
            raise ValueError(f"Record {key!r}: tsum deve contenere frasi non vuote")
        item = {
            "video_id": str(record.get("video_id", key)),
            "sampled_frames": frame_count,
            "references": references,
            "reference_summary": " ".join(str(sentence).strip() for sentence in captions),
        }
        if require_chosen:
            chosen = record["vsum_onehot_chosen"]
            expected = timeline_length(frame_count, timeline_unit)
            if len(chosen) != expected:
                raise ValueError(
                    f"Record {key!r}: sampled_frames={frame_count}, "
                    f"timeline_unit={timeline_unit} richiede len(chosen)={expected}, "
                    f"trovato {len(chosen)}"
                )
            target = build_v2vt_target(chosen, captions)
            try:
                # Un target che la reward GRPO rifiuterebbe (selezione vuota)
                # insegnerebbe al modello un output che poi vale reward zero.
                parse_and_validate_completion(target, expected, max_selected(expected))
            except CompletionValidationError:
                invalid_targets += 1
                continue
            item["target"] = target
        records.append(item)
    if not records:
        raise ValueError(f"Nessun record utilizzabile in {path}")
    if invalid_targets:
        print(f"[data] {path.name}: {invalid_targets} record scartati "
              "(label consensus non rappresentabile nel formato V2VT)")
    return records


class VideoXumV2VTDataset(VideoXumSFTDataset):
    """Prompt e system prompt di train_grpo.py, target congiunto frame+summary."""

    def __init__(self, records, video_dir, timeline_unit="frame", summary_style="reference",
                 index_phrase="indexed"):
        super().__init__(records, video_dir, timeline_unit)
        self.summary_style, self.index_phrase = summary_style, index_phrase

    def __getitem__(self, index):
        record = self.records[index]
        count = record["sampled_frames"]
        timeline_count = timeline_length(count, self.timeline_unit)
        sample = {
            "video_id": record["video_id"],
            "video_path": str(self.paths[index]),
            "sampled_frames": count,
            "timeline_unit": self.timeline_unit,
            "timeline_count": timeline_count,
            "system_prompt": SYSTEM_PROMPTS[self.summary_style],
            "prompt": build_grpo_prompt(
                timeline_count, index_phrase=INDEX_PHRASES[self.index_phrase],
                summary_style=self.summary_style, unit=self.timeline_unit,
            ),
            "references": record["references"],
            "reference_summary": record["reference_summary"],
        }
        if "target" in record:
            sample["target"] = record["target"]
        return sample


def load_v2vt_datasets(args):
    """Come sft_train_videoxum.load_datasets, con i record e il prompt V2VT."""
    records = load_v2vt_records(args.data_json, require_chosen=True,
                                timeline_unit=args.timeline_unit)
    if args.val_json:
        train_records = records
        val_records = load_v2vt_records(args.val_json, require_chosen=False)
    else:
        train_records, val_records = split_records(records, args.val_ratio, args.seed)
    val_video_dir = args.val_video_dir or args.video_dir
    build = lambda items, video_dir: VideoXumV2VTDataset(
        items, video_dir, args.timeline_unit, args.summary_style, args.index_phrase)
    train, val = build(train_records, args.video_dir), build(val_records, val_video_dir)
    print(f"[data] train={len(train)}, val={len(val)}, val_video_dir={val_video_dir}")
    return train, val


def parse_v2vt_completion(response, sample):
    """Completion GRPO -> selezione sugli N frame originali e summary."""
    timeline_count = sample["timeline_count"]
    parsed = parse_and_validate_completion(response, timeline_count, max_selected(timeline_count))
    if sample["timeline_unit"] == "pair":
        parsed = expand_pair_completion(parsed, sample["sampled_frames"])
    return parsed.pred_mask.int().tolist(), parsed.summary


def summary_metrics(candidate, reference):
    """Metriche per-sample del summary; un summary assente vale 0, come nello
    script ufficiale replicato da evaluation/eval_coco.py."""
    tools = text_tools()
    scores = tools.rouge.score(reference, candidate)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
        "bleu4_sentence": tools.sentence_bleu([candidate], [reference])[0],
        "summary_words": float(len(candidate.split())),
    }


def best_score(metric_values, args):
    """Punteggio di checkpoint-best, sempre sui primi --eval_steps_samples video
    (come sft_train_videoxum.selection_score), cosi' validation periodica e di
    fine epoca restano confrontabili."""
    if args.best_metric == "v2vt":
        selection = "kappa" if args.timeline_unit == "pair" else "f1_mean"
        values = [(a + b) / 2 for a, b in zip(metric_values[selection], metric_values["rougeL"])]
    else:
        values = metric_values[args.best_metric]
    if args.eval_steps:
        values = values[:args.eval_steps_samples]
    return sum(values) / len(values)


def format_v2vt_summary(metrics, limit, invalid_predictions):
    # cider e' None quando la validation ha meno di 2 video (metrica corpus-level).
    cider = "n/d" if metrics["cider"] is None else f"{metrics['cider']:.4f}"
    return (
        f"[val] VideoXum F1_MEAN={metrics['f1_mean']:.4f} "
        f"Kappa={metrics['kappa']:.4f} Kendall={metrics['kendall']:.4f} | "
        f"ROUGE-L={metrics['rougeL']:.4f} BLEU@4={metrics['bleu4_sentence']:.4f} "
        f"CIDEr={cider} words={metrics['summary_words']:.1f} "
        f"valid_predictions={limit - invalid_predictions}/{limit}"
    )


@torch.no_grad()
def evaluate(model, processor, dataset, args, device, metrics_file, run_id, epoch,
             model_type, limit=None, optimizer_step=None):
    """Validation generativa del task congiunto: stesse metriche di selezione
    della SFT V2V piu' le metriche testuali del summary."""
    model.eval()
    limit = validation_limit(dataset, args.eval_samples if limit is None else limit)
    invalid_predictions = 0
    metric_values = {name: [] for name in METRIC_NAMES + TEXT_METRIC_NAMES}
    responses, predictions, references_list = [], [], []
    candidates, reference_summaries = [], []
    for index in tqdm(range(limit), desc="validation", unit="video", leave=False):
        sample = dataset[index]
        frames = decode_video_frames(sample, args)
        response = generate_response(model, processor, sample, args, device, frames, model_type)
        parse_error = None
        try:
            prediction, summary = parse_v2vt_completion(response, sample)
        except (CompletionValidationError, json.JSONDecodeError, ValueError) as error:
            prediction, summary = [0] * sample["sampled_frames"], ""
            parse_error = str(error)
            invalid_predictions += 1
        sample_metrics = videoxum_metrics(prediction, sample["references"])
        sample_metrics.update(summary_metrics(summary, sample["reference_summary"]))
        for name, value in sample_metrics.items():
            metric_values[name].append(value)
        responses.append(response)
        predictions.append(prediction)
        references_list.append(sample["references"])
        candidates.append(summary)
        reference_summaries.append(sample["reference_summary"])

        write_metric(
            metrics_file, run_id=run_id, record_type="sample", split="validation",
            epoch=epoch, optimizer_step=optimizer_step, sample_position=index + 1,
            dataset_index=index, video_id=sample["video_id"],
            generated_response=response, model_type=model_type,
            prediction_valid=parse_error is None, parse_error=parse_error,
            summary=summary, selected_frames=sum(prediction),
            selection_ratio=sum(prediction) / len(prediction), **sample_metrics,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    tools = text_tools()
    summary_values = {name: sum(values) / len(values) for name, values in metric_values.items()}
    summary_values["selection_score"] = best_score(metric_values, args)
    corpus_bleu = tools.corpus_bleu(candidates, reference_summaries)
    cider, _ = tools.cider(candidates, reference_summaries)
    summary_values.update({name: finite(value) for name, value in corpus_bleu.items()})
    summary_values["cider"] = finite(cider)
    collapse = collapse_metrics(responses, predictions, references_list)
    # Un modello che impara un summary template lo ripete identico su video
    # diversi: e' il corrispettivo testuale di distinct_output_ratio.
    collapse["distinct_summary_ratio"] = (
        len({" ".join(candidate.split()) for candidate in candidates}) / len(candidates)
    )
    print(format_v2vt_summary(summary_values, limit, invalid_predictions))
    print(f"[val] collasso: {format_collapse_summary(collapse)} "
          f"distinct_summary_ratio={collapse['distinct_summary_ratio']:.3f}")
    write_metric(
        metrics_file, run_id=run_id, record_type="summary", split="validation",
        epoch=epoch, optimizer_step=optimizer_step, samples=limit,
        invalid_predictions=invalid_predictions,
        metric_protocol="VideoXum aggregation; binary generated saliency; "
                        "summary vs joined tsum (split() tokenization)",
        video_time_mode=args.video_time_mode, timeline_unit=args.timeline_unit,
        summary_style=args.summary_style, best_metric=args.best_metric,
        collapse=collapse, **summary_values,
    )
    return summary_values


class V2VTSchedule(SFTSchedule):
    """SFTSchedule con la validation V2VT; il resto (budget, checkpoint, resume)
    e' quello della SFT V2V."""

    def _validate(self, state, epoch, limit=None):
        trainer = self.trainer
        metrics = evaluate(trainer.model, trainer.processing_class, self.val_dataset, self.args,
                           trainer.accelerator.device, self.metrics_file, self.run_id, epoch,
                           self.model_type, limit=limit, optimizer_step=state.global_step)
        trainer.model.train()
        if state.best_metric is None or metrics["selection_score"] > state.best_metric:
            state.best_metric = metrics["selection_score"]
            save_named(trainer, self.save_dir / "checkpoint-best", self.args)
            print(f"[best] checkpoint-best aggiornato: {self.args.best_metric}={state.best_metric:.4f}")
        return metrics


def guard_v2vt_dir(save_dir):
    """Rifiuta una directory con checkpoint V2V: sovrascriverebbe checkpoint-best
    di una run con un altro task (vedi sft_train_videoxum.guard_checkpoint_dir)."""
    for checkpoint in sorted(Path(save_dir).glob("checkpoint-*")):
        if not (checkpoint / "adapter_config.json").is_file():
            continue
        task = (read_video_encoding(checkpoint) or {}).get("task", "v2v")
        if task != "v2vt":
            raise SystemExit(
                f"[task] {save_dir} contiene {checkpoint.name} con task={task}; questo "
                "run e' v2vt. Usare un --saveroot dedicato."
            )


def train(args):
    if not args.resume_from:
        guard_v2vt_dir(args.saveroot / args.model.name)
    print(f"[v2vt] summary_style={args.summary_style} index_phrase={args.index_phrase} "
          f"eval_max_new_tokens={args.eval_max_new_tokens}")
    sft_train_videoxum_trl.train(args, datasets_loader=load_v2vt_datasets,
                                 schedule_cls=V2VTSchedule)


def parse_args():
    """Flag propri della V2VT, poi la CLI di sft_train_videoxum_trl.py sul resto."""
    parser = argparse.ArgumentParser(add_help=False, description="Opzioni V2VT di "
                                     "sft_train_videoxum_v2vt_trl.py (le altre sono quelle di "
                                     "sft_train_videoxum_trl.py e sft_train_videoxum.py)")
    parser.add_argument("--summary_style", choices=SUMMARY_STYLES, default="reference",
        help="stile del summary chiesto nel prompt (train_grpo.SUMMARY_STYLES); deve "
             "coincidere con quello di GRPO e dell'inferenza")
    parser.add_argument("--index_phrase", choices=tuple(INDEX_PHRASES), default="indexed",
        help="clausola della timeline nel prompt: indexed come build_grpo_prompt e "
             "infer_grpo_videoxum.py, numbered come la SFT V2V; ignorata con "
             "--timeline_unit pair, dove il testo e' unico")
    parser.add_argument("--best_metric", choices=BEST_METRIC_CHOICES, default=None,
        help="metrica di validation per checkpoint-best; v2vt e' la media fra la "
             "metrica di selezione di default (kappa con --timeline_unit pair, "
             "altrimenti f1_mean) e rougeL. Default: come la SFT V2V")
    parser.add_argument("--eval_max_new_tokens", type=positive_int, default=384,
        help="token massimi in validation: il target V2VT contiene anche il summary "
             "(default uguale a --max_completion_length di GRPO)")
    if {"-h", "--help"} & set(sys.argv[1:]): parser.print_help(); print()
    v2vt_args, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    args = sft_train_videoxum_trl.parse_args()
    vars(args).update(vars(v2vt_args))
    # Finisce in video_encoding.json: distingue i checkpoint V2VT da quelli V2V.
    args.task = "v2vt"
    if args.diversity_check_samples or args.video_dependence_samples:
        raise SystemExit("--diversity_check_samples e --video_dependence_samples non sono "
                         "supportati: leggono l'output V2V. Il collasso e' gia' misurato a "
                         "ogni validation (collapse, distinct_summary_ratio)")
    return args


if __name__ == "__main__": train(parse_args())

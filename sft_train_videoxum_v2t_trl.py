#!/usr/bin/env python3
"""SFT V2T VideoXum su TRL: dal video al solo summary testuale.

sft_train_videoxum_trl.py insegna la selezione dei frame (V2V) e
sft_train_videoxum_v2vt_trl.py selezione e summary insieme (V2VT). Qui il
modello vede lo stesso video ma il target e' soltanto {"summary":"..."}, con il
summary di riferimento VideoXum (tsum unito in una stringa, lo stesso reference
di grpo_rewards e di evaluation/eval_coco.py). Prompt e system prompt sono quelli
di infer_grpo_videoxum.py --summary_only, cosi' il checkpoint si valuta sul test
con lo stesso script (SUMMARY_ONLY=1 in infer_grpo_test_consensus_hpc.sh).

Serve come ablation del task congiunto (V2T contro V2VT a parita' di dati) e come
baseline testuale; non e' un warm-start per il GRPO congiunto, che chiede anche
selected_frames. Loop, collator, checkpoint e resume sono quelli di
sft_train_videoxum_trl.py: cambiano dati/prompt/target e la validation.
"""
from __future__ import annotations
import argparse, json, math, os, sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from tqdm.auto import tqdm

import sft_train_videoxum_trl
from sft_train_videoxum import (VideoXumSFTDataset, decode_video_frames,
                                generate_interval_response as generate_response, positive_int,
                                read_video_encoding, split_records, timeline_length,
                                validation_limit, write_metric)
from sft_train_videoxum_trl import SFTSchedule, save_named
from train_grpo import SUMMARY_STYLES, SYSTEM_PROMPTS, build_summary_only_prompt

# Metriche testuali per-sample del summary generato contro " ".join(tsum).
TEXT_METRIC_NAMES = ("rouge1", "rougeL", "bleu4_sentence", "summary_words")
# CIDEr e BLEU di corpus restano fuori: sono corpus-level e non confrontabili fra
# validation di dimensione diversa (vedi evaluation/eval.compute_cider_score).
BEST_METRIC_CHOICES = ("rouge1", "rougeL", "bleu4_sentence")
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


def join_tsum(tsum):
    return " ".join(str(sentence).strip() for sentence in tsum)


def build_v2t_target(tsum):
    """Target nel formato che infer_grpo_videoxum.py --summary_only accetta."""
    return json.dumps({"summary": join_tsum(tsum)}, ensure_ascii=False)


def parse_summary_response(response):
    """Stesse regole di infer_grpo_videoxum.parse_summary_only_response: una
    risposta che il test rifiuterebbe vale summary vuoto anche in validation."""
    payload = json.loads(response)
    if not isinstance(payload, dict) or set(payload) != {"summary"}:
        raise ValueError('la risposta deve contenere solo il campo "summary"')
    if not isinstance(payload["summary"], str) or not payload["summary"].strip():
        raise ValueError("summary deve essere una stringa non vuota")
    return payload["summary"].strip()


def load_v2t_records(path):
    """Record con video e tsum; la selezione (vsum_onehot*) non serve. Una sola
    passata sul JSON (i file consensus sono da 170 MB)."""
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    items = payload.items() if isinstance(payload, dict) else enumerate(payload)
    records = []
    for key, record in items:
        if missing := {"sampled_frames", "tsum"} - set(record):
            raise ValueError(f"Record {key!r}: campi mancanti {sorted(missing)}")
        captions = record["tsum"]
        if not isinstance(captions, list) or not captions or any(
            not str(sentence).strip() for sentence in captions
        ):
            raise ValueError(f"Record {key!r}: tsum deve contenere frasi non vuote")
        records.append({
            "video_id": str(record.get("video_id", key)),
            "sampled_frames": int(record["sampled_frames"]),
            "reference_summary": join_tsum(captions),
            "target": build_v2t_target(captions),
        })
    if not records:
        raise ValueError(f"Nessun record utilizzabile in {path}")
    return records


class VideoXumV2TDataset(VideoXumSFTDataset):
    """Prompt e system prompt di infer_grpo_videoxum.py --summary_only."""

    def __init__(self, records, video_dir, timeline_unit="frame", summary_style="reference"):
        super().__init__(records, video_dir, timeline_unit)
        self.summary_style = summary_style

    def __getitem__(self, index):
        record = self.records[index]
        count = record["sampled_frames"]
        timeline_count = timeline_length(count, self.timeline_unit)
        return {
            "video_id": record["video_id"],
            "video_path": str(self.paths[index]),
            "sampled_frames": count,
            "timeline_unit": self.timeline_unit,
            "timeline_count": timeline_count,
            "system_prompt": SYSTEM_PROMPTS[self.summary_style],
            "prompt": build_summary_only_prompt(timeline_count, self.summary_style,
                                                self.timeline_unit),
            "reference_summary": record["reference_summary"],
            "target": record["target"],
        }


def load_v2t_datasets(args):
    """Come sft_train_videoxum.load_datasets, con i record e il prompt V2T."""
    records = load_v2t_records(args.data_json)
    if args.val_json:
        train_records, val_records = records, load_v2t_records(args.val_json)
    else:
        train_records, val_records = split_records(records, args.val_ratio, args.seed)
    val_video_dir = args.val_video_dir or args.video_dir
    build = lambda items, video_dir: VideoXumV2TDataset(
        items, video_dir, args.timeline_unit, args.summary_style)
    train, val = build(train_records, args.video_dir), build(val_records, val_video_dir)
    print(f"[data] train={len(train)}, val={len(val)}, val_video_dir={val_video_dir}")
    return train, val


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
    """Punteggio di checkpoint-best, sempre sui primi --eval_steps_samples video,
    cosi' validation periodica e di fine epoca restano confrontabili."""
    values = metric_values[args.best_metric]
    if args.eval_steps:
        values = values[:args.eval_steps_samples]
    return sum(values) / len(values)


@torch.no_grad()
def evaluate(model, processor, dataset, args, device, metrics_file, run_id, epoch,
             model_type, limit=None, optimizer_step=None):
    """Validation generativa del solo summary."""
    model.eval()
    limit = validation_limit(dataset, args.eval_samples if limit is None else limit)
    invalid_predictions = 0
    metric_values = {name: [] for name in TEXT_METRIC_NAMES}
    candidates, reference_summaries = [], []
    for index in tqdm(range(limit), desc="validation", unit="video", leave=False):
        sample = dataset[index]
        frames = decode_video_frames(sample, args)
        response = generate_response(model, processor, sample, args, device, frames, model_type)
        parse_error = None
        try:
            summary = parse_summary_response(response)
        except (json.JSONDecodeError, ValueError) as error:
            summary, parse_error = "", str(error)
            invalid_predictions += 1
        sample_metrics = summary_metrics(summary, sample["reference_summary"])
        for name, value in sample_metrics.items():
            metric_values[name].append(value)
        candidates.append(summary)
        reference_summaries.append(sample["reference_summary"])

        write_metric(
            metrics_file, run_id=run_id, record_type="sample", split="validation",
            epoch=epoch, optimizer_step=optimizer_step, sample_position=index + 1,
            dataset_index=index, video_id=sample["video_id"],
            generated_response=response, model_type=model_type,
            prediction_valid=parse_error is None, parse_error=parse_error,
            summary=summary, **sample_metrics,
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
    # Un modello che impara un summary template lo ripete identico su video diversi.
    summary_values["distinct_summary_ratio"] = (
        len({" ".join(candidate.split()) for candidate in candidates}) / len(candidates)
    )
    # cider e' None quando la validation ha meno di 2 video (metrica corpus-level).
    cider_text = "n/d" if summary_values["cider"] is None else f"{summary_values['cider']:.4f}"
    print(f"[val] ROUGE-1={summary_values['rouge1']:.4f} ROUGE-L={summary_values['rougeL']:.4f} "
          f"BLEU@4={summary_values['bleu4_sentence']:.4f} CIDEr={cider_text} "
          f"words={summary_values['summary_words']:.1f} "
          f"distinct_summary_ratio={summary_values['distinct_summary_ratio']:.3f} "
          f"valid_predictions={limit - invalid_predictions}/{limit}")
    write_metric(
        metrics_file, run_id=run_id, record_type="summary", split="validation",
        epoch=epoch, optimizer_step=optimizer_step, samples=limit,
        invalid_predictions=invalid_predictions,
        metric_protocol="summary vs joined tsum (split() tokenization)",
        video_time_mode=args.video_time_mode, timeline_unit=args.timeline_unit,
        summary_style=args.summary_style, best_metric=args.best_metric, **summary_values,
    )
    return summary_values


class V2TSchedule(SFTSchedule):
    """SFTSchedule con la validation V2T; budget, checkpoint e resume restano
    quelli della SFT V2V."""

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

    def _end_of_epoch(self, state, epoch):
        metrics = self._validate(state, epoch)
        save_named(self.trainer, self.save_dir / f"checkpoint-epoch{epoch}", self.args)
        print(f"[epoch {epoch}] ROUGE-L={metrics['rougeL']:.4f}")


def guard_v2t_dir(save_dir):
    """Rifiuta una directory con checkpoint di un altro task: sovrascriverebbe
    checkpoint-best di una run V2V/V2VT (vedi sft_train_videoxum.guard_checkpoint_dir)."""
    for checkpoint in sorted(Path(save_dir).glob("checkpoint-*")):
        if not (checkpoint / "adapter_config.json").is_file():
            continue
        task = (read_video_encoding(checkpoint) or {}).get("task", "v2v")
        if task != "v2t":
            raise SystemExit(
                f"[task] {save_dir} contiene {checkpoint.name} con task={task}; questo "
                "run e' v2t. Usare un --saveroot dedicato."
            )


def train(args):
    if not args.resume_from:
        guard_v2t_dir(args.saveroot / args.model.name)
    print(f"[v2t] summary_style={args.summary_style} "
          f"eval_max_new_tokens={args.eval_max_new_tokens}")
    sft_train_videoxum_trl.train(args, datasets_loader=load_v2t_datasets,
                                 schedule_cls=V2TSchedule)


def parse_args():
    """Flag propri della V2T, poi la CLI di sft_train_videoxum_trl.py sul resto."""
    parser = argparse.ArgumentParser(add_help=False, description="Opzioni V2T di "
                                     "sft_train_videoxum_v2t_trl.py (le altre sono quelle di "
                                     "sft_train_videoxum_trl.py e sft_train_videoxum.py)")
    parser.add_argument("--summary_style", choices=SUMMARY_STYLES, default="reference",
        help="stile del summary chiesto nel prompt (train_grpo.SUMMARY_STYLES); deve "
             "coincidere con quello dell'inferenza --summary_only")
    parser.add_argument("--best_metric", choices=BEST_METRIC_CHOICES, default="rougeL",
        help="metrica testuale di validation per checkpoint-best")
    parser.add_argument("--eval_max_new_tokens", type=positive_int, default=192,
        help="token massimi in validation: il target contiene solo il summary "
             "(~45 parole con --summary_style reference)")
    if {"-h", "--help"} & set(sys.argv[1:]): parser.print_help(); print()
    v2t_args, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    args = sft_train_videoxum_trl.parse_args()
    vars(args).update(vars(v2t_args))
    # Finisce in video_encoding.json: distingue i checkpoint V2T da quelli V2V/V2VT.
    args.task = "v2t"
    if args.diversity_check_samples or args.video_dependence_samples:
        raise SystemExit("--diversity_check_samples e --video_dependence_samples non sono "
                         "supportati: leggono l'output V2V. La ripetizione dei summary e' gia' "
                         "misurata a ogni validation (distinct_summary_ratio)")
    return args


if __name__ == "__main__": train(parse_args())

"""
Valutazione T-Sum confrontabile con i numeri pubblicati su VideoXum.

Replica la pipeline dello script ufficiale VTSUM-BLIP (COCO + COCOEvalCap):
  - PTBTokenizer di Stanford CoreNLP (Java): lowercase, rimozione punteggiatura
  - BLEU@1-4, METEOR 1.5 (Java), ROUGE-L, CIDEr-D di pycocoevalcap
  - un solo riferimento per video (le frasi di ground_truth unite con uno spazio)

A differenza di eval.py (tokenizzazione `.split()`, METEOR/BLEU di NLTK) questi
valori si possono mettere accanto a quelli dei paper. I paper riportano x100.

Per default i video di riferimento senza predizione valida valgono come summary
vuoto (punteggio 0), come succederebbe nello script ufficiale, invece di essere
scartati: scartarli gonfierebbe i punteggi delle run con parsing fallito.

Richiede Java nel PATH (su Leonardo: `module load openjdk/11.0.20.1_1`).
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

from eval import load_candidate_records, load_reference_records


def build_pairs(candidates_path: str, references_path: str, missing: str):
    candidate_records = load_candidate_records(candidates_path)
    references = load_reference_records(references_path)

    gts, res = {}, {}
    missing_ids, empty_ids = [], []
    for video_id, reference in references.items():
        if not reference:
            continue
        record = candidate_records.get(video_id)
        summary = (record or {}).get("summary", "").strip()
        if not summary:
            (missing_ids if record is None else empty_ids).append(video_id)
            if missing == "skip":
                continue
        gts[video_id] = [{"caption": reference.replace("\n", " ")}]
        res[video_id] = [{"caption": summary.replace("\n", " ")}]

    extra = sorted(set(candidate_records) - set(references))
    coverage = {
        "reference_videos": len(references),
        "prediction_videos": len(candidate_records),
        "evaluated_videos": len(gts),
        "missing_predictions": len(missing_ids),
        "empty_summaries": len(empty_ids),
        "predictions_without_reference": len(extra),
        "missing_policy": missing,
    }
    return gts, res, coverage


def evaluate(gts, res):
    tokenizer = PTBTokenizer()
    gts_tok = tokenizer.tokenize(gts)
    res_tok = tokenizer.tokenize(res)

    metrics, per_sample = {}, {}
    scorers = [
        (Bleu(4), ["BLEU@1", "BLEU@2", "BLEU@3", "BLEU@4"]),
        (Meteor(), ["METEOR"]),
        (Rouge(), ["ROUGE-L"]),
        (Cider(), ["CIDEr"]),
    ]
    for scorer, names in scorers:
        print(f"⏳ Calcolando {', '.join(names)}...")
        score, scores = scorer.compute_score(gts_tok, res_tok)
        if isinstance(score, list):
            for name, value, values in zip(names, score, scores):
                metrics[name] = float(value)
                per_sample[name] = [float(v) for v in values]
        else:
            metrics[names[0]] = float(score)
            per_sample[names[0]] = [float(v) for v in scores]

    lengths = {
        "candidate_words_mean": float(np.mean([len(res_tok[k][0].split()) for k in res_tok])),
        "reference_words_mean": float(np.mean([len(gts_tok[k][0].split()) for k in gts_tok])),
    }
    return metrics, per_sample, gts_tok, res_tok, lengths


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates-json", required=True,
                        help="JSONL delle predizioni o directory di shard JSONL")
    parser.add_argument("--references-json", required=True,
                        help="test.json di VideoXum (campi id, ground_truth)")
    parser.add_argument("--output-json", required=True,
                        help="JSON con metriche aggregate, coverage e punteggi per sample")
    parser.add_argument("--missing", choices=("empty", "skip"), default="empty",
                        help="empty: video senza summary valgono 0 (confrontabile); "
                             "skip: vengono esclusi (solo diagnostica)")
    args = parser.parse_args()

    if shutil.which("java") is None:
        raise SystemExit("Java non trovato: esegui `module load openjdk/11.0.20.1_1`")

    gts, res, coverage = build_pairs(args.candidates_json, args.references_json, args.missing)
    if len(gts) < 2:
        raise SystemExit("Servono almeno 2 video: CIDEr e' corpus-level")
    print(f"📊 Valuto {coverage['evaluated_videos']} video "
          f"(mancanti {coverage['missing_predictions']}, vuoti {coverage['empty_summaries']}, "
          f"policy={args.missing})")

    metrics, per_sample, gts_tok, res_tok, lengths = evaluate(gts, res)

    # i scorer iterano su gts.keys(): stesso ordine per i punteggi per sample
    video_ids = list(gts_tok.keys())
    samples = [
        {"video_id": vid, "summary_tok": res_tok[vid][0], "reference_tok": gts_tok[vid][0],
         **{name: values[i] for name, values in per_sample.items() if name in ("CIDEr", "METEOR", "ROUGE-L")}}
        for i, vid in enumerate(video_ids)
    ]
    output = {
        "metrics": metrics,
        "metrics_percent": {k: v * 100 for k, v in metrics.items()},
        "coverage": coverage,
        "lengths": lengths,
        "reference_mode": "join",
        "tokenizer": "PTBTokenizer (stanford-corenlp-3.4.1)",
        "candidates": str(args.candidates_json),
        "references": str(args.references_json),
        "samples": samples,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (COCO / VideoXum protocol, x100)")
    print("=" * 60)
    for name, value in metrics.items():
        print(f"  {name:8s} {value * 100:6.2f}")
    print(f"  parole medie: candidati {lengths['candidate_words_mean']:.1f}, "
          f"riferimenti {lengths['reference_words_mean']:.1f}")
    print(f"✅ Salvato in {output_path}")


if __name__ == "__main__":
    main()

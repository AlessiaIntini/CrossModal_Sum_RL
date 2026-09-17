import textwrap
import warnings
from typing import List, NamedTuple, Dict
import torch
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
import numpy as np
from scipy.spatial.distance import cosine
import json
import re
import argparse
from pathlib import Path
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

# Pesi n-gram per BLEU@1..BLEU@4
BLEU_WEIGHTS = {
    'bleu1': (1.0, 0.0, 0.0, 0.0),
    'bleu2': (0.5, 0.5, 0.0, 0.0),
    'bleu3': (1 / 3, 1 / 3, 1 / 3, 0.0),
    'bleu4': (0.25, 0.25, 0.25, 0.25),
}

_cider_warned = False


def compute_meteor_scores(candidates: List[str], references: List[str]) -> List[float]:
    """Calcola METEOR (stemming + sinonimi WordNet) per ogni coppia candidato/riferimento."""
    scores = []
    for candidate, reference in zip(candidates, references):
        # meteor_score vuole reference tokenizzate (lista di liste) e hypothesis tokenizzata
        scores.append(meteor_score([reference.split()], candidate.split()))
    return scores


def compute_corpus_bleu(candidates: List[str], references: List[str]) -> Dict[str, float]:
    """
    Calcola BLEU@1..BLEU@4 a livello di corpus (e' cosi' che lo riportano i paper:
    conteggi di n-grammi accumulati su tutti i samples, non media dei BLEU per frase).

    Nota: la tokenizzazione e' `.split()`, coerente con compute_meteor_scores ma
    diversa dal PTBTokenizer usato dal benchmark VideoXum. I valori sono quindi
    confrontabili fra run propri, solo indicativi verso i numeri pubblicati.
    """
    tok_candidates = [c.split() for c in candidates]
    tok_references = [[r.split()] for r in references]   # una reference per sample

    scores = {}
    with warnings.catch_warnings():
        # nltk avvisa quando un ordine di n-gram non ha nessun match: e' un caso
        # normale su summary lunghi e parafrastici, il valore 0 e' l'informazione
        warnings.simplefilter("ignore", UserWarning)
        for name, weights in BLEU_WEIGHTS.items():
            score = float(corpus_bleu(tok_references, tok_candidates, weights=weights))
            # senza match nltk sostituisce sys.float_info.min per evitare log(0),
            # restituendo denormali tipo 4e-79: significano zero, meglio dirlo
            scores[name] = 0.0 if score < 1e-12 else score
    return scores


def compute_sentence_bleu_scores(
    candidates: List[str],
    references: List[str],
    weights=BLEU_WEIGHTS['bleu4'],
) -> List[float]:
    """
    Calcola BLEU per singolo sample (per le colonne per-riga dell'Excel).
    Usa lo smoothing method1: senza, BLEU@4 su una frase sola e' 0 quasi ovunque
    perche' basta un n-gram order senza match per azzerare la media geometrica.
    """
    smoothing = SmoothingFunction().method1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        scores = [
            float(sentence_bleu([reference.split()], candidate.split(),
                                weights=weights, smoothing_function=smoothing))
            for candidate, reference in zip(candidates, references)
        ]
    return [0.0 if s < 1e-12 else s for s in scores]


def compute_cider_score(candidates: List[str], references: List[str]):
    """
    Calcola CIDEr-D sull'intero insieme di samples.

    ATTENZIONE: e' una metrica corpus-level. I pesi TF-IDF usano ref_len = log(N)
    e df = log(max(1, document_frequency)): con N=1 entrambi valgono 0, tutti i
    pesi si azzerano e il punteggio e' 0 per costruzione, non perche' il candidato
    sia sbagliato. Per questo con meno di 2 samples ritorna NaN invece di 0.

    Per lo stesso motivo (log(N) nella scala dei pesi) i valori NON sono
    confrontabili fra insiemi di dimensione diversa.

    Returns:
        (media, lista di punteggi per sample) - NaN se i samples sono meno di 2
    """
    global _cider_warned

    if len(candidates) < 2:
        if not _cider_warned:
            print("⚠️  CIDEr richiede almeno 2 samples (e' corpus-level): valore -> NaN")
            _cider_warned = True
        return float('nan'), [float('nan')] * len(candidates)

    from pycocoevalcap.cider.cider import Cider

    gts = {i: [ref] for i, ref in enumerate(references)}
    res = {i: [cand] for i, cand in enumerate(candidates)}

    mean_score, per_sample = Cider().compute_score(gts, res)
    return float(mean_score), [float(s) for s in per_sample]


class MetricsResult(NamedTuple):
    """Risultato completo di valutazione"""
    rouge1_p: float
    rouge1_r: float
    rouge1_f1: float
    rouge2_p: float
    rouge2_r: float
    rouge2_f1: float
    rougeL_p: float
    rougeL_r: float
    rougeL_f1: float
    sbert: float
    meteor: float
    num_samples: int
    # Metriche corpus-level: hanno senso solo su un intero dataset.
    # cider resta NaN quando i samples sono meno di 2 (vedi compute_cider_score)
    bleu1: float = float('nan')
    bleu2: float = float('nan')
    bleu3: float = float('nan')
    bleu4: float = float('nan')
    cider: float = float('nan')


class Candidate(NamedTuple):
    """Candidato da valutare"""
    response: str


class VideoEvaluator:
    """
    Valutatore per risposte video usando metriche da paper:
    - ROUGE-1/2/L (Precision, Recall, F1)
    - SBERT (Semantic Similarity)
    - METEOR
    """
    
    def __init__(self, sbert_model: str = "all-MiniLM-L6-v2", device: str = None):
        """
        Inizializza il valutatore.
        
        Args:
            sbert_model: Modello SentenceTransformer da usare
            device: Device ('cuda' o 'cpu'). Se None, usa cuda se disponibile
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        
        # ROUGE Scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
        
        # SBERT Scorer
        self.sbert_model = SentenceTransformer(sbert_model, device=device)
        
        self.text_width = 70
    
    def _compute_rouge_metrics(
        self, 
        candidates: List[str], 
        references: List[str]
    ) -> Dict[str, List[float]]:
        """Calcola ROUGE per tutti i samples"""
        rouge1_p, rouge1_r, rouge1_f1 = [], [], []
        rouge2_p, rouge2_r, rouge2_f1 = [], [], []
        rougeL_p, rougeL_r, rougeL_f1 = [], [], []
        
        for candidate, reference in zip(candidates, references):
            # ROUGE-1
            score_rouge1 = self.rouge_scorer.score(reference, candidate)['rouge1']
            rouge1_p.append(score_rouge1.precision)
            rouge1_r.append(score_rouge1.recall)
            rouge1_f1.append(score_rouge1.fmeasure)
            
            # ROUGE-2
            score_rouge2 = self.rouge_scorer.score(reference, candidate)['rouge2']
            rouge2_p.append(score_rouge2.precision)
            rouge2_r.append(score_rouge2.recall)
            rouge2_f1.append(score_rouge2.fmeasure)
            
            # ROUGE-L
            score_rougeL = self.rouge_scorer.score(reference, candidate)['rougeL']
            rougeL_p.append(score_rougeL.precision)
            rougeL_r.append(score_rougeL.recall)
            rougeL_f1.append(score_rougeL.fmeasure)
        
        return {
            'rouge1_p': rouge1_p,
            'rouge1_r': rouge1_r,
            'rouge1_f1': rouge1_f1,
            'rouge2_p': rouge2_p,
            'rouge2_r': rouge2_r,
            'rouge2_f1': rouge2_f1,
            'rougeL_p': rougeL_p,
            'rougeL_r': rougeL_r,
            'rougeL_f1': rougeL_f1,
        }
    
    def _compute_sbert_similarity(
        self, 
        candidates: List[str], 
        references: List[str]
    ) -> float:
        """Calcola similarità semantica usando SBERT (cosine similarity media)"""
        # Encode tutti i testi
        candidate_embeddings = self.sbert_model.encode(candidates, convert_to_tensor=True)
        reference_embeddings = self.sbert_model.encode(references, convert_to_tensor=True)
        
        # Calcola cosine similarity
        similarities = []
        for cand_emb, ref_emb in zip(candidate_embeddings, reference_embeddings):
            sim = 1 - cosine(cand_emb.cpu().numpy(), ref_emb.cpu().numpy())
            similarities.append(sim)
        
        return float(np.mean(similarities))
    
    def _compute_meteor(self, candidates: List[str], references: List[str]) -> List[float]:
        """Calcola METEOR (stemming + sinonimi WordNet) per ogni sample."""
        return compute_meteor_scores(candidates, references)
    
    def evaluate(
        self, 
        candidates: List[Candidate], 
        references: List[str],
        verbose: bool = True
    ) -> MetricsResult:
        """
        Valuta le risposte candidate rispetto alle reference.
        
        Args:
            candidates: Lista di Candidate con .response
            references: Lista di stringhe di riferimento
            verbose: Se True, stampa i risultati per ogni coppia
            
        Returns:
            MetricsResult con tutte le metriche
        """
        if len(candidates) != len(references):
            raise ValueError(f"Numero di candidati ({len(candidates)}) != riferimenti ({len(references)})")
        
        candidate_responses = [c.response for c in candidates]
        
        # Stampa confronto
        if verbose:
            for i, (candidate, reference) in enumerate(zip(candidate_responses, references)):
                hyp = textwrap.fill(candidate, width=self.text_width)
                ref = textwrap.fill(reference, width=self.text_width)
                
                print(f"\n{'='*80}")
                print(f"Sample {i+1}/{len(candidates)}")
                print(f"{'='*80}")
                print(f"Hypothesis:\n{hyp}")
                print(f"\nGround Truth:\n{ref}")
        
        # Calcola metriche
        print("\n⏳ Calcolando ROUGE metrics...")
        rouge_metrics = self._compute_rouge_metrics(candidate_responses, references)
        
        print("⏳ Calcolando SBERT similarity...")
        sbert_score = self._compute_sbert_similarity(candidate_responses, references)
        
        print("⏳ Calcolando METEOR scores...")
        meteor_scores = self._compute_meteor(candidate_responses, references)

        print("⏳ Calcolando BLEU scores...")
        bleu_scores = compute_corpus_bleu(candidate_responses, references)

        print("⏳ Calcolando CIDEr score...")
        cider_score, _ = compute_cider_score(candidate_responses, references)

        return MetricsResult(
            rouge1_p=float(np.mean(rouge_metrics['rouge1_p'])),
            rouge1_r=float(np.mean(rouge_metrics['rouge1_r'])),
            rouge1_f1=float(np.mean(rouge_metrics['rouge1_f1'])),
            rouge2_p=float(np.mean(rouge_metrics['rouge2_p'])),
            rouge2_r=float(np.mean(rouge_metrics['rouge2_r'])),
            rouge2_f1=float(np.mean(rouge_metrics['rouge2_f1'])),
            rougeL_p=float(np.mean(rouge_metrics['rougeL_p'])),
            rougeL_r=float(np.mean(rouge_metrics['rougeL_r'])),
            rougeL_f1=float(np.mean(rouge_metrics['rougeL_f1'])),
            sbert=sbert_score,
            meteor=float(np.mean(meteor_scores)),
            num_samples=len(candidates),
            bleu1=bleu_scores['bleu1'],
            bleu2=bleu_scores['bleu2'],
            bleu3=bleu_scores['bleu3'],
            bleu4=bleu_scores['bleu4'],
            cider=cider_score,
        )
    
    def print_summary(self, result: MetricsResult) -> None:
        """Stampa il riassunto completo delle metriche"""
        print(f"\n{'='*80}")
        print("EVALUATION RESULTS")
        print(f"{'='*80}")
        print(f"Samples: {result.num_samples}\n")
        
        print("ROUGE-1")
        print(f"  Precision: {result.rouge1_p:.4f}")
        print(f"  Recall:    {result.rouge1_r:.4f}")
        print(f"  F1:        {result.rouge1_f1:.4f}")
        
        print("\nROUGE-2")
        print(f"  Precision: {result.rouge2_p:.4f}")
        print(f"  Recall:    {result.rouge2_r:.4f}")
        print(f"  F1:        {result.rouge2_f1:.4f}")
        
        print("\nROUGE-L")
        print(f"  Precision: {result.rougeL_p:.4f}")
        print(f"  Recall:    {result.rougeL_r:.4f}")
        print(f"  F1:        {result.rougeL_f1:.4f}")
        
        print("\nBLEU (corpus-level)")
        print(f"  BLEU@1:    {result.bleu1:.4f}")
        print(f"  BLEU@2:    {result.bleu2:.4f}")
        print(f"  BLEU@3:    {result.bleu3:.4f}")
        print(f"  BLEU@4:    {result.bleu4:.4f}")

        print(f"\nSBERT Similarity: {result.sbert:.4f}")
        print(f"METEOR Score: {result.meteor:.4f}")
        cider = f"{result.cider:.4f}" if not np.isnan(result.cider) else "n/d (servono >= 2 samples)"
        print(f"CIDEr-D Score: {cider}")
        print(f"{'='*80}\n")

    def get_csv_header(self) -> str:
        """Intestazione per salvare i risultati in CSV"""
        return ("rouge1_p,rouge1_r,rouge1_f1,rouge2_p,rouge2_r,rouge2_f1,"
                "rougeL_p,rougeL_r,rougeL_f1,sbert,meteor,"
                "bleu1,bleu2,bleu3,bleu4,cider")

    def result_to_csv_row(self, result: MetricsResult) -> str:
        """Converte il risultato in una riga CSV"""
        return f"{result.rouge1_p:.4f},{result.rouge1_r:.4f},{result.rouge1_f1:.4f}," \
               f"{result.rouge2_p:.4f},{result.rouge2_r:.4f},{result.rouge2_f1:.4f}," \
               f"{result.rougeL_p:.4f},{result.rougeL_r:.4f},{result.rougeL_f1:.4f}," \
               f"{result.sbert:.4f},{result.meteor:.4f}," \
               f"{result.bleu1:.4f},{result.bleu2:.4f},{result.bleu3:.4f}," \
               f"{result.bleu4:.4f},{result.cider:.4f}"

# Parametri fissi di configurazione
CANDIDATES_JSON = "/leonardo_work/IscrC_MSMU/aintini0/V2Xum-LLM/results/videoxum_train_tsum"
REFERENCES_JSON = "/leonardo_work/IscrC_HydroFM/aintini0/datasets/videoxum/metadata/train.json"
OUTPUT_JSONL = "/leonardo_work/IscrC_HydroFM/aintini0/GRPOVideo/results/evaluation/videoxum_train_tsum_metrics.jsonl"
VERBOSE = False


def load_candidate_records(path: str) -> Dict[str, dict]:
    """Carica un JSONL oppure tutti gli shard JSONL di una directory."""
    candidate_path = Path(path)
    if candidate_path.is_dir():
        files = sorted(candidate_path.glob("*.jsonl"))
    elif candidate_path.is_file():
        files = [candidate_path]
    else:
        raise FileNotFoundError(f"Candidati non trovati: {candidate_path}")
    if not files:
        raise FileNotFoundError(f"Nessun file JSONL trovato in: {candidate_path}")

    records: Dict[str, dict] = {}
    duplicates = 0
    for file_path in files:
        with file_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"JSON non valido in {file_path}:{line_number}: {error}"
                    ) from error
                video_id = record.get("video_id")
                if not video_id:
                    video_file = record.get("video_file", "")
                    video_id = Path(video_file).stem if video_file else None
                if not video_id:
                    print(f"⚠️  Record senza video_id in {file_path}:{line_number}, skipping...")
                    continue
                if video_id in records:
                    duplicates += 1
                    continue
                summary = record.get("summary")
                if summary is None:
                    summaries = record.get("summaries", [])
                    summary = summaries[0] if summaries else None

                if summary is None:
                    raw_generated_response = record.get("generated_response", "")
                    try:
                        generated_response = json.loads(raw_generated_response)
                        summary = (generated_response.get("summary", "")
                                   if isinstance(generated_response, dict) else "")
                    except (json.JSONDecodeError, TypeError):
                        # Alcune generazioni hanno JSON non valido nella parte
                        # selected_frames, ma un campo summary ancora leggibile.
                        match = re.search(
                            r'"summary"\s*:\s*("(?:\\.|[^"\\])*")',
                            raw_generated_response,
                            flags=re.DOTALL,
                        )
                        if match:
                            try:
                                summary = json.loads(match.group(1))
                            except json.JSONDecodeError:
                                summary = ""
                        else:
                            summary = ""
                records[str(video_id)] = {**record, "video_id": str(video_id),
                                          "summary": str(summary)}
    print(f"📂 Candidati caricati: {len(records)} da {len(files)} file")
    if duplicates:
        print(f"⚠️  Record candidati duplicati ignorati: {duplicates}")
    return records


def load_reference_records(path: str) -> Dict[str, str]:
    """Carica le ground truth di train.json indicizzandole per campo `id`."""
    reference_path = Path(path)
    with reference_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Il file dei riferimenti deve contenere una lista: {reference_path}")

    references: Dict[str, str] = {}
    for item in data:
        video_id = item.get("id") or item.get("video_id")
        if not video_id:
            continue
        ground_truth = item.get("ground_truth", "")
        if isinstance(ground_truth, list):
            ground_truth = " ".join(str(sentence).strip() for sentence in ground_truth)
        references[str(video_id)] = str(ground_truth).strip()
    print(f"📂 Riferimenti caricati: {len(references)}")
    return references

# Esempio di utilizzo
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Valuta video summaries")
    parser.add_argument("--candidates-json", type=str, default=CANDIDATES_JSON,
                        help=f"Path al JSON con i candidati (default: {CANDIDATES_JSON})")
    parser.add_argument("--references-json", type=str, default=REFERENCES_JSON,
                        help=f"Path al JSON con i riferimenti (default: {REFERENCES_JSON})")
    parser.add_argument("--output-jsonl", type=str, default=OUTPUT_JSONL,
                        help=f"Path al file JSONL di output (default: {OUTPUT_JSONL})")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=("cpu", "cuda"),
                        help="Device per SBERT; la valutazione usa CPU di default")
    parser.add_argument("--sbert-model", type=str,
                        default="/leonardo_work/IscrC_HydroFM/aintini0/models/all-MiniLM-L6-v2",
                        help="Nome o percorso locale del modello SentenceTransformer")
    parser.add_argument("--verbose", action="store_true", default=VERBOSE,
                        help="Stampa risultati dettagliati")
    
    args = parser.parse_args()
    
    print(f"📂 Caricando candidati da {args.candidates_json}")
    candidates_data = load_candidate_records(args.candidates_json)
    print(f"📂 Caricando riferimenti da {args.references_json}")
    references_by_video = load_reference_records(args.references_json)

    candidates = []
    references = []
    video_ids = []
    summaries = []
    skipped = 0
    for video_id, cand_data in candidates_data.items():
        summary = cand_data.get("summary", "").strip()
        if not summary:
            print(f"⚠️  Nessun summary per {video_id}, skipping...")
            skipped += 1
            continue
        reference = references_by_video.get(video_id, "")
        if not reference:
            print(f"⚠️  Nessun riferimento per {video_id}, skipping...")
            skipped += 1
            continue
        video_ids.append(video_id)
        summaries.append(summary)
        candidates.append(Candidate(response=summary))
        references.append(reference)

    if not candidates:
        raise RuntimeError("Nessuna coppia candidato/riferimento valida da valutare")

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    evaluator = VideoEvaluator(sbert_model=args.sbert_model, device=args.device)
    print(f"\n📊 Valuto {len(candidates)} coppie a livello corpus...")
    eval_result = evaluator.evaluate(candidates, references, verbose=args.verbose)
    metrics = eval_result._asdict()
    output_entries = [
        {"record_type": "sample", "video_id": video_id, "summary": summary,
         "ground_truth": reference}
        for video_id, summary, reference in zip(video_ids, summaries, references)
    ]
    output_entries.append({"record_type": "summary", "metrics": metrics,
                           "skipped": skipped})
    with output_path.open("w", encoding="utf-8") as jsonl_file:
        for entry in output_entries:
            jsonl_file.write(json.dumps(entry, ensure_ascii=False, allow_nan=True) + "\n")

    evaluator.print_summary(eval_result)
    print(f"✅ Valutazione completata! Risultati salvati in {output_path}")
    print(f"   Total videos: {len(candidates)}; skipped: {skipped}")
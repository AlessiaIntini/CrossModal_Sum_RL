"""Reward componibili per GRPO su VideoXum (frame discreti + summary testuale).

Il modulo non dipende dal trainer: i dati statici di un video vengono preparati una
sola volta da :class:`VideoSummaryReward` e riutilizzati per tutti i rollout GRPO.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Hashable, Protocol, Sequence

import torch
import torch.nn.functional as F


EPSILON = 1e-8
EXPECTED_FIELDS = {"selected_frames", "summary"}
# Pesi della somma pesata in VideoSummaryReward.score. Un peso 0 spegne il
# termine nella reward di training, ma la componente resta calcolata e loggata.
DEFAULT_REWARD_WEIGHTS = {
    # GRPO e' il terzo stadio (dopo SFT e DPO): la quantita' di frame e' gia'
    # stata appresa, quindi il segnale va su QUALI frame (f_beta) e sulla qualita'
    # del riassunto (grounding), non sul budget.
    "f_beta": 0.30,
    # Inerte in training: zero nel 74% dei rollout e ~2.5% della varianza
    # intra-gruppo. Resta calcolata e loggata, ma non guida l'advantage.
    "diversity": 0.00,
    "representativeness": 0.10,
    "grounding": 0.25,
    # Quasi inerte: la similarita' SBERT col reference satura, std intra-gruppo
    # 0.017 contro 0.094 di grounding. Alzarne il peso diluisce senza aggiungere
    # segnale.
    "reference": 0.05,
    "format": 0.05,
    # Lunghezza del summary rispetto al reference: SBERT (reference) e grounding
    # sono quasi insensibili alla lunghezza, mentre CIDEr-D/ROUGE/METEOR la
    # penalizzano molto (CIDEr-D con sigma=6 parole). Senza questo termine GRPO
    # converge su una frase da ~25 parole contro reference da ~49.
    "summary_length": 0.10,
    # Budget di frame come componente additiva leggera, non piu' come
    # moltiplicatore globale: il vincolo riguarda la SELEZIONE, e da moltiplicatore
    # scalava anche grounding/reference/summary_length, che col numero di frame non
    # c'entrano. Misurato sul run precedente, il moltiplicatore decideva il 73% dei
    # confronti intra-gruppo contro il 67% del contenuto, e annullava l'advantage
    # nel 28% dei gruppi.
    "length_budget": 0.05,
    # Spearman fra la selezione binaria e la saliency media dei 10 annotatori, in
    # [0, 1] come (rho+1)/2: e' lo Spearman di eval_v2vt_sum.py (VideoXum), che
    # confronta con la saliency MEDIA. f_beta con aggregazione max premia l'accordo
    # con un annotatore qualsiasi: sui rollout del run grpo_sft8k_4k concorda con
    # lo Spearman solo nel 64% dei confronti intra-gruppo. Spento di default, cosi'
    # total_reward_default resta quello dei run precedenti.
    "saliency_rank": 0.00,
}
# Deviazione standard in log-rapporto parole(summary)/parole(reference):
# rapporto 0.7 o 1.4 -> ~0.6, rapporto 0.5 o 2.0 -> ~0.14.
SUMMARY_LENGTH_SIGMA = 0.35
# Stessa forma, ma sul rapporto frame_selezionati/budget. Costante separata da
# SUMMARY_LENGTH_SIGMA perche' la tolleranza sul budget di frame e quella sulla
# lunghezza del testo vanno tarate in modo indipendente.
LENGTH_BUDGET_SIGMA = 0.35
# Riscalatura affine del coseno SBERT summary-vs-reference in [0, 1]:
# clamp((cos - floor) / scale). I default -1.0 / 2.0 riproducono la vecchia
# formula (cos+1)/2, che pero' satura: i coseni reali stanno fra p5=0.373 e
# p95=0.770, quindi finiscono compressi in [0.69, 0.89] con std intra-gruppo
# 0.029, cioe' il termine "reference" non discrimina fra i rollout di un gruppo.
# floor=0.30 scale=0.50 porta la std intra-gruppo a 0.114 (3.9x) clippando circa
# l'1% per lato.
REFERENCE_COSINE_FLOOR = -1.0
REFERENCE_COSINE_SCALE = 2.0
# Aggregazioni ammesse per la F-beta sui 10 annotatori.
F_BETA_AGGREGATIONS = ("mean", "max")

_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)

def _strip_code_fence(completion: str) -> str:
    if not isinstance(completion, str):
        return completion
    match = _CODE_FENCE_RE.match(completion.strip())
    return match.group(1) if match else completion


class CompletionValidationError(ValueError):
    """Completion JSON non conforme al contratto richiesto."""


@dataclass(frozen=True)
class ParsedCompletion:
    selected_frames: list[list[int]]
    summary: str
    pred_mask: torch.Tensor
    selected_indices: list[int]
    merged_adjacent_count: int = 0


def pair_timeline_length(frame_count: int) -> int:
    """Numero di segmenti (coppie di frame) di un video da frame_count frame."""
    return (frame_count + 1) // 2


def expand_pair_completion(
    parsed: ParsedCompletion, frame_count: int
) -> ParsedCompletion:
    """Riporta una completion espressa in segmenti agli N frame originali.

    Il segmento g copre i frame 2g e 2g+1 (l'ultimo solo N-1 se N e' dispari).
    Le feature visive, le maschere degli eventi e il budget della reward restano
    sui frame, quindi ogni componente conserva la stessa semantica."""
    intervals = [
        [2 * start, min(2 * end + 1, frame_count - 1)]
        for start, end in parsed.selected_frames
    ]
    indices = [index for start, end in intervals for index in range(start, end + 1)]
    mask = torch.zeros(frame_count, dtype=torch.bool)
    mask[indices] = True
    return ParsedCompletion(
        selected_frames=intervals,
        summary=parsed.summary,
        pred_mask=mask,
        selected_indices=indices,
        merged_adjacent_count=parsed.merged_adjacent_count,
    )


class SemanticEncoder(Protocol):
    """Interfaccia minima per Sentence-BERT o encoder semantici equivalenti."""

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        """Restituisce un tensore float ``[len(texts), embedding_dim]``."""


class SentenceTransformerEncoder:
    """Adapter lazy per sentence-transformers, sempre eseguito senza gradienti."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise ImportError(
                "Installa sentence-transformers per usare SentenceTransformerEncoder"
            ) from error
        self.model = SentenceTransformer(model_name, device=device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        values = self.model.encode(
            list(texts), convert_to_tensor=True, normalize_embeddings=True
        )
        return values.detach().float()


ReferenceMetric = Callable[[str, str, SemanticEncoder], float]
EventMaskConverter = Callable[[Any, int], Any]


@dataclass
class VideoRewardCache:
    count: int
    max_selected: int
    vsum_onehot: torch.Tensor       # bool [10, T]
    event_masks: torch.Tensor       # bool [J, 10, T]
    frame_features: torch.Tensor    # normalized float [T, D]
    caption_embeddings: torch.Tensor
    reference_embedding: torch.Tensor
    consensus: torch.Tensor         # float [T]
    captions: tuple[str, ...]
    reference_summary: str
    frame_floor_sim: float          # similarita' coseno "tipica" tra due frame del video (ridondanza di base)
    frame_min_sim: float            # similarita' coseno minima osservabile tra due frame del video
    consensus_ranks: torch.Tensor   # float64 [T], ranghi medi (ties) di consensus


def _is_integer(value: Any) -> bool:
    # bool e' sottoclasse di int in Python, ma non e' un indice valido.
    return isinstance(value, int) and not isinstance(value, bool)


def _strict_json_loads(completion: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"costante JSON non standard: {value}")

    def reject_duplicate_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"campo JSON duplicato: {key}")
            result[key] = value
        return result

    return json.loads(
        completion,
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicate_fields,
    )


def _decode_json(completion: str) -> Any:
    if not isinstance(completion, str):
        raise CompletionValidationError("completion deve essere una stringa")
    try:
        return _strict_json_loads(_strip_code_fence(completion))
    except (json.JSONDecodeError, ValueError) as error:
        raise CompletionValidationError(f"JSON non valido: {error}") from error

def parse_and_validate_completion(
    completion: str,
    count: int,
    max_selected: int,
) -> ParsedCompletion:
    """Parsing rigido dell'unico oggetto JSON ammesso.

    Gli intervalli sono inclusivi, cronologici, separati da almeno un frame e
    coprono complessivamente al massimo ``max_selected`` indici unici.
    """
    if count <= 0:
        raise ValueError("count deve essere positivo")
    if max_selected <= 0:
        raise ValueError("max_selected deve essere positivo")

    payload = _decode_json(completion)
    if not isinstance(payload, dict):
        raise CompletionValidationError("la radice JSON deve essere un oggetto")
    if set(payload) != EXPECTED_FIELDS:
        raise CompletionValidationError(
            "l'oggetto deve contenere esattamente selected_frames e summary"
        )

    intervals = payload["selected_frames"]
    if not isinstance(intervals, list) or not intervals:
        raise CompletionValidationError("selected_frames deve essere una lista non vuota")

    selected_indices: list[int] = []
    clean_intervals: list[list[int]] = []
    previous_end = -2
    merged_adjacent_count = 0
    for item in intervals:
        if not isinstance(item, list) or len(item) != 2:
            raise CompletionValidationError("ogni intervallo deve essere [start, end]")
        start, end = item
        if not _is_integer(start) or not _is_integer(end):
            raise CompletionValidationError("start ed end devono essere interi")
        if not 0 <= start <= end < count:
            raise CompletionValidationError(f"intervallo fuori range: [{start}, {end}]")
        if start <= previous_end:
            raise CompletionValidationError("intervalli non ordinati o sovrapposti")
        if previous_end >= 0 and start == previous_end + 1:
            # Cosmetico, non logico: i frame coperti non cambiano rispetto a un
            # unico intervallo [prev_start, end]. Si unisce automaticamente
            # invece di scartare l'intero completion (vedi compute_format_reward
            # e VideoSummaryReward.score per la piccola penalita' associata),
            # cosi' un errore puramente sintattico non azzera anche f1/grounding
            # e non elimina la varianza di reward all'interno del gruppo GRPO.
            clean_intervals[-1][1] = end
            merged_adjacent_count += 1
        else:
            clean_intervals.append([start, end])
        selected_indices.extend(range(start, end + 1))
        previous_end = end

    if len(selected_indices) != len(set(selected_indices)):
        raise CompletionValidationError("uno stesso frame e' selezionato piu' volte")

    summary = payload["summary"]
    if not isinstance(summary, str) or not summary.strip():
        raise CompletionValidationError("summary deve essere una stringa non vuota")

    # Il superamento del budget non e' piu' un errore strutturale: viene
    # penalizzato in modo continuo da VideoSummaryReward.score (length_penalty)
    # invece di azzerare qui l'intera completion con un muro discreto.

    pred_mask = torch.zeros(count, dtype=torch.bool)
    pred_mask[selected_indices] = True
    return ParsedCompletion(
        selected_frames=clean_intervals,
        summary=summary.strip(),
        pred_mask=pred_mask,
        selected_indices=selected_indices,
        merged_adjacent_count=merged_adjacent_count,
    )


def compute_format_reward(completion: str, count: int, max_selected: int) -> float:
    """Credito graduale a quattro livelli, indipendente dalla validita' finale."""
    try:
        payload = _strict_json_loads(_strip_code_fence(completion)) if isinstance(completion, str) else None
    except (json.JSONDecodeError, ValueError):
        return 0.0

    reward = 0.25
    if not isinstance(payload, dict):
        return reward

    intervals = payload.get("selected_frames")
    structural = (
        isinstance(intervals, list)
        and bool(intervals)
        and all(isinstance(item, list) and len(item) == 2 for item in intervals)
    )
    if structural:
        reward += 0.25

    indices_valid = structural
    selected_count = 0
    previous_end = -2
    if structural:
        for start, end in intervals:
            if (
                not _is_integer(start)
                or not _is_integer(end)
                or not 0 <= start <= end < count
                or start <= previous_end  # overlap/ordine: resta un errore vero
            ):
                indices_valid = False
                break
            selected_count += end - start + 1
            previous_end = end
        indices_valid = indices_valid and selected_count <= max_selected
    if indices_valid:
        reward += 0.25

    summary = payload.get("summary")
    if isinstance(summary, str) and bool(summary.strip()):
        reward += 0.25
    return reward


def _log_ratio_gaussian(ratio: float, sigma: float) -> float:
    """Gaussiana sul log-rapporto: 1 a rapporto esatto, simmetrica tra eccesso e
    difetto (meta' e doppio valgono uguale)."""
    if ratio <= 0.0:
        return 0.0
    return math.exp(-(math.log(ratio) ** 2) / (2.0 * sigma ** 2))


def compute_summary_length_reward(
    summary: str,
    reference: str,
    sigma: float = SUMMARY_LENGTH_SIGMA,
    target_words: int | None = None,
) -> float:
    """Gaussiana sul log-rapporto delle parole: 1 a pari lunghezza, simmetrica
    tra summary troppo corti e troppo lunghi (meta' e doppio valgono uguale).

    ``target_words`` sostituisce la lunghezza del reference come bersaglio. Serve
    perche' quella e' imparabile solo in media: sul train ha std 28 parole e
    correla 0.28 con la durata del video (R^2 0.078), quindi il 32% dei video
    spinge il summary ad accorciarsi e il 68% ad allungarsi, e i due gradienti si
    cancellano. Con un bersaglio fisso tutti i gruppi spingono nella stessa
    direzione: 37 -> 45 parole vale +0.38 a sigma 0.20, contro +0.013 con il
    bersaglio per-video."""
    words = len(summary.split())
    reference_words = target_words if target_words is not None else len(reference.split())
    if words == 0 or reference_words <= 0:
        return 0.0
    return _log_ratio_gaussian(words / reference_words, sigma)


def compute_length_budget_reward(
    selected: int, budget: int, sigma: float = LENGTH_BUDGET_SIGMA
) -> float:
    """Stessa gaussiana di compute_summary_length_reward, ma sul rapporto
    frame_selezionati/budget: 1 a budget centrato, e penalizza simmetricamente sia
    la sovra-selezione sia il selezionare troppo poco (che una penalita' sul solo
    sforamento non coglierebbe)."""
    if selected <= 0 or budget <= 0:
        return 0.0
    return _log_ratio_gaussian(selected / budget, sigma)


def average_ranks(values: torch.Tensor) -> torch.Tensor:
    """Ranghi 1-based con i pari merito al rango medio (scipy.stats.rankdata)."""
    values = values.detach().double().flatten()
    sorted_values, order = torch.sort(values, stable=True)
    _, inverse, counts = torch.unique_consecutive(
        sorted_values, return_inverse=True, return_counts=True
    )
    counts = counts.double()
    mean_rank = counts.cumsum(0) - (counts - 1.0) / 2.0
    ranks = torch.empty_like(values)
    ranks[order] = mean_rank[inverse]
    return ranks


def compute_saliency_rank_reward(pred: torch.Tensor, consensus_ranks: torch.Tensor) -> float:
    """(rho+1)/2 con rho = Spearman fra selezione binaria e saliency media.

    Con una predizione binaria i suoi ranghi sono una funzione affine della
    maschera, quindi lo Spearman e' il Pearson fra maschera e ranghi della
    saliency. Maschera o saliency costanti valgono rho=0, come in evalVtV
    (NaN -> 0)."""
    mask = pred.detach().double().flatten()
    mask = mask - mask.mean()
    ranks = consensus_ranks - consensus_ranks.mean()
    denominator = torch.sqrt((mask * mask).sum() * (ranks * ranks).sum())
    if denominator.item() <= EPSILON:
        return 0.5
    rho = ((mask * ranks).sum() / denominator).clamp(-1.0, 1.0).item()
    return (rho + 1.0) / 2.0


def split_into_sentences(text: str) -> list[str]:
    """Sentence/claim splitter leggero, deterministico e senza dipendenze NLP."""
    claims = [part.strip() for part in re.split(r"(?<=[.!?])\s+|[\r\n]+", text)]
    return [claim for claim in claims if claim]


def default_event_mask_converter(vsum: Any, count: int) -> torch.Tensor:
    """Converte intervalli temporali VideoXum in maschere sulla griglia a 1 fps.

    Un frame campionato all'indice ``t`` rappresenta la posizione temporale ``t``;
    viene incluso se tale posizione cade nell'intervallo annotato inclusivo. Il
    converter e' sostituibile con quello usato nella preparazione locale dei dati.
    """
    intervals = torch.as_tensor(vsum, dtype=torch.float64)
    if intervals.ndim != 3 or intervals.shape[1:] != (10, 2):
        raise ValueError("vsum deve avere shape [J, 10, 2]")
    if not torch.isfinite(intervals).all():
        raise ValueError("vsum contiene valori non finiti")
    if (intervals[..., 0] > intervals[..., 1]).any():
        raise ValueError("vsum contiene intervalli invertiti")
    positions = torch.arange(count, dtype=torch.float64).view(1, 1, count)
    return (
        (positions >= intervals[..., 0].unsqueeze(-1))
        & (positions <= intervals[..., 1].unsqueeze(-1))
    )


def _as_normalized_embeddings(values: Any, expected_rows: int) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32).detach()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2 or tensor.shape[0] != expected_rows:
        raise ValueError(
            f"encoder semantico: atteso [{expected_rows}, D], ottenuto {tuple(tensor.shape)}"
        )
    return F.normalize(tensor, dim=-1, eps=EPSILON)


class VideoSummaryReward:
    """Reward engine con cache per-video e solo inferenza/no-grad."""

    def __init__(
        self,
        semantic_encoder: SemanticEncoder,
        *,
        event_mask_converter: EventMaskConverter | None = None,
        reference_metric: ReferenceMetric | None = None,
        epsilon: float = EPSILON,
        f_beta: float = 0.5,
        f_beta_aggregation: str = "max",
        weights: dict[str, float] | None = None,
        use_length_penalty: bool = False,
        summary_length_sigma: float = SUMMARY_LENGTH_SIGMA,
        summary_target_words: int | None = None,
        length_budget_sigma: float = LENGTH_BUDGET_SIGMA,
        reference_cosine_floor: float = REFERENCE_COSINE_FLOOR,
        reference_cosine_scale: float = REFERENCE_COSINE_SCALE,
        timeline_unit: str = "frame",
    ) -> None:
        if timeline_unit not in ("frame", "pair"):
            raise ValueError(f"timeline_unit non valida: {timeline_unit}")
        # Unita' degli indici nella completion; la cache e le componenti restano
        # sempre sui frame (vedi expand_pair_completion).
        self.timeline_unit = timeline_unit
        self.semantic_encoder = semantic_encoder
        self.weights = {**DEFAULT_REWARD_WEIGHTS, **(weights or {})}
        if unknown := set(self.weights) - set(DEFAULT_REWARD_WEIGHTS):
            raise ValueError(f"pesi reward sconosciuti: {sorted(unknown)}")
        if any(value < 0 for value in self.weights.values()):
            raise ValueError("i pesi della reward devono essere >= 0")
        if not any(self.weights.values()):
            raise ValueError("almeno un peso della reward deve essere > 0")
        if f_beta_aggregation not in F_BETA_AGGREGATIONS:
            raise ValueError(
                f"f_beta_aggregation non valida: {f_beta_aggregation} "
                f"(usare una fra {F_BETA_AGGREGATIONS})"
            )
        self.f_beta_aggregation = f_beta_aggregation
        # Moltiplicatore legacy sul budget di frame: spento di default, sostituito
        # dalla componente additiva "length_budget". Riaccenderlo serve solo a
        # riprodurre i run precedenti (insieme a w_length_budget=0).
        self.use_length_penalty = bool(use_length_penalty)
        if summary_length_sigma <= 0:
            raise ValueError("summary_length_sigma deve essere positivo")
        self.summary_length_sigma = float(summary_length_sigma)
        if summary_target_words is not None and summary_target_words <= 0:
            raise ValueError("summary_target_words deve essere positivo")
        self.summary_target_words = (
            int(summary_target_words) if summary_target_words is not None else None
        )
        if length_budget_sigma <= 0:
            raise ValueError("length_budget_sigma deve essere positivo")
        self.length_budget_sigma = float(length_budget_sigma)
        if reference_cosine_scale <= 0:
            raise ValueError("reference_cosine_scale deve essere positivo")
        self.reference_cosine_floor = float(reference_cosine_floor)
        self.reference_cosine_scale = float(reference_cosine_scale)
        self.event_mask_converter = event_mask_converter or default_event_mask_converter
        self.reference_metric = reference_metric
        self.epsilon = float(epsilon)
        if f_beta <= 0:
            raise ValueError("f_beta deve essere positivo")
        # beta<1 pesa la precisione piu' della recall: selezionare frame extra
        # che non colpiscono gli intervalli annotati costa piu' di quanto valga
        # l'eventuale recall guadagnata, cosi' il reward non premia piu' selezioni
        # ampie di per se' (vedi discussione F1 -> F-beta).
        self.f_beta = float(f_beta)
        self._cache: dict[Hashable, VideoRewardCache] = {}

    def clear_cache(self) -> None:
        self._cache.clear()

    def has_video(self, cache_key: Hashable) -> bool:
        return cache_key in self._cache

    @torch.no_grad()
    def prepare_video(
        self,
        cache_key: Hashable,
        *,
        count: int,
        vsum_onehot: Any,
        vsum: Any,
        tsum: Sequence[str],
        frame_features: Any,
    ) -> VideoRewardCache:
        """Valida e precalcola tutti i dati invarianti di un video."""
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if cached.count != count:
                raise ValueError("cache_key riutilizzata per un video con count diverso")
            return cached
        if count <= 0:
            raise ValueError("count deve essere positivo")

        gt = torch.as_tensor(vsum_onehot).bool().detach().cpu()
        if gt.shape != (10, count):
            raise ValueError(f"vsum_onehot deve avere shape [10, {count}]")
        captions = tuple(str(item).strip() for item in tsum)
        if not captions or any(not item for item in captions):
            raise ValueError("tsum deve contenere caption non vuote")

        event_masks = torch.as_tensor(
            self.event_mask_converter(vsum, count), dtype=torch.bool
        ).detach().cpu()
        if event_masks.shape != (len(captions), 10, count):
            raise ValueError(
                f"event_masks deve avere shape [{len(captions)}, 10, {count}]"
            )

        visual = torch.as_tensor(frame_features, dtype=torch.float32).detach().cpu()
        if visual.ndim != 2 or visual.shape[0] != count:
            raise ValueError(f"frame_features deve avere shape [{count}, D]")
        visual = F.normalize(visual, dim=-1, eps=self.epsilon)

        # Baseline di ridondanza del video: quanto sono simili tra loro due frame
        # qualsiasi, in media e nel caso migliore. I frame di uno stesso video sono
        # spesso gia' molto simili anche senza alcuna selezione mirata: senza
        # calibrare rispetto a questa baseline, representativeness e diversity
        # saturano rispettivamente vicino a 1 e vicino a 0 indipendentemente dalla
        # qualita' reale della selezione (vedi commenti in score()).
        if count >= 2:
            frame_sim_matrix = visual @ visual.T
            off_diag = frame_sim_matrix[~torch.eye(count, dtype=torch.bool)]
            frame_floor_sim = off_diag.mean().item()
            frame_min_sim = off_diag.min().item()
        else:
            frame_floor_sim, frame_min_sim = 0.0, -1.0

        reference_summary = " ".join(captions)
        encoded = self.semantic_encoder.encode([*captions, reference_summary])
        encoded = _as_normalized_embeddings(encoded, len(captions) + 1).cpu()
        # Budget legato alla ricchezza del reference (parole totali delle caption),
        # non a una frazione fissa della timeline: un video con un reference
        # sintetico deve restare compatto, uno con un reference ricco puo'
        # coprire piu' frame. 1.3x lascia un margine sopra la lunghezza "attesa".
        # Il floor al 20% del video e' un vincolo di osservabilita': il prompt
        # annuncia al modello un target del ~20% (15-25%), mentre la lunghezza del
        # reference non e' visibile al modello. Senza il floor un reference corto
        # produrrebbe un budget piu' stretto di quello annunciato, penalizzando il
        # modello per un'informazione che non puo' vedere; con il floor un
        # reference ricco concede solo headroom in piu'.
        max_selected = max(1, min(count, max(
            math.ceil(0.20 * count),
            math.ceil(1.3 * len(reference_summary.split())),
        )))
        cache = VideoRewardCache(
            count=count,
            max_selected=max_selected,
            vsum_onehot=gt,
            event_masks=event_masks,
            frame_features=visual,
            caption_embeddings=encoded[:-1],
            reference_embedding=encoded[-1],
            consensus=gt.float().mean(dim=0),
            captions=captions,
            reference_summary=reference_summary,
            frame_floor_sim=frame_floor_sim,
            frame_min_sim=frame_min_sim,
            consensus_ranks=average_ranks(gt.float().mean(dim=0)),
        )
        self._cache[cache_key] = cache
        return cache

    def timeline_limits(self, data: VideoRewardCache) -> tuple[int, int]:
        """(numero di indici, budget di selezione) nell'unita' della completion."""
        if self.timeline_unit == "pair":
            return (
                pair_timeline_length(data.count),
                max(1, math.ceil(data.max_selected / 2)),
            )
        return data.count, data.max_selected

    @staticmethod
    def _invalid_result(format_reward: float) -> dict[str, Any]:
        return {
            "total_reward": float(format_reward - 1.0),
            "total_reward_default": float(format_reward - 1.0),
            "format_reward": float(format_reward),
            "f_beta_reward": 0.0,
            "diversity_reward": 0.0,
            "representativeness_reward": 0.0,
            "grounding_reward": 0.0,
            "grounded_coverage": 0.0,
            "grounded_precision": 0.0,
            "reference_reward": 0.0,
            "summary_length_reward": 0.0,
            "length_budget_reward": 0.0,
            "saliency_rank_reward": 0.0,
            "summary_words": 0,
            "selected_count": 0,
            "selected_ratio": 0.0,
            "selection_budget": 0,
            "length_penalty": 0.0,
            "valid": False,
            "merged_adjacent_count": 0,
            "f_beta_max": 0.0,
            "f_beta_min": 0.0,
            "f_beta_std": 0.0,
            "f_beta_per_annotator": [],
        }

    @torch.no_grad()
    def score(self, completion: str, cache_key: Hashable) -> dict[str, Any]:
        """Calcola reward totale, componenti e diagnostiche per un rollout."""
        if cache_key not in self._cache:
            raise KeyError(f"video {cache_key!r} non presente nella cache")
        data = self._cache[cache_key]
        timeline_count, timeline_budget = self.timeline_limits(data)
        #reward format dell'output
        format_reward = compute_format_reward(
            completion, timeline_count, timeline_budget
        )
        try:
            # parsing e validazione rigida della completion JSON
            parsed = parse_and_validate_completion(
                completion, timeline_count, timeline_budget
            )
        except CompletionValidationError:
            return self._invalid_result(format_reward)
        if self.timeline_unit == "pair":
            parsed = expand_pair_completion(parsed, data.count)

        if parsed.merged_adjacent_count:
            # Penalita' graduale invece di rigetto totale: l'errore e' puramente
            # sintattico (i frame selezionati non cambiano), ma va comunque
            # disincentivato senza azzerare f1/grounding/reference dell'intero
            # completion (vedi discussione sul reward -0.25 sistematico quando
            # tutto il gruppo GRPO violava questa regola allo stesso modo).
            format_reward = max(0.0, format_reward - 0.1 * parsed.merged_adjacent_count)

        pred = parsed.pred_mask.cpu()
        pred_count = int(pred.sum().item())

        beta2 = self.f_beta ** 2
        intersections = (data.vsum_onehot & pred.unsqueeze(0)).sum(dim=1).float()
        gt_counts = data.vsum_onehot.sum(dim=1).float()
        # F-beta = (1+beta^2)*TP / (beta^2*GT + pred): forma chiusa equivalente a
        # (1+beta^2)*P*R/(beta^2*P+R) ma senza dividere esplicitamente per
        # precision/recall (beta=1 riduce alla vecchia formula F1).
        f_beta_denominators = beta2 * gt_counts + pred_count
        f_beta_values = torch.where(
            f_beta_denominators > 0,
            (1.0 + beta2) * intersections / f_beta_denominators,
            0.0,
        )
        #reward f-beta medio su tutti gli annotatori, normalizzato in [0, 1]: premia la precisione della selezione, non solo la copertura
        # max sui 10 annotatori invece della media: gli annotatori non concordano
        # fra loro, e mediare appiattisce le differenze fra rollout (std intra-gruppo
        # 0.024 contro 0.042 del max). E' anche l'aggregazione usata dalla metrica di
        # validation (F1_MAX): con la media si ottimizza una quantita' diversa da
        # quella che poi si riporta.
        f_beta_reward = (
            f_beta_values.max() if self.f_beta_aggregation == "max"
            else f_beta_values.mean()
        ).item()

        prototypes = []
        for start, end in parsed.selected_frames:
            prototype = data.frame_features[start:end + 1].mean(dim=0)
            prototypes.append(F.normalize(prototype, dim=0, eps=self.epsilon))
        if len(prototypes) < 2:
            diversity_reward = 0.0
        else:
            group_features = torch.stack(prototypes)
            similarity = group_features @ group_features.T
            pairs = torch.triu_indices(len(prototypes), len(prototypes), offset=1)
            raw_pair_sim = similarity[pairs[0], pairs[1]]
            # Calibrata sulla ridondanza di base del video (vedi frame_floor_sim in
            # prepare_video): 0 se due segmenti sono simili quanto una coppia
            # tipica dello stesso video, 1 se simili quanto la coppia piu'
            # dissimile osservabile. Senza questa calibrazione la similarita' di
            # base gia' alta tra frame dello stesso video schiaccia la reward
            # vicino al suo stesso soffitto (~0.02-0.05), qualunque selezione.
            diversity_denom = max(data.frame_floor_sim - data.frame_min_sim, 0.05)
            diversity_reward = (
                (data.frame_floor_sim - raw_pair_sim) / diversity_denom
            ).clamp(0.0, 1.0).mean().item()

        selected_features = data.frame_features[parsed.selected_indices]
        raw_coverage = (data.frame_features @ selected_features.T).max(dim=1).values
        # Calibrata sulla stessa baseline di ridondanza: 0 se il frame selezionato
        # piu' vicino non e' piu' simile del "rumore di fondo" tipico del video, 1
        # solo per un match (quasi) esatto. Senza questa calibrazione quasi ogni
        # selezione sparsa sulla timeline ottiene copertura vicina a 1, perche' la
        # similarita' di base tra frame dello stesso video e' gia' molto alta.
        representativeness_denom = max(1.0 - data.frame_floor_sim, 0.05)
        coverage = (
            (raw_coverage - data.frame_floor_sim) / representativeness_denom
        ).clamp(0.0, 1.0)
        representativeness_reward = (
            (data.consensus * coverage).sum()
            / (data.consensus.sum() + self.epsilon)
        ).item()

        event_intersections = (
            data.event_masks & pred.view(1, 1, -1)
        ).sum(dim=-1).float()
        event_sizes = data.event_masks.sum(dim=-1).float()
        visual_support = (
            event_intersections / (event_sizes + self.epsilon)
        ).mean(dim=1)

        claims = split_into_sentences(parsed.summary)
        online_texts = [parsed.summary] if claims == [parsed.summary] else [*claims, parsed.summary]
        online = _as_normalized_embeddings(
            self.semantic_encoder.encode(online_texts), len(online_texts)
        ).cpu()
        claim_embeddings = online[:len(claims)]
        summary_embedding = online[-1]
        semantic_similarity = (
            (data.caption_embeddings @ claim_embeddings.T + 1.0) / 2.0
        ).clamp(0.0, 1.0)
        text_match = semantic_similarity.max(dim=1).values
        grounded_coverage = (visual_support * text_match).mean().item()
        grounded_precision = (
            visual_support.unsqueeze(1) * semantic_similarity
        ).max(dim=0).values.mean().item()
        grounding_reward = min(1.0, max(0.0, (
            2.0 * grounded_coverage * grounded_precision
            / (grounded_coverage + grounded_precision + self.epsilon)
        )))

        if self.reference_metric is None:
            # Riscalatura affine del coseno (embedding gia' normalizzati, quindi
            # il prodotto scalare e' il coseno): costo zero, nessuna ri-encoding.
            reference_reward = float(
                (
                    (summary_embedding @ data.reference_embedding
                     - self.reference_cosine_floor)
                    / self.reference_cosine_scale
                )
                .clamp(0.0, 1.0)
                .item()
            )
        else:
            reference_reward = float(self.reference_metric(
                parsed.summary, data.reference_summary, self.semantic_encoder
            ))
            reference_reward = min(1.0, max(0.0, reference_reward))

        summary_length_reward = compute_summary_length_reward(
            parsed.summary,
            data.reference_summary,
            self.summary_length_sigma,
            self.summary_target_words,
        )

        selection_ratio = pred_count / data.max_selected
        # Il budget di frame entra nella somma pesata come qualunque altra
        # componente, con la stessa forma gaussiana gia' usata per la lunghezza del
        # testo: il vincolo riguarda la selezione, quindi non deve scalare anche
        # grounding/reference/summary_length. Cosi' il termine e' limitato dal suo
        # peso e non puo' dominare l'advantage, e la simmetria della gaussiana
        # penalizza anche il selezionare troppo poco.
        length_budget_reward = compute_length_budget_reward(
            pred_count, data.max_selected, self.length_budget_sigma
        )
        # Moltiplicatore legacy: calcolato sempre come diagnostica, applicato solo
        # con use_length_penalty=True. Cappato a 0.85, quindi piatto oltre
        # ratio=1.30: li' l'intero gruppo GRPO collassava sullo stesso valore e
        # l'advantage si annullava (28% dei gruppi nel run di riferimento).
        length_penalty = min(0.85, (selection_ratio ** 2) / 2.0)
        saliency_rank_reward = compute_saliency_rank_reward(pred, data.consensus_ranks)

        components = {
            "f_beta": f_beta_reward,
            "diversity": diversity_reward,
            "representativeness": representativeness_reward,
            "grounding": grounding_reward,
            "reference": reference_reward,
            "format": format_reward,
            "summary_length": summary_length_reward,
            "length_budget": length_budget_reward,
            "saliency_rank": saliency_rank_reward,
        }
        # total_reward usa i pesi dell'esperimento e guida l'advantage GRPO;
        # total_reward_default usa sempre la configurazione completa, cosi' la
        # validation resta confrontabile tra ablation diverse. Entrambi normalizzati
        # per la somma dei pesi: i pesi non sommano piu' a 1, e senza normalizzare
        # il clamp finale e la scala dipenderebbero dall'ablation scelta.
        weight_sum = sum(self.weights.values()) or 1.0
        default_sum = sum(DEFAULT_REWARD_WEIGHTS.values()) or 1.0
        weighted_reward = sum(self.weights[k] * v for k, v in components.items()) / weight_sum
        default_reward = sum(
            DEFAULT_REWARD_WEIGHTS[k] * v for k, v in components.items()
        ) / default_sum
        applied_penalty = length_penalty if self.use_length_penalty else 0.0
        total_reward = weighted_reward * (1.0 - applied_penalty)
        total_reward_default = default_reward * (1.0 - applied_penalty)
        return {
            "total_reward": float(min(1.0, max(0.0, total_reward))),
            "total_reward_default": float(min(1.0, max(0.0, total_reward_default))),
            "format_reward": float(format_reward),
            "f_beta_reward": float(f_beta_reward),
            "diversity_reward": float(diversity_reward),
            "representativeness_reward": float(representativeness_reward),
            "grounding_reward": float(grounding_reward),
            "grounded_coverage": float(grounded_coverage),
            "grounded_precision": float(grounded_precision),
            "reference_reward": float(reference_reward),
            "summary_length_reward": float(summary_length_reward),
            "length_budget_reward": float(length_budget_reward),
            "saliency_rank_reward": float(saliency_rank_reward),
            "summary_words": len(parsed.summary.split()),
            "reference_words": len(data.reference_summary.split()),
            "selected_count": pred_count,
            "selected_ratio": pred_count / data.count,
            "selection_budget": data.max_selected,
            "length_penalty": float(length_penalty),
            "valid": True,
            "merged_adjacent_count": parsed.merged_adjacent_count,
            "f_beta_max": float(f_beta_values.max().item()),
            "f_beta_min": float(f_beta_values.min().item()),
            "f_beta_std": float(f_beta_values.std(unbiased=False).item()),
            "f_beta_per_annotator": [float(value) for value in f_beta_values.tolist()],
        }

    @torch.no_grad()
    def prepare_and_score(
        self,
        completion: str,
        *,
        cache_key: Hashable,
        count: int,
        vsum_onehot: Any,
        vsum: Any,
        tsum: Sequence[str],
        frame_features: Any,
    ) -> dict[str, Any]:
        """API principale comoda: prepara al primo rollout e poi riusa la cache."""
        self.prepare_video(
            cache_key,
            count=count,
            vsum_onehot=vsum_onehot,
            vsum=vsum,
            tsum=tsum,
            frame_features=frame_features,
        )
        return self.score(completion, cache_key)


def compute_video_summary_reward(
    reward_engine: VideoSummaryReward,
    *,
    completion: str,
    cache_key: Hashable,
    count: int,
    vsum_onehot: Any,
    vsum: Any,
    tsum: Sequence[str],
    frame_features: Any,
) -> dict[str, Any]:
    """Funzione principale stateless nell'uso, con cache posseduta dall'engine."""
    return reward_engine.prepare_and_score(
        completion,
        cache_key=cache_key,
        count=count,
        vsum_onehot=vsum_onehot,
        vsum=vsum,
        tsum=tsum,
        frame_features=frame_features,
    )

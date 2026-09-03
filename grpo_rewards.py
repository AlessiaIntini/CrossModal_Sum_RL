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


class CompletionValidationError(ValueError):
    """Completion JSON non conforme al contratto richiesto."""


@dataclass(frozen=True)
class ParsedCompletion:
    selected_frames: list[list[int]]
    summary: str
    pred_mask: torch.Tensor
    selected_indices: list[int]


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
        # json.loads richiede che non esistano Markdown o spiegazioni fuori dal JSON.
        return _strict_json_loads(completion)
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
            raise CompletionValidationError("gli intervalli adiacenti devono essere uniti")
        clean_intervals.append([start, end])
        selected_indices.extend(range(start, end + 1))
        previous_end = end

    if len(selected_indices) != len(set(selected_indices)):
        raise CompletionValidationError("uno stesso frame e' selezionato piu' volte")
    if len(selected_indices) > max_selected:
        raise CompletionValidationError(
            f"selezionati {len(selected_indices)} frame, massimo consentito {max_selected}"
        )

    summary = payload["summary"]
    if not isinstance(summary, str) or not summary.strip():
        raise CompletionValidationError("summary deve essere una stringa non vuota")

    pred_mask = torch.zeros(count, dtype=torch.bool)
    pred_mask[selected_indices] = True
    return ParsedCompletion(
        selected_frames=clean_intervals,
        summary=summary.strip(),
        pred_mask=pred_mask,
        selected_indices=selected_indices,
    )


def compute_format_reward(completion: str, count: int, max_selected: int) -> float:
    """Credito graduale a quattro livelli, indipendente dalla validita' finale."""
    try:
        payload = _strict_json_loads(completion) if isinstance(completion, str) else None
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
                or start <= previous_end + 1
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
    ) -> None:
        self.semantic_encoder = semantic_encoder
        self.event_mask_converter = event_mask_converter or default_event_mask_converter
        self.reference_metric = reference_metric
        self.epsilon = float(epsilon)
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

        reference_summary = " ".join(captions)
        encoded = self.semantic_encoder.encode([*captions, reference_summary])
        encoded = _as_normalized_embeddings(encoded, len(captions) + 1).cpu()
        cache = VideoRewardCache(
            count=count,
            max_selected=max(1, math.ceil(0.20 * count)),
            vsum_onehot=gt,
            event_masks=event_masks,
            frame_features=visual,
            caption_embeddings=encoded[:-1],
            reference_embedding=encoded[-1],
            consensus=gt.float().mean(dim=0),
            captions=captions,
            reference_summary=reference_summary,
        )
        self._cache[cache_key] = cache
        return cache

    @staticmethod
    def _invalid_result(format_reward: float) -> dict[str, Any]:
        return {
            "total_reward": float(format_reward - 1.0),
            "format_reward": float(format_reward),
            "f1_reward": 0.0,
            "diversity_reward": 0.0,
            "representativeness_reward": 0.0,
            "grounding_reward": 0.0,
            "grounded_coverage": 0.0,
            "grounded_precision": 0.0,
            "reference_reward": 0.0,
            "selected_count": 0,
            "selected_ratio": 0.0,
            "valid": False,
            "f1_max": 0.0,
            "f1_min": 0.0,
            "f1_std": 0.0,
            "f1_per_annotator": [],
        }

    @torch.no_grad()
    def score(self, completion: str, cache_key: Hashable) -> dict[str, Any]:
        """Calcola reward totale, componenti e diagnostiche per un rollout."""
        if cache_key not in self._cache:
            raise KeyError(f"video {cache_key!r} non presente nella cache")
        data = self._cache[cache_key]
        #reward format dell'output
        format_reward = compute_format_reward(
            completion, data.count, data.max_selected
        )
        try:
            # parsing e validazione rigida della completion JSON
            parsed = parse_and_validate_completion(
                completion, data.count, data.max_selected
            )
        except CompletionValidationError:
            return self._invalid_result(format_reward)

        pred = parsed.pred_mask.cpu()
        pred_count = int(pred.sum().item())

        intersections = (data.vsum_onehot & pred.unsqueeze(0)).sum(dim=1).float()
        denominators = pred_count + data.vsum_onehot.sum(dim=1).float()
        f1_values = torch.where(
            denominators > 0, 2.0 * intersections / denominators, 0.0
        )
        #reward f1 medio su tutti gli annotatori, normalizzato in [0, 1], vede se i frame selezionati coprono bene gli intervalli annotati
        f1_reward = f1_values.mean().item()

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
            diversity_reward = (
                (1.0 - similarity[pairs[0], pairs[1]]) / 2.0
            ).clamp(0.0, 1.0).mean().item()

        selected_features = data.frame_features[parsed.selected_indices]
        coverage = (data.frame_features @ selected_features.T).max(dim=1).values
        coverage = ((coverage + 1.0) / 2.0).clamp(0.0, 1.0)
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
            reference_reward = float(
                ((summary_embedding @ data.reference_embedding + 1.0) / 2.0)
                .clamp(0.0, 1.0)
                .item()
            )
        else:
            reference_reward = float(self.reference_metric(
                parsed.summary, data.reference_summary, self.semantic_encoder
            ))
            reference_reward = min(1.0, max(0.0, reference_reward))

        total_reward = (
            0.25 * f1_reward
            + 0.10 * diversity_reward
            + 0.10 * representativeness_reward
            + 0.30 * grounding_reward
            + 0.15 * reference_reward
            + 0.10 * format_reward
        )
        return {
            "total_reward": float(min(1.0, max(0.0, total_reward))),
            "format_reward": float(format_reward),
            "f1_reward": float(f1_reward),
            "diversity_reward": float(diversity_reward),
            "representativeness_reward": float(representativeness_reward),
            "grounding_reward": float(grounding_reward),
            "grounded_coverage": float(grounded_coverage),
            "grounded_precision": float(grounded_precision),
            "reference_reward": float(reference_reward),
            "selected_count": pred_count,
            "selected_ratio": pred_count / data.count,
            "valid": True,
            "f1_max": float(f1_values.max().item()),
            "f1_min": float(f1_values.min().item()),
            "f1_std": float(f1_values.std(unbiased=False).item()),
            "f1_per_annotator": [float(value) for value in f1_values.tolist()],
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

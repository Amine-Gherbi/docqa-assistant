"""Reranking behind one async interface.

The intended backend is the ``Xenova/ms-marco-MiniLM-L-6-v2`` cross-encoder
(Apache-2.0, ONNX, CPU). fastembed 0.9.0 — the version pinned here — exposes no
cross-encoder class, so on this machine :func:`build_reranker` falls back to a
deterministic lexical-overlap scorer. The fallback is never silent: it is named
``dummy:lexical-overlap``, reported by the ablation harness, and recorded in
``docs/ENV.md``.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import anyio

from docqa.onnx import onnx_semaphore

DEFAULT_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

# Function words carry no passage-specific signal for an overlap score.
_STOPWORDS = frozenset(
    [
        "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "can",
        "could", "did", "do", "does", "for", "from", "had", "has", "have", "how",
        "in", "into", "is", "it", "its", "may", "might", "of", "on", "or", "should",
        "than", "that", "the", "their", "them", "then", "there", "these", "they",
        "this", "those", "to", "was", "were", "what", "when", "where", "which",
        "who", "why", "will", "with", "would",
    ]
)


@runtime_checkable
class Reranker(Protocol):
    """Scores (query, passage) pairs; higher means more relevant."""

    name: str
    is_dummy: bool

    async def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Return one finite score per passage, preserving order."""
        ...


@dataclass(frozen=True)
class RerankerChoice:
    """The reranker actually selected, plus why."""

    reranker: Reranker
    requested: str
    fallback_reason: str | None

    @property
    def banner(self) -> str:
        """Human-readable line printed before use."""
        if self.reranker.is_dummy:
            return (
                f"WARNING: dummy reranker ({self.reranker.name}) — deterministic "
                "lexical overlap, NOT cross-encoder logits."
            )
        return f"Reranker: {self.reranker.name} (ONNX CPU)"


def _terms(text: str) -> list[str]:
    """Lower-case content terms, dropping function words."""
    return [term for term in _TOKEN_RE.findall(text.lower()) if term not in _STOPWORDS]


def _overlap_score(query_terms: Sequence[str], passage: str) -> float:
    """Deterministic relevance in roughly [0, 1.5]; higher is better."""
    unique = set(query_terms)
    if not unique:
        return 0.0
    counts = Counter(_terms(passage))
    if not counts:
        return 0.0
    covered = sum(1 for term in unique if term in counts)
    coverage = covered / len(unique)
    frequency = sum(min(counts.get(term, 0), 3) for term in unique)
    density = frequency / math.sqrt(len(counts))
    return coverage + 0.05 * density


class LexicalOverlapReranker:
    """Deterministic stand-in for a cross-encoder.

    It rewards query-term coverage and capped term frequency, length-normalised
    by the square root of the passage vocabulary. It is labelled ``dummy`` and
    must not be described as a learned reranker.
    """

    name = "dummy:lexical-overlap"
    is_dummy = True

    async def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Score passages by query-term overlap."""
        query_terms = _terms(query)
        return [_overlap_score(query_terms, passage) for passage in passages]


class CrossEncoderReranker:
    """Xenova/ms-marco-MiniLM-L-6-v2 through fastembed's ONNX runtime."""

    def __init__(
        self, model_name: str, cache_dir: str | None = None, threads: int | None = None
    ) -> None:
        """Load the cross-encoder; raises when this fastembed build cannot."""
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.name = f"fastembed:{model_name}"
        self.is_dummy = False
        kwargs: dict[str, object] = {"model_name": model_name}
        if cache_dir is not None:
            kwargs["cache_dir"] = cache_dir
        if threads is not None:
            kwargs["threads"] = threads
        self._model = TextCrossEncoder(**kwargs)

    def _score_sync(self, query: str, passages: list[str]) -> list[float]:
        scores = [float(value) for value in self._model.rerank(query, passages)]
        if not all(math.isfinite(value) for value in scores):
            raise ValueError("reranker produced a non-finite score")
        return scores

    async def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Run ONNX inference in a worker thread, at most two calls at once."""
        if not passages:
            return []
        async with onnx_semaphore():
            return await anyio.to_thread.run_sync(
                self._score_sync, query, list(passages)
            )


def build_reranker(
    *,
    mode: str = "auto",
    model_name: str = DEFAULT_RERANK_MODEL,
    cache_dir: str | None = None,
    threads: int | None = None,
) -> RerankerChoice:
    """Select the configured reranker, falling back only in ``auto``."""
    if mode not in {"auto", "fastembed", "lexical"}:
        raise ValueError("reranker mode must be one of: auto, fastembed, lexical")
    if mode == "lexical":
        return RerankerChoice(LexicalOverlapReranker(), requested=mode, fallback_reason=None)
    try:
        reranker = CrossEncoderReranker(model_name, cache_dir, threads=threads)
    except Exception as exc:  # Import, download and ONNX failures differ widely.
        if mode == "fastembed":
            raise
        return RerankerChoice(
            LexicalOverlapReranker(),
            requested=mode,
            fallback_reason=f"{type(exc).__name__}: {exc}",
        )
    return RerankerChoice(reranker, requested=mode, fallback_reason=None)

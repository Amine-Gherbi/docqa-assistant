"""Hybrid retrieval: vector top-20 union lexical top-20, RRF, optional rerank.

``retrieve`` is the only retrieval entry point. Callers never issue retrieval
SQL themselves; they depend on the :class:`CandidateSource` interface, which
:class:`PgCandidateSource` implements for PostgreSQL and the unit tests
implement in memory.

The lexical half ranks with PostgreSQL's ``ts_rank_cd`` (term frequency plus
cover density on a ``tsvector``). It is **not BM25** and carries no
``idf`` component.

Fusion uses reciprocal rank fusion with ``k = 60``: each list contributes
``1 / (k + rank)`` (rank is 1-based), and a chunk found by both paths keeps both
ranks and the sum of both contributions. The top ``rerank_limit`` fused chunks
are reranked, then the best ``k`` are returned.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from sqlalchemy import text
from sqlalchemy.engine import RowMapping
from sqlalchemy.ext.asyncio import AsyncSession

from docqa.config import EMBEDDING_DIM
from docqa.ingest.embedder import Embedder
from docqa.rerank import Reranker

RetrievalMode = Literal["vector", "hybrid", "hybrid_rerank"]

DEFAULT_VECTOR_LIMIT = 20
DEFAULT_LEXICAL_LIMIT = 20
DEFAULT_RRF_K = 60
DEFAULT_RERANK_LIMIT = 24

_VECTOR_SQL = text(
    """
    SELECT c.id AS chunk_id,
           c.chunk_index,
           d.doc_id,
           c.page_no,
           c.char_start,
           c.char_end,
           c.page_text_sha256,
           c.breadcrumb,
           c.text,
           1 - (c.embedding <=> CAST(CAST(:vec AS text) AS vector)) AS score
    FROM chunks AS c
    JOIN documents AS d ON d.id = c.document_id
    ORDER BY c.embedding <=> CAST(CAST(:vec AS text) AS vector), c.id
    LIMIT :limit
    """
)

_LEXICAL_SQL = text(
    """
    SELECT c.id AS chunk_id,
           c.chunk_index,
           d.doc_id,
           c.page_no,
           c.char_start,
           c.char_end,
           c.page_text_sha256,
           c.breadcrumb,
           c.text,
           ts_rank_cd(c.tsv, plainto_tsquery('english', :query)) AS score
    FROM chunks AS c
    JOIN documents AS d ON d.id = c.document_id
    WHERE c.tsv @@ plainto_tsquery('english', :query)
    ORDER BY score DESC, c.id
    LIMIT :limit
    """
)


@dataclass(frozen=True)
class ScoredChunk:
    """A chunk plus the score assigned by one retrieval path."""

    chunk_id: int
    chunk_index: int
    doc_id: str
    page_no: int
    char_start: int
    char_end: int
    page_text_sha256: str
    breadcrumb: str
    text: str
    score: float


@dataclass(frozen=True)
class FusedChunk:
    """A chunk after RRF, retaining each path's rank."""

    chunk: ScoredChunk
    vector_rank: int | None
    lexical_rank: int | None
    rrf_score: float


@dataclass(frozen=True)
class RetrievedChunk:
    """One chunk in final answer context order."""

    chunk_id: int
    chunk_index: int
    doc_id: str
    page_no: int
    char_start: int
    char_end: int
    page_text_sha256: str
    breadcrumb: str
    text: str
    vector_rank: int | None
    lexical_rank: int | None
    rrf_score: float
    rerank_score: float | None
    final_rank: int


@runtime_checkable
class CandidateSource(Protocol):
    """The two candidate lists fusion consumes."""

    async def vector_candidates(
        self, embedding: Sequence[float], limit: int
    ) -> list[ScoredChunk]:
        """Return the ``limit`` chunks nearest to ``embedding``."""
        ...

    async def lexical_candidates(self, query: str, limit: int) -> list[ScoredChunk]:
        """Return the ``limit`` chunks with the highest lexical rank."""
        ...


class PgCandidateSource:
    """Candidate lists from PostgreSQL + pgvector."""

    def __init__(self, session: AsyncSession) -> None:
        """Bind the session used for both reads."""
        self._session = session

    async def vector_candidates(
        self, embedding: Sequence[float], limit: int
    ) -> list[ScoredChunk]:
        """Nearest chunks by cosine distance, best first."""
        if limit <= 0:
            return []
        result = await self._session.execute(
            _VECTOR_SQL, {"vec": _vector_literal(embedding), "limit": limit}
        )
        return [_scored_chunk(row) for row in result.mappings()]

    async def lexical_candidates(self, query: str, limit: int) -> list[ScoredChunk]:
        """Chunks matching the query's lexemes, ranked by ``ts_rank_cd``."""
        if limit <= 0 or not query.strip():
            return []
        result = await self._session.execute(
            _LEXICAL_SQL, {"query": query, "limit": limit}
        )
        return [_scored_chunk(row) for row in result.mappings()]


def _vector_literal(embedding: Sequence[float]) -> str:
    """Render a pgvector text literal, refusing bad vectors early."""
    if len(embedding) != EMBEDDING_DIM:
        raise ValueError(
            f"expected a {EMBEDDING_DIM}-d query embedding, got {len(embedding)}"
        )
    values = [float(value) for value in embedding]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("query embedding contained a non-finite value")
    return "[" + ",".join(repr(value) for value in values) + "]"


def _scored_chunk(row: RowMapping) -> ScoredChunk:
    """Map one SQL row onto a :class:`ScoredChunk`."""
    return ScoredChunk(
        chunk_id=int(row["chunk_id"]),
        chunk_index=int(row["chunk_index"]),
        doc_id=str(row["doc_id"]),
        page_no=int(row["page_no"]),
        char_start=int(row["char_start"]),
        char_end=int(row["char_end"]),
        page_text_sha256=str(row["page_text_sha256"]),
        breadcrumb=str(row["breadcrumb"]),
        text=str(row["text"]),
        score=float(row["score"]),
    )


@dataclass
class _Accumulator:
    """Mutable fusion state for one chunk id."""

    chunk: ScoredChunk
    vector_rank: int | None = None
    lexical_rank: int | None = None
    rrf_score: float = 0.0


def reciprocal_rank_fusion(
    vector_hits: Sequence[ScoredChunk],
    lexical_hits: Sequence[ScoredChunk],
    *,
    k: int = DEFAULT_RRF_K,
) -> list[FusedChunk]:
    """Fuse two ranked lists, deduplicating by chunk id and keeping both ranks."""
    if k < 1:
        raise ValueError("RRF k must be at least 1")
    accumulated: dict[int, _Accumulator] = {}

    def add(hit: ScoredChunk, rank: int, *, lexical: bool) -> None:
        entry = accumulated.get(hit.chunk_id)
        if entry is None:
            entry = _Accumulator(hit)
            accumulated[hit.chunk_id] = entry
        if lexical:
            if entry.lexical_rank is not None:
                return  # Keep the best (first) rank for a repeated chunk.
            entry.lexical_rank = rank
        else:
            if entry.vector_rank is not None:
                return
            entry.vector_rank = rank
        entry.rrf_score += 1.0 / (k + rank)

    for rank, hit in enumerate(vector_hits, start=1):
        add(hit, rank, lexical=False)
    for rank, hit in enumerate(lexical_hits, start=1):
        add(hit, rank, lexical=True)

    ordered = sorted(
        accumulated.values(), key=lambda entry: (-entry.rrf_score, entry.chunk.chunk_id)
    )
    return [
        FusedChunk(
            chunk=entry.chunk,
            vector_rank=entry.vector_rank,
            lexical_rank=entry.lexical_rank,
            rrf_score=entry.rrf_score,
        )
        for entry in ordered
    ]


def _retrieved(fused: FusedChunk, rerank_score: float | None, rank: int) -> RetrievedChunk:
    """Flatten a fused chunk into the public result type."""
    chunk = fused.chunk
    return RetrievedChunk(
        chunk_id=chunk.chunk_id,
        chunk_index=chunk.chunk_index,
        doc_id=chunk.doc_id,
        page_no=chunk.page_no,
        char_start=chunk.char_start,
        char_end=chunk.char_end,
        page_text_sha256=chunk.page_text_sha256,
        breadcrumb=chunk.breadcrumb,
        text=chunk.text,
        vector_rank=fused.vector_rank,
        lexical_rank=fused.lexical_rank,
        rrf_score=fused.rrf_score,
        rerank_score=rerank_score,
        final_rank=rank,
    )


async def retrieve(
    query: str,
    k: int = 8,
    *,
    embedder: Embedder,
    source: CandidateSource,
    reranker: Reranker,
    vector_limit: int = DEFAULT_VECTOR_LIMIT,
    lexical_limit: int = DEFAULT_LEXICAL_LIMIT,
    rrf_k: int = DEFAULT_RRF_K,
    rerank_limit: int = DEFAULT_RERANK_LIMIT,
    mode: RetrievalMode = "hybrid_rerank",
) -> list[RetrievedChunk]:
    """Return up to ``k`` chunks in final order with the full retrieval trace.

    ``mode`` exists for the ablation: ``"vector"`` skips the lexical path and
    fusion, ``"hybrid"`` stops after RRF, and ``"hybrid_rerank"`` is the
    production path.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    if vector_limit < 0 or lexical_limit < 0:
        raise ValueError("candidate limits must be non-negative")
    if rerank_limit < 1:
        raise ValueError("rerank_limit must be at least 1")

    vectors = await embedder.embed_query([query])
    if len(vectors) != 1:
        raise RuntimeError(f"embedder returned {len(vectors)} vectors for one query")
    query_vector = vectors[0]

    if mode == "vector":
        hits = await source.vector_candidates(query_vector, k)
        results = [
            _retrieved(
                FusedChunk(chunk=hit, vector_rank=rank, lexical_rank=None, rrf_score=0.0),
                rerank_score=None,
                rank=rank,
            )
            for rank, hit in enumerate(hits[:k], start=1)
        ]
        return results

    vector_hits = await source.vector_candidates(query_vector, vector_limit)
    lexical_hits = await source.lexical_candidates(query, lexical_limit)
    fused = reciprocal_rank_fusion(vector_hits, lexical_hits, k=rrf_k)

    if mode == "hybrid":
        return [
            _retrieved(entry, rerank_score=None, rank=rank)
            for rank, entry in enumerate(fused[:k], start=1)
        ]

    candidates = fused[:rerank_limit]
    if not candidates:
        return []
    rerank_scores = await reranker.score(query, [entry.chunk.text for entry in candidates])
    if len(rerank_scores) != len(candidates):
        raise RuntimeError(
            f"reranker returned {len(rerank_scores)} scores for {len(candidates)} chunks"
        )
    if not all(math.isfinite(value) for value in rerank_scores):
        raise RuntimeError("reranker returned a non-finite score")

    ranked = sorted(
        zip(candidates, rerank_scores, strict=True),
        key=lambda pair: (-pair[1], -pair[0].rrf_score, pair[0].chunk.chunk_id),
    )
    return [
        _retrieved(entry, rerank_score=score, rank=rank)
        for rank, (entry, score) in enumerate(ranked[:k], start=1)
    ]

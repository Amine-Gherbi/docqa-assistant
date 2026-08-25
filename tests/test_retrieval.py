"""Retrieval tests driven by a fake embedder; no network, no database."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace

import pytest

from docqa.ingest.tokens import HashTokenCounter
from docqa.rerank import LexicalOverlapReranker, build_reranker
from docqa.retrieval import (
    FusedChunk,
    ScoredChunk,
    reciprocal_rank_fusion,
    retrieve,
)

_WORD_RE = re.compile(r"[a-z0-9]+")


class FakeEmbedder:
    """Maps each exact input string to a pinned vector."""

    name = "fake:topic-vectors"
    is_dummy = True
    dim = 3

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors
        self.counter = HashTokenCounter()

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vectors[text] for text in texts]

    async def embed_query(self, texts: Sequence[str]) -> list[list[float]]:
        return await self.embed(texts)


class ConstantReranker:
    """Preserves RRF order by scoring every passage identically."""

    name = "fake:constant"
    is_dummy = True

    async def score(self, query: str, passages: Sequence[str]) -> list[float]:
        return [0.0 for _ in passages]


class FixedReranker:
    """Scores passages by exact text lookup."""

    name = "fake:fixed"
    is_dummy = True

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores

    async def score(self, query: str, passages: Sequence[str]) -> list[float]:
        return [self._scores[passage] for passage in passages]


class ShortReranker:
    """Returns one score too few, to exercise the contract check."""

    name = "fake:short"
    is_dummy = True

    async def score(self, query: str, passages: Sequence[str]) -> list[float]:
        return [0.0 for _ in passages[:-1]]


def _chunk(
    chunk_id: int,
    text: str,
    *,
    doc_id: str = "doc",
    chunk_index: int | None = None,
    page_no: int = 1,
) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        chunk_index=chunk_id if chunk_index is None else chunk_index,
        doc_id=doc_id,
        page_no=page_no,
        char_start=0,
        char_end=len(text),
        page_text_sha256="0" * 64,
        breadcrumb=f"{doc_id} | p.{page_no}",
        text=text,
        score=0.0,
    )


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norm if norm else 0.0


@dataclass(frozen=True)
class _Entry:
    chunk: ScoredChunk
    vector: tuple[float, ...]


class InMemorySource:
    """Implements the candidate interface with cosine and term matching."""

    def __init__(self, entries: Sequence[tuple[ScoredChunk, Sequence[float]]]) -> None:
        self._entries = [
            _Entry(chunk=chunk, vector=tuple(vector)) for chunk, vector in entries
        ]

    async def vector_candidates(
        self, embedding: Sequence[float], limit: int
    ) -> list[ScoredChunk]:
        scored = [
            replace(entry.chunk, score=_cosine(embedding, entry.vector))
            for entry in self._entries
        ]
        scored.sort(key=lambda chunk: (-chunk.score, chunk.chunk_id))
        return scored[:limit]

    async def lexical_candidates(self, query: str, limit: int) -> list[ScoredChunk]:
        query_terms = set(_WORD_RE.findall(query.lower()))
        scored: list[ScoredChunk] = []
        for entry in self._entries:
            terms = _WORD_RE.findall(entry.chunk.text.lower())
            matches = sum(1 for term in terms if term in query_terms)
            if matches:
                scored.append(replace(entry.chunk, score=float(matches)))
        scored.sort(key=lambda chunk: (-chunk.score, chunk.chunk_id))
        return scored[:limit]


def test_rrf_keeps_both_ranks_and_dedupes() -> None:
    a, b, c, d = (_chunk(number, f"text {number}") for number in range(1, 5))
    fused = reciprocal_rank_fusion([a, b, c], [c, a, d], k=60)

    by_id = {entry.chunk.chunk_id: entry for entry in fused}
    assert len(fused) == 4
    assert by_id[a.chunk_id].vector_rank == 1
    assert by_id[a.chunk_id].lexical_rank == 2
    assert math.isclose(by_id[a.chunk_id].rrf_score, 1 / 61 + 1 / 62)
    assert by_id[c.chunk_id].vector_rank == 3
    assert by_id[c.chunk_id].lexical_rank == 1
    assert math.isclose(by_id[c.chunk_id].rrf_score, 1 / 63 + 1 / 61)
    assert by_id[d.chunk_id].vector_rank is None
    assert by_id[d.chunk_id].lexical_rank == 3

    # a (1/61 + 1/62) beats c (1/61 + 1/63).
    assert [entry.chunk.chunk_id for entry in fused[:2]] == [a.chunk_id, c.chunk_id]


def test_rrf_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError, match="RRF k"):
        reciprocal_rank_fusion([], [], k=0)


async def test_rare_term_is_found_through_the_lexical_path() -> None:
    decoys = [
        (_chunk(100 + number, f"common phrase number {number}"), (1.0, 0.0, 0.0))
        for number in range(12)
    ]
    rare = _chunk(1, "The protocol requires a quokka token before retrying.")
    source = InMemorySource([*decoys, (rare, (0.0, 1.0, 0.0))])
    embedder = FakeEmbedder({"quokka": [1.0, 0.0, 0.0]})

    vector_only = await retrieve(
        "quokka",
        8,
        embedder=embedder,
        source=source,
        reranker=ConstantReranker(),
        mode="vector",
    )
    assert rare.chunk_id not in {chunk.chunk_id for chunk in vector_only}

    hybrid = await retrieve(
        "quokka",
        8,
        embedder=embedder,
        source=source,
        reranker=ConstantReranker(),
        vector_limit=3,
    )
    match = next((chunk for chunk in hybrid if chunk.chunk_id == rare.chunk_id), None)
    assert match is not None
    assert match.lexical_rank == 1
    assert match.vector_rank is None
    assert match.final_rank <= 8


async def test_semantic_match_is_found_without_lexical_overlap() -> None:
    semantic = _chunk(1, "Aquatic mammals propel themselves with flippers.")
    unrelated = _chunk(2, "Volcanic rock formations cool into basalt columns.")
    source = InMemorySource(
        [(semantic, (1.0, 0.0, 0.0)), (unrelated, (0.0, 1.0, 0.0))]
    )
    embedder = FakeEmbedder({"How do dolphins swim?": [1.0, 0.0, 0.0]})

    results = await retrieve(
        "How do dolphins swim?",
        8,
        embedder=embedder,
        source=source,
        reranker=ConstantReranker(),
    )

    assert [chunk.chunk_id for chunk in results] == [semantic.chunk_id, unrelated.chunk_id]
    assert results[0].vector_rank == 1
    assert results[0].lexical_rank is None
    assert results[0].final_rank == 1


async def test_rerank_can_reorder_the_fused_list() -> None:
    less_relevant = _chunk(1, "alpha passage")
    more_relevant = _chunk(2, "beta passage")
    source = InMemorySource(
        [(less_relevant, (1.0, 0.0, 0.0)), (more_relevant, (0.0, 1.0, 0.0))]
    )
    embedder = FakeEmbedder({"query": [1.0, 0.0, 0.0]})
    reranker = FixedReranker({"alpha passage": 0.1, "beta passage": 2.5})

    results = await retrieve(
        "query", 2, embedder=embedder, source=source, reranker=reranker
    )

    assert [chunk.chunk_id for chunk in results] == [more_relevant.chunk_id,
                                                    less_relevant.chunk_id]
    assert results[0].rerank_score == 2.5
    assert results[0].vector_rank == 2
    assert results[1].rerank_score == 0.1


async def test_vector_mode_returns_k_with_contiguous_final_ranks() -> None:
    source = InMemorySource(
        [(_chunk(number, f"passage {number}"), (float(number), 1.0, 0.0)) for number in range(1, 6)]
    )
    embedder = FakeEmbedder({"anything": [1.0, 0.0, 0.0]})

    results = await retrieve(
        "anything", 3, embedder=embedder, source=source,
        reranker=ConstantReranker(), mode="vector"
    )

    assert len(results) == 3
    assert [chunk.final_rank for chunk in results] == [1, 2, 3]
    assert [chunk.vector_rank for chunk in results] == [1, 2, 3]
    assert all(chunk.rerank_score is None for chunk in results)


async def test_reranker_length_mismatch_is_fatal() -> None:
    source = InMemorySource(
        [(_chunk(1, "one"), (1.0, 0.0, 0.0)), (_chunk(2, "two"), (0.5, 0.5, 0.0))]
    )
    embedder = FakeEmbedder({"query": [1.0, 0.0, 0.0]})

    with pytest.raises(RuntimeError, match="reranker returned"):
        await retrieve(
            "query", 2, embedder=embedder, source=source, reranker=ShortReranker()
        )


async def test_empty_candidates_return_no_results() -> None:
    source = InMemorySource([])
    results = await retrieve(
        "query", 8, embedder=FakeEmbedder({"query": [1.0, 0.0, 0.0]}),
        source=source, reranker=ConstantReranker()
    )
    assert results == []


async def test_lexical_reranker_is_deterministic_and_rewards_overlap() -> None:
    reranker = LexicalOverlapReranker()
    query = "masked language model pre-training"
    passages = [
        "We mask 15% of tokens and predict them with a masked language model.",
        "The optimizer is Adam with a warmup schedule.",
    ]
    first = await reranker.score(query, passages)
    second = await reranker.score(query, passages)
    assert first == second
    assert first[0] > first[1]
    assert reranker.is_dummy
    assert reranker.name.startswith("dummy:")


async def test_lexical_reranker_handles_empty_input() -> None:
    assert await LexicalOverlapReranker().score("anything", []) == []


def test_build_reranker_explicit_lexical_mode_is_dummy() -> None:
    choice = build_reranker(mode="lexical")
    assert choice.reranker.is_dummy
    assert choice.fallback_reason is None
    assert "WARNING" in choice.banner


def test_build_reranker_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="reranker mode"):
        build_reranker(mode="magic")


def test_fused_chunk_is_exposed_for_trace_consumers() -> None:
    entry = reciprocal_rank_fusion([_chunk(1, "one")], [])[0]
    assert isinstance(entry, FusedChunk)
    assert entry.vector_rank == 1

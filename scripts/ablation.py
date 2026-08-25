"""Run the retrieval ablation over ten fixed, hand-labelled corpus queries.

Each query names exactly one gold chunk by ``(doc_id, chunk_index)`` because
``chunk_index`` is deterministic across re-ingest. Metrics are Hit@8 and MRR@8;
hits are reported with a Wilson 95% confidence interval and the denominator.

Usage::

    uv run python scripts/ablation.py
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import statistics
import time
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from docqa.config import Settings, get_settings
from docqa.db import make_engine, make_session_factory
from docqa.ingest.embedder import EmbedderChoice, build_embedder
from docqa.rerank import DEFAULT_RERANK_MODEL, RerankerChoice, build_reranker
from docqa.retrieval import PgCandidateSource, RetrievalMode, retrieve

Z_95 = 1.959963984540054


@dataclass(frozen=True)
class LabelledQuery:
    """One query and the single chunk a human judged relevant."""

    text: str
    doc_id: str
    chunk_index: int


QUERIES: tuple[LabelledQuery, ...] = (
    LabelledQuery(
        "Which optimizer and learning-rate warmup schedule trained the Transformer?",
        "attention-is-all-you-need",
        16,
    ),
    LabelledQuery(
        "What sinusoidal positional encodings does the Transformer use?",
        "attention-is-all-you-need",
        12,
    ),
    LabelledQuery(
        "How many parallel attention heads does multi-head attention use?",
        "attention-is-all-you-need",
        8,
    ),
    LabelledQuery(
        "What percentage of WordPiece tokens does BERT mask during pre-training?",
        "bert",
        12,
    ),
    LabelledQuery(
        "What is the next sentence prediction objective used for?",
        "bert",
        4,
    ),
    LabelledQuery(
        "How does DPR compute similarity between a question and a passage?",
        "dense-passage-retrieval",
        8,
    ),
    LabelledQuery(
        "What are in-batch negatives used for during training?",
        "dense-passage-retrieval",
        12,
    ),
    LabelledQuery(
        "How does Sentence-BERT build sentence embeddings at inference time?",
        "sentence-bert",
        7,
    ),
    LabelledQuery(
        "Which pre-trained retriever initialises RAG's document index?",
        "retrieval-augmented-generation",
        9,
    ),
    LabelledQuery(
        "What does RAG use as its non-parametric memory?",
        "retrieval-augmented-generation",
        4,
    ),
)

MODES: tuple[RetrievalMode, ...] = ("vector", "hybrid", "hybrid_rerank")
MODE_LABELS: dict[RetrievalMode, str] = {
    "vector": "vector-only",
    "hybrid": "hybrid+RRF",
    "hybrid_rerank": "hybrid+RRF+rerank",
}


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def nearest_rank_percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile of a non-empty sample."""
    ordered = sorted(values)
    rank = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[rank]


async def _check_gold_chunks(session: AsyncSession) -> None:
    """Fail before measuring if a hand-labelled gold chunk is missing."""
    for query in QUERIES:
        result = await session.execute(
            text(
                "SELECT count(*) FROM chunks AS c JOIN documents AS d "
                "ON d.id = c.document_id "
                "WHERE d.doc_id = :doc_id AND c.chunk_index = :chunk_index"
            ),
            {"doc_id": query.doc_id, "chunk_index": query.chunk_index},
        )
        if not result.scalar_one():
            raise SystemExit(
                f"gold chunk missing: {query.doc_id} #{query.chunk_index}; re-ingest first"
            )


async def _run(
    settings: Settings,
    embedder_choice: EmbedderChoice,
    reranker_choice: RerankerChoice,
    k: int,
) -> tuple[dict[RetrievalMode, list[int | None]], list[float]]:
    engine = make_engine(settings)
    factory = make_session_factory(engine)
    ranks: dict[RetrievalMode, list[int | None]] = {mode: [] for mode in MODES}
    try:
        async with factory() as session:
            await _check_gold_chunks(session)
            source = PgCandidateSource(session)
            for query in QUERIES:
                for mode in MODES:
                    results = await retrieve(
                        query.text,
                        k,
                        embedder=embedder_choice.embedder,
                        source=source,
                        reranker=reranker_choice.reranker,
                        mode=mode,
                    )
                    found = next(
                        (
                            chunk.final_rank
                            for chunk in results
                            if (chunk.doc_id, chunk.chunk_index)
                            == (query.doc_id, query.chunk_index)
                        ),
                        None,
                    )
                    ranks[mode].append(found)

            # Warm the caches once, then time twenty production-path calls.
            await retrieve(
                QUERIES[0].text,
                k,
                embedder=embedder_choice.embedder,
                source=source,
                reranker=reranker_choice.reranker,
            )
            timings: list[float] = []
            for _ in range(20):
                started = time.perf_counter()
                await retrieve(
                    QUERIES[0].text,
                    k,
                    embedder=embedder_choice.embedder,
                    source=source,
                    reranker=reranker_choice.reranker,
                )
                timings.append((time.perf_counter() - started) * 1000.0)
    finally:
        await engine.dispose()
    return ranks, timings


def _print_table(ranks: dict[RetrievalMode, list[int | None]]) -> None:
    header = "| # | query | gold (doc #chunk) | vector-only | hybrid+RRF | +rerank |"
    print("\n" + header)
    print("|---|---|---|---:|---:|---:|")
    for index, query in enumerate(QUERIES, start=1):
        cells = []
        for mode in MODES:
            rank = ranks[mode][index - 1]
            cells.append(str(rank) if rank is not None else "miss")
        print(
            f"| {index} | {query.text} | {query.doc_id} #{query.chunk_index} | "
            + " | ".join(cells)
            + " |"
        )

    print("\n| stage | Hit@8 | 95% Wilson CI | MRR@8 | n |")
    print("|---|---:|---|---:|---:|")
    n = len(QUERIES)
    for mode in MODES:
        mode_ranks = ranks[mode]
        hits = sum(1 for rank in mode_ranks if rank is not None)
        reciprocal = sum(1.0 / rank for rank in mode_ranks if rank is not None)
        low, high = wilson_interval(hits, n)
        print(
            f"| {MODE_LABELS[mode]} | {hits}/{n} ({hits / n:.0%}) | "
            f"[{low:.0%}, {high:.0%}] | {reciprocal / n:.3f} | {n} |"
        )
    print("\nA miss means the single gold chunk was outside the top 8.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=8, help="final chunks kept (default 8)")
    args = parser.parse_args()

    settings = get_settings()
    embedder_choice = build_embedder(settings)
    print(embedder_choice.banner)
    if embedder_choice.embedder.is_dummy:
        print(
            "ERROR: the ablation needs the real bge-small-en-v1.5 vectors; "
            "set DOCQA_EMBEDDER=fastembed."
        )
        return 1

    reranker_choice = build_reranker(
        mode=os.environ.get("DOCQA_RERANKER", "auto"),
        model_name=os.environ.get("DOCQA_RERANK_MODEL", DEFAULT_RERANK_MODEL),
        cache_dir=str(settings.hf_cache_dir) if settings.hf_cache_dir else None,
    )
    print(reranker_choice.banner)
    if reranker_choice.fallback_reason:
        print(f"  fallback reason: {reranker_choice.fallback_reason}")
    if reranker_choice.reranker.is_dummy:
        print("NOTE: the +rerank row is a dummy scorer; it is not cross-encoder evidence.")

    ranks, timings = asyncio.run(_run(settings, embedder_choice, reranker_choice, args.k))
    _print_table(ranks)
    print(f"\nRetrieval latency over 20 warm calls ({MODE_LABELS['hybrid_rerank']}):")
    print(
        f"  p50 {statistics.median(timings):.1f} ms · "
        f"p95 {nearest_rank_percentile(timings, 0.95):.1f} ms · "
        f"max {max(timings):.1f} ms"
    )
    print("  command: uv run python scripts/ablation.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

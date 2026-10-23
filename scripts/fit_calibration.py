"""Fit the refusal calibration from (query, chunk) pairs and write eval/calibration.json.

The artifact is a Platt scaling ``p = sigmoid(a * s + b)`` over the reranker
score ``s`` plus the balanced-accuracy-optimal threshold. Nothing in the output
is typed by hand: ``a`` and ``b`` come from Newton's method on the labelled
samples, and the threshold maximises ``TPR + TNR`` over the observed scores.

Provenance matters. Until R5 supplies human-labelled gold questions, the pairs
are machine-generated: each query is a sentence from a real chunk in the local
corpus (positive) against other chunks from the same corpus (negatives, of which
the nine highest-scoring are kept as hard negatives). This fit is **optimistic**
for real question phrasing: the live smoke test reports top-1 reranker scores of
2.4–6.8 for answerable questions, below the sentence-derived threshold. The
artifact records its provenance and fit-set hash, and R5 must refit on the
100-question gold set rather than treating this threshold as an eval result.

Run with the real cross-encoder:

    uv run python scripts/fit_calibration.py
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import math
import re
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from docqa.config import PROJECT_ROOT, get_settings
from docqa.rerank import build_reranker

SEED_QUERIES = 30
NEGATIVES_PER_QUERY = 9
CANDIDATES_PER_QUERY = 20
SEED = 20261023

UNANSWERABLE_QUERIES = [
    "What is the boiling point of mercury?",
    "Who painted the ceiling of the Sistine Chapel?",
    "How many moons does Jupiter have?",
    "What is the capital of Australia?",
    "When did the Berlin Wall fall?",
    "What is the recommended daily intake of vitamin C?",
    "How do you change a flat bicycle tyre?",
    "What was the population of Rome in 100 BC?",
    "Who won the 2010 FIFA World Cup?",
    "What causes earthquakes along the San Andreas fault?",
    "How long does it take to bake sourdough bread?",
    "What is the chemical formula for table salt?",
]

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9-]{3,}")
_STOPWORDS = frozenset(
    {
        "the", "and", "for", "that", "with", "this", "from", "are", "was", "were",
        "which", "their", "have", "has", "had", "not", "but", "its", "our", "can",
        "may", "also", "such", "these", "those", "than", "then", "they", "them",
        "into", "over", "under", "more", "most", "some", "any", "all", "use", "used",
        "using", "we", "you", "your", "does", "what", "how", "why", "when", "where",
    }
)


@dataclass(frozen=True)
class ChunkRow:
    """The pieces of a stored chunk the fitter needs."""

    chunk_id: int
    text: str
    doc_id: str


@dataclass(frozen=True)
class Pair:
    """One labelled (query, chunk) pair and its reranker score."""

    query: str
    chunk_id: int
    label: int
    score: float


def _seed_indices(size: int, wanted: int) -> list[int]:
    """Evenly spread ``wanted`` indices over ``range(size)``."""
    if size <= wanted:
        return list(range(size))
    step = size / wanted
    return sorted({min(size - 1, int(index * step)) for index in range(wanted)})


def _query_for(chunk: ChunkRow) -> str | None:
    """Pick a long-ish sentence with content terms; ``None`` if the chunk is unusable."""
    for sentence in _SENTENCE_RE.split(chunk.text.strip()):
        candidate = " ".join(sentence.split())
        if not 40 <= len(candidate) <= 240:
            continue
        terms = [
            term for term in _TERM_RE.findall(candidate) if term.lower() not in _STOPWORDS
        ]
        if len(set(terms)) >= 5:
            return candidate
    return None


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def fit_platt(pairs: list[Pair]) -> tuple[float, float]:
    """Fit ``p = sigmoid(a*s + b)`` by ridge-regularised Newton-Raphson.

    Scores are standardised before fitting: the synthetic fit set is close to
    separable, and unregularised Newton would drive ``a`` to infinity and make
    every probability exactly 0 or 1. A ridge term on the standardised slope
    keeps the sigmoid usable while still letting the data set the slope.
    """
    scores = [pair.score for pair in pairs]
    mean = sum(scores) / len(scores)
    variance = sum((score - mean) ** 2 for score in scores) / len(scores)
    scale = math.sqrt(variance) or 1.0
    a, b = 1.0, 0.0
    ridge = 1.0
    for _ in range(200):
        grad_a = grad_b = 0.0
        h_aa = h_ab = h_bb = 0.0
        for pair in pairs:
            z = (pair.score - mean) / scale
            probability = _sigmoid(a * z + b)
            residual = probability - pair.label
            weight = probability * (1.0 - probability)
            grad_a += residual * z
            grad_b += residual
            h_aa += weight * z * z
            h_ab += weight * z
            h_bb += weight
        grad_a += ridge * a
        h_aa += ridge
        h_bb += ridge
        determinant = h_aa * h_bb - h_ab * h_ab
        if determinant == 0.0:
            break
        delta_a = (h_bb * grad_a - h_ab * grad_b) / determinant
        delta_b = (h_aa * grad_b - h_ab * grad_a) / determinant
        a -= delta_a
        b -= delta_b
        if abs(delta_a) < 1e-10 and abs(delta_b) < 1e-10:
            break
    return a / scale, b - a * mean / scale


def balanced_accuracy(pairs: list[Pair], a: float, b: float, threshold: float) -> float:
    """TPR + TNR, the quantity the threshold maximises."""
    positives = [pair for pair in pairs if pair.label == 1]
    negatives = [pair for pair in pairs if pair.label == 0]
    if not positives or not negatives:
        raise ValueError("calibration needs both positive and negative pairs")
    true_positive = sum(1 for p in positives if _sigmoid(a * p.score + b) >= threshold)
    true_negative = sum(1 for p in negatives if _sigmoid(a * p.score + b) < threshold)
    return true_positive / len(positives) + true_negative / len(negatives)


def choose_threshold(pairs: list[Pair], a: float, b: float) -> tuple[float, float]:
    """Return the balanced-accuracy-optimal threshold and its score."""
    probabilities = sorted({_sigmoid(a * pair.score + b) for pair in pairs})
    candidates = [0.0, *probabilities, 1.0]
    boundaries = [
        (low + high) / 2 for low, high in itertools.pairwise(candidates)
    ]
    scored = [(balanced_accuracy(pairs, a, b, value), value) for value in boundaries]
    # Highest accuracy; on a tie prefer the tighter (higher) threshold.
    accuracy, threshold = max(scored, key=lambda item: (item[0], item[1]))
    return threshold, accuracy / 2.0  # mean of TPR and TNR


def canonical_fit_set(pairs: list[Pair]) -> bytes:
    """Hashable form of the fit set: labels and identities, not model scores."""
    rows = [
        {"query": pair.query, "chunk_id": pair.chunk_id, "label": pair.label}
        for pair in sorted(pairs, key=lambda pair: (pair.query, pair.chunk_id))
    ]
    return json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")


async def _load_chunks() -> list[ChunkRow]:
    engine = create_async_engine(get_settings().database_url)
    try:
        async with engine.connect() as connection:
            result = await connection.execute(
                text(
                    "SELECT c.id, c.text, d.doc_id FROM chunks c "
                    "JOIN documents d ON d.id = c.document_id ORDER BY c.id"
                )
            )
            return [
                ChunkRow(chunk_id=int(row[0]), text=str(row[1]), doc_id=str(row[2]))
                for row in result
            ]
    finally:
        await engine.dispose()


async def build_pairs() -> tuple[list[Pair], str]:
    """Score machine-generated answerable queries against hard negatives."""
    chunks = await _load_chunks()
    if len(chunks) < CANDIDATES_PER_QUERY + 1:
        raise RuntimeError("corpus is too small to fit a calibration")
    choice = build_reranker(mode="fastembed")
    reranker = choice.reranker
    seeds = _seed_indices(len(chunks), SEED_QUERIES)
    stride = max(1, len(chunks) // CANDIDATES_PER_QUERY)
    pairs: list[Pair] = []
    used = 0
    for seed_position in seeds:
        seed = chunks[seed_position]
        query = _query_for(seed)
        if query is None:
            continue
        pool: list[ChunkRow] = []
        seen = {seed.chunk_id}
        for offset in range(CANDIDATES_PER_QUERY):
            candidate = chunks[(seed_position * 7 + offset * stride) % len(chunks)]
            if candidate.chunk_id not in seen:
                seen.add(candidate.chunk_id)
                pool.append(candidate)
        scores = await reranker.score(query, [seed.text, *(chunk.text for chunk in pool)])
        positive_score = scores[0]
        decorated = sorted(
            zip(pool, scores[1:], strict=True),
            key=lambda item: (-item[1], item[0].chunk_id),
        )
        pairs.append(Pair(query=query, chunk_id=seed.chunk_id, label=1, score=positive_score))
        for chunk, score in decorated[:NEGATIVES_PER_QUERY]:
            pairs.append(Pair(query=query, chunk_id=chunk.chunk_id, label=0, score=score))
        used += 1
    if used < 20:
        raise RuntimeError(f"only {used} usable seed queries; need at least 20")
    return pairs, reranker.name


def main() -> int:
    """Fit, write both artifacts, and print the fit summary."""
    pairs, reranker_name = asyncio.run(build_pairs())
    positives = sum(1 for pair in pairs if pair.label == 1)
    negatives = len(pairs) - positives
    a, b = fit_platt(pairs)
    threshold, mean_balanced_accuracy = choose_threshold(pairs, a, b)
    fit_hash = hashlib.sha256(canonical_fit_set(pairs)).hexdigest()

    fit_set_path = PROJECT_ROOT / "eval" / "calibration_fit_set.json"
    fit_set_path.parent.mkdir(parents=True, exist_ok=True)
    fit_set = [
        {
            "query": pair.query,
            "chunk_id": pair.chunk_id,
            "label": pair.label,
            "rerank_score": pair.score,
        }
        for pair in sorted(pairs, key=lambda pair: (pair.query, pair.chunk_id))
    ]
    fit_set_path.write_text(json.dumps(fit_set, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    calibration = {
        "a": a,
        "b": b,
        "threshold": threshold,
        "fit_set_sha256": fit_hash,
        "fit_size": len(pairs),
        "positives": positives,
        "negatives": negatives,
        "mean_balanced_accuracy": mean_balanced_accuracy,
        "reranker": reranker_name,
        "provenance": (
            "provisional-self-retrieval-v1: machine-generated queries from local corpus "
            "sentences, hard negatives by reranker score; R5 refits on human gold"
        ),
        "fitted_at": "2026-10-23",
    }
    target = PROJECT_ROOT / "eval" / "calibration.json"
    target.write_text(json.dumps(calibration, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        f"fitted {len(pairs)} pairs ({positives} positive, {negatives} negative) "
        f"with {reranker_name}"
    )
    print(f"a={a:.6f} b={b:.6f} threshold={threshold:.6f} "
          f"balanced_accuracy={mean_balanced_accuracy:.3f}")
    print(f"fit_set_sha256={fit_hash}")
    print(f"wrote {target.relative_to(PROJECT_ROOT)} and {fit_set_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

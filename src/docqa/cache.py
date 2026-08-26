"""Precomputed demo answers and the retrieval-trace serialiser.

The free keyless endpoint answers once per 30 s, so the API cannot call it on
every demo click. ``docqa warm`` writes ``data/demo-answers.json`` with the
exact question, the model name, and the timestamp; the API serves those entries
instantly and only calls the live endpoint on a cache miss.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from docqa.answer import AnswerResponse, Calibration
from docqa.retrieval import RerankTraceHit, RetrievalTrace

LOGGER = logging.getLogger(__name__)

CACHE_VERSION = 1
DEMO_QUESTION_COUNT = 6

AnswerFn = Callable[[str], Awaitable[tuple[AnswerResponse, RetrievalTrace | None]]]
ProgressFn = Callable[[int, int, "CachedAnswer"], None]


@dataclass(frozen=True)
class DemoQuestion:
    """One fixed question the demo page offers."""

    id: str
    question: str
    kind: str


@dataclass(frozen=True)
class CachedAnswer:
    """A validated answer recorded with the model that produced it."""

    question: str
    model: str
    generated_at: str
    structured_output: str
    response: dict[str, object]
    trace: dict[str, object] | None

    def as_dict(self) -> dict[str, object]:
        """Return the on-disk shape naming the model and exact question."""
        return {
            "question": self.question,
            "model": self.model,
            "generated_at": self.generated_at,
            "structured_output": self.structured_output,
            "response": self.response,
            "trace": self.trace,
        }


@dataclass(frozen=True)
class AnswerCache:
    """Exact-question lookup over ``data/demo-answers.json``."""

    path: Path
    generated_at: str | None
    model: str | None
    entries: dict[str, CachedAnswer]

    def get(self, question: str) -> CachedAnswer | None:
        """Return the recorded answer for the exact question text, if any."""
        return self.entries.get(question.strip())

    def __len__(self) -> int:
        """Return the number of usable entries."""
        return len(self.entries)

    @classmethod
    def empty(cls, path: Path) -> AnswerCache:
        """Return a cache that will never hit."""
        return cls(path=path, generated_at=None, model=None, entries={})

    @classmethod
    def load(cls, path: Path) -> AnswerCache:
        """Load the cache; a missing or malformed file is an empty cache.

        A corrupt demo cache must not stop the API booting: the live endpoint is
        still reachable, and the miss path reports its own state.
        """
        if not path.is_file():
            return cls.empty(path)
        try:
            raw: object = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            LOGGER.warning("ignoring unreadable demo cache %s: %s", path, exc)
            return cls.empty(path)
        if not isinstance(raw, dict) or raw.get("version") != CACHE_VERSION:
            LOGGER.warning("ignoring demo cache %s: unexpected version", path)
            return cls.empty(path)
        answers = raw.get("answers")
        if not isinstance(answers, list):
            LOGGER.warning("ignoring demo cache %s: answers is not a list", path)
            return cls.empty(path)
        entries: dict[str, CachedAnswer] = {}
        for item in answers:
            entry = _parse_entry(item)
            if entry is not None:
                entries[entry.question.strip()] = entry
        generated_at = raw.get("generated_at")
        model = raw.get("model")
        return cls(
            path=path,
            generated_at=generated_at if isinstance(generated_at, str) else None,
            model=model if isinstance(model, str) else None,
            entries=entries,
        )


def _parse_entry(item: object) -> CachedAnswer | None:
    if not isinstance(item, dict):
        return None
    question = item.get("question")
    model = item.get("model")
    generated_at = item.get("generated_at")
    structured_output = item.get("structured_output")
    response = item.get("response")
    trace = item.get("trace")
    if (
        not isinstance(question, str)
        or not question.strip()
        or not isinstance(model, str)
        or not isinstance(generated_at, str)
        or not isinstance(structured_output, str)
        or not isinstance(response, dict)
        or (trace is not None and not isinstance(trace, dict))
    ):
        return None
    return CachedAnswer(
        question=question,
        model=model,
        generated_at=generated_at,
        structured_output=structured_output,
        response=cast(dict[str, object], response),
        trace=cast(dict[str, object] | None, trace),
    )


def trace_payload(
    trace: RetrievalTrace, response: AnswerResponse, calibration: Calibration
) -> dict[str, object]:
    """Serialise a retrieval trace, attaching the answer-level refusal numbers."""
    top = trace.rerank_hits[0] if trace.rerank_hits else None
    return {
        "mode": trace.mode,
        "embedder": trace.embedder,
        "reranker": trace.reranker,
        "vector_hits": [dataclasses.asdict(hit) for hit in trace.vector_hits],
        "lexical_hits": [dataclasses.asdict(hit) for hit in trace.lexical_hits],
        "fusion_hits": [dataclasses.asdict(hit) for hit in trace.fusion_hits],
        "rerank_hits": [
            {
                **dataclasses.asdict(hit),
                "calibrated_probability": _calibrated(calibration, hit),
            }
            for hit in trace.rerank_hits
        ],
        "context_chunk_ids": list(trace.context_chunk_ids),
        "context_tokens": trace.context_tokens,
        "refusal": {
            "top_rerank_score": None if top is None else top.rerank_score,
            "top_probability": response.refusal_probability,
            "threshold": calibration.threshold,
            "decision": "refused" if response.insufficient_evidence else "answered",
            "reason": response.reason,
            "uncalibrated": calibration.uncalibrated,
        },
    }


def _calibrated(calibration: Calibration, hit: RerankTraceHit) -> float | None:
    """Apply the Platt scaling to one reranked candidate."""
    return calibration.probability(hit.rerank_score)


def load_demo_questions(path: Path) -> tuple[DemoQuestion, ...]:
    """Load the fixed demo questions, refusing an incomplete replacement."""
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read demo questions at {path}: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("questions"), list):
        raise ValueError(f"{path} must contain a JSON object with a questions list")
    questions: list[DemoQuestion] = []
    for index, item in enumerate(raw["questions"]):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: questions[{index}] must be an object")
        question_id = item.get("id")
        question = item.get("question")
        kind = item.get("kind")
        if (
            not isinstance(question_id, str)
            or not isinstance(question, str)
            or not question.strip()
            or not isinstance(kind, str)
        ):
            raise ValueError(
                f"{path}: questions[{index}] needs id, question and kind"
            )
        questions.append(
            DemoQuestion(id=question_id, question=question.strip(), kind=kind)
        )
    if len(questions) != DEMO_QUESTION_COUNT:
        raise ValueError(
            f"{path} must hold exactly {DEMO_QUESTION_COUNT} demo questions, "
            f"found {len(questions)}"
        )
    return tuple(questions)


def _write_cache(
    path: Path, model_name: str, entries: dict[str, CachedAnswer]
) -> None:
    """Write atomically so a rate-limited run never leaves a truncated file."""
    payload = {
        "version": CACHE_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "model": model_name,
        "questions": len(entries),
        "answers": [entry.as_dict() for entry in entries.values()],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


async def warm_demo_answers(
    questions: Sequence[DemoQuestion],
    *,
    answer: AnswerFn,
    out_path: Path,
    model_name: str,
    calibration: Calibration,
    reuse: bool = True,
    progress: ProgressFn | None = None,
) -> AnswerCache:
    """Generate or reuse a recorded answer for every demo question.

    Each completed answer is written before the next request, so a rate limit
    on question 5 leaves the first four usable and the last one missing rather
    than corrupting the file.
    """
    existing = AnswerCache.load(out_path) if reuse else AnswerCache.empty(out_path)
    entries: dict[str, CachedAnswer] = {}
    for position, question in enumerate(questions, start=1):
        prior = existing.get(question.question)
        if prior is not None and prior.model == model_name:
            entries[question.question] = prior
            if progress is not None:
                progress(position, len(questions), prior)
            continue

        response, trace = await answer(question.question)
        entry = CachedAnswer(
            question=question.question,
            model=model_name,
            generated_at=datetime.now(UTC).isoformat(),
            structured_output=response.structured_output,
            response=dataclasses.asdict(response),
            trace=(
                None
                if trace is None
                else trace_payload(trace, response, calibration)
            ),
        )
        entries[question.question] = entry
        _write_cache(out_path, model_name, entries)
        if progress is not None:
            progress(position, len(questions), entry)

    _write_cache(out_path, model_name, entries)
    return AnswerCache(
        path=out_path,
        generated_at=datetime.now(UTC).isoformat(),
        model=model_name,
        entries=entries,
    )

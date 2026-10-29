"""Honest evaluation harness: span-level gold, Wilson intervals, gates, ablation.

The harness does three things and refuses to pretend about the rest:

* it loads ``eval/gold.jsonl`` and verifies every gold quote is a normalised
  substring of the named chunk **in the live database** before measuring;
* it reports every rate with a Wilson 95% interval and its denominator, records
  ``stage_failed`` per question, and exits non-zero when a gate misses;
* its citation-precision and faithfulness metrics are computed by a
  deterministic rule-based checker and are labelled
  ``rule-based, not a model judge`` — no judge model is reachable here.

``--ablate vector-only`` runs the same questions through a deliberately
weakened retriever (vector-only, top-3). That configuration must miss at least
one gate; the report records it separately rather than overwriting the
production report.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast
from urllib.parse import urlparse

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from docqa.answer import (
    AnswerResponse,
    Calibration,
    ModelCallError,
    RateLimitedError,
    StructuredModel,
    answer_question,
    build_model,
    load_calibration,
    normalise_text,
)
from docqa.config import PROJECT_ROOT, Settings, get_settings
from docqa.db import make_engine, make_session_factory
from docqa.ingest.embedder import Embedder, build_embedder
from docqa.judge import (
    JudgeStatus,
    kappa_decision,
    probe_judge,
    read_flat_yaml,
    write_judge_status,
)
from docqa.rerank import Reranker, build_reranker
from docqa.retrieval import (
    PgCandidateSource,
    RetrievalMode,
    RetrievalResult,
    RetrievedChunk,
    retrieve_with_trace,
)

Z_95 = 1.959963984540054
DEFAULT_GOLD_PATH = PROJECT_ROOT / "eval" / "gold.jsonl"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "eval-report.json"
DEFAULT_ABLATION_REPORT_PATH = PROJECT_ROOT / "eval-report-ablation.json"
DEFAULT_FORCED_REPORT_PATH = PROJECT_ROOT / "eval-report-forced.json"
EXIT_GATE = 6

ANSWERABLE_COUNT = 24
UNANSWERABLE_COUNTS: dict[str, int] = {
    "near_miss": 2,
    "out_of_corpus": 2,
    "under_specified": 1,
    "ambiguous": 1,
}

GATE_RETRIEVAL_RECALL = 0.75
GATE_ANSWER_RATE = 0.50
GATE_UNANSWERABLE_REFUSAL = 0.80
GATE_CITATION_PRECISION = 0.80
GATE_FAITHFULNESS = 0.80

RETRIEVAL_ABLATION: tuple[tuple[str, RetrievalMode], ...] = (
    ("vector-only top-8", "vector"),
    ("+lexical/RRF top-8", "hybrid"),
    ("+rerank top-8", "hybrid_rerank"),
)

STAGES: tuple[str, ...] = (
    "retrieval",
    "fusion",
    "rerank",
    "generation",
    "schema",
    "validation",
)
RULE_BASED_LABEL = "rule-based, not a model judge"


class EvalError(RuntimeError):
    """The evaluation cannot run as configured."""


class GoldError(EvalError):
    """The gold file is malformed or disagrees with the corpus."""


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (0, 0 for an empty sample)."""
    if n == 0:
        return 0.0, 0.0
    if not 0 <= successes <= n:
        raise ValueError("successes must be between 0 and n")
    p = successes / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


@dataclass(frozen=True)
class Rate:
    """One binomial rate. The denominator is always carried with it."""

    successes: int
    n: int

    @property
    def value(self) -> float | None:
        """The observed rate, or ``None`` when nothing was measurable."""
        return self.successes / self.n if self.n else None

    @property
    def wilson_95(self) -> tuple[float, float] | None:
        """Wilson 95% interval, or ``None`` for an empty denominator."""
        return wilson_interval(self.successes, self.n) if self.n else None

    def as_dict(self) -> dict[str, object]:
        """JSON-safe form, including n and the interval."""
        interval = self.wilson_95
        return {
            "successes": self.successes,
            "n": self.n,
            "rate": self.value,
            "wilson_95": None if interval is None else [interval[0], interval[1]],
        }


def format_rate(rate: Rate) -> str:
    """Render a rate with its denominator and Wilson interval."""
    if rate.n == 0 or rate.value is None or rate.wilson_95 is None:
        return "n/a (n=0)"
    low, high = rate.wilson_95
    return (
        f"{rate.successes}/{rate.n} ({rate.value:.1%}) "
        f"[{low:.1%}, {high:.1%}]"
    )


# --------------------------------------------------------------------------- #
# Gold
# --------------------------------------------------------------------------- #

GoldKind = Literal[
    "answerable", "near_miss", "out_of_corpus", "under_specified", "ambiguous"
]


@dataclass(frozen=True)
class GoldSpan:
    """The hand-labelled supporting span for one answerable question."""

    doc_id: str
    page_no: int
    chunk_index: int
    quote: str


@dataclass(frozen=True)
class EvalQuestion:
    """One fixed evaluation question and its label."""

    id: str
    question: str
    kind: GoldKind
    gold: GoldSpan | None
    note: str | None = None

    @property
    def answerable(self) -> bool:
        """True only for the ``answerable`` kind."""
        return self.kind == "answerable"


@dataclass(frozen=True)
class ResolvedGold:
    """A gold span resolved to its live chunk id and text."""

    span: GoldSpan
    chunk_id: int
    chunk_text: str


def _as_dict(value: object) -> dict[str, object] | None:
    return cast(dict[str, object], value) if isinstance(value, dict) else None


def _as_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _as_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _parse_question(raw: object, line_no: int) -> EvalQuestion:
    record = _as_dict(raw)
    if record is None:
        raise GoldError(f"line {line_no}: expected a JSON object")
    question_id = _as_str(record.get("id"))
    question = _as_str(record.get("question"))
    kind = _as_str(record.get("kind"))
    if not question_id or not question:
        raise GoldError(f"line {line_no}: id and question must be non-empty strings")
    if kind not in {"answerable", *UNANSWERABLE_COUNTS}:
        raise GoldError(f"line {line_no}: unknown kind {kind!r}")

    gold_raw = record.get("gold")
    gold: GoldSpan | None = None
    if gold_raw is not None:
        gold_record = _as_dict(gold_raw)
        if gold_record is None:
            raise GoldError(f"line {line_no}: gold must be an object or null")
        doc_id = _as_str(gold_record.get("doc_id"))
        page_no = _as_int(gold_record.get("page_no"))
        chunk_index = _as_int(gold_record.get("chunk_index"))
        quote = _as_str(gold_record.get("quote"))
        if not doc_id or page_no is None or chunk_index is None or not quote:
            raise GoldError(
                f"line {line_no}: gold needs doc_id, page_no, chunk_index and quote"
            )
        gold = GoldSpan(
            doc_id=doc_id, page_no=page_no, chunk_index=chunk_index, quote=quote
        )

    answerable = kind == "answerable"
    if answerable and gold is None:
        raise GoldError(f"line {line_no}: answerable question {question_id} has no gold")
    if not answerable and gold is not None:
        raise GoldError(
            f"line {line_no}: unanswerable question {question_id} must not carry gold"
        )
    note = _as_str(record.get("note"))
    return EvalQuestion(
        id=question_id,
        question=question,
        kind=cast(GoldKind, kind),
        gold=gold,
        note=note,
    )


def load_gold(path: Path = DEFAULT_GOLD_PATH) -> tuple[EvalQuestion, ...]:
    """Load and shape-check the fixed gold set.

    The count check is deliberate: a smaller set is allowed only after the
    reduction is documented, so this harness refuses to run silently on one.
    """
    if not path.is_file():
        raise GoldError(f"gold file not found: {path}")
    questions: list[EvalQuestion] = []
    seen: set[str] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw: object = json.loads(line)
        except ValueError as exc:
            raise GoldError(f"line {line_no}: invalid JSON: {exc}") from exc
        question = _parse_question(raw, line_no)
        if question.id in seen:
            raise GoldError(f"line {line_no}: duplicate id {question.id}")
        seen.add(question.id)
        questions.append(question)

    answerable = sum(1 for question in questions if question.answerable)
    if answerable != ANSWERABLE_COUNT:
        raise GoldError(
            f"expected {ANSWERABLE_COUNT} answerable questions, found {answerable}; "
            "document any reduction in eval/README.md before changing this bound"
        )
    kind_counts = Counter(question.kind for question in questions if not question.answerable)
    if dict(kind_counts) != UNANSWERABLE_COUNTS:
        raise GoldError(
            f"expected unanswerable kinds {UNANSWERABLE_COUNTS}, found {dict(kind_counts)}"
        )
    return tuple(questions)


async def resolve_gold(
    questions: Sequence[EvalQuestion], session: AsyncSession
) -> dict[str, ResolvedGold]:
    """Resolve every gold span against the live corpus, failing on any mismatch."""
    resolved: dict[str, ResolvedGold] = {}
    errors: list[str] = []
    for question in questions:
        span = question.gold
        if span is None:
            continue
        result = await session.execute(
            text(
                "SELECT c.id, c.page_no, c.text FROM chunks AS c "
                "JOIN documents AS d ON d.id = c.document_id "
                "WHERE d.doc_id = :doc_id AND c.chunk_index = :chunk_index"
            ),
            {"doc_id": span.doc_id, "chunk_index": span.chunk_index},
        )
        row = result.one_or_none()
        if row is None:
            errors.append(
                f"{question.id}: chunk {span.doc_id} #{span.chunk_index} is not ingested"
            )
            continue
        chunk_id, page_no, chunk_text = int(row[0]), int(row[1]), str(row[2])
        if page_no != span.page_no:
            errors.append(
                f"{question.id}: gold page {span.page_no} != chunk page {page_no}"
            )
        if normalise_text(span.quote) not in normalise_text(chunk_text):
            errors.append(
                f"{question.id}: gold quote is not a substring of "
                f"{span.doc_id} #{span.chunk_index}"
            )
        resolved[question.id] = ResolvedGold(
            span=span, chunk_id=chunk_id, chunk_text=chunk_text
        )
    if errors:
        raise GoldError("gold does not match the ingested corpus:\n" + "\n".join(errors))
    return resolved


def gold_sha256(path: Path = DEFAULT_GOLD_PATH) -> str:
    """Hash the gold file so a report names the exact labels it measured."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Question outcomes and stage attribution
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class QuestionOutcome:
    """Everything the report says about one question."""

    id: str
    question: str
    kind: str
    answerable: bool
    gold: dict[str, object] | None
    retrieved: bool | None
    gold_rank: int | None
    refused: bool
    answered: bool
    refusal_reason: str | None
    claims: int
    citations: int
    citations_correct: int
    claims_faithful: int
    stage_failed: str | None
    schema_failures: int
    top_rerank_score: float | None
    refusal_probability: float | None
    refusal_threshold: float | None
    latency_ms: float
    error: str | None
    model_requested: str | None = None
    model_answered: str | None = None
    # The mode that produced the final payload, and the concrete measured
    # reason when ``stage_failed`` is not null.
    structured_mode: str | None = None
    failure: dict[str, object] | None = None

    def as_dict(self) -> dict[str, object]:
        """JSON-safe row."""
        return dataclasses.asdict(self)


def attribute_stage(
    *,
    mode: RetrievalMode,
    answerable: bool,
    in_context: bool,
    in_vector: bool,
    in_lexical: bool,
    in_fusion: bool,
    refused: bool,
    refused_for_schema: bool,
    refused_for_validation: bool,
    has_claims: bool,
) -> str | None:
    """Name the earliest stage that lost the gold span or the answer.

    ``None`` means the question met its expected outcome (gold in context and
    answered for answerable questions, refused for unanswerable ones).
    """
    if answerable and not in_context:
        if mode == "vector":
            return "retrieval"
        if not in_vector and not in_lexical:
            return "retrieval"
        if mode == "hybrid":
            return "fusion"
        if not in_fusion:
            return "fusion"
        return "rerank"
    if not answerable:
        return None if refused else "generation"
    if refused:
        # A claim dropped by the citation contract is not a schema failure;
        # keep the two causes separate so the counts mean what they say.
        if refused_for_validation:
            return "validation"
        return "schema" if refused_for_schema else "generation"
    return None if has_claims else "generation"


def _gold_matches(chunk: RetrievedChunk, gold: ResolvedGold) -> bool:
    return chunk.chunk_id == gold.chunk_id


def _quotes_overlap(claim_quote: str, gold_quote: str) -> bool:
    claim = normalise_text(claim_quote)
    gold = normalise_text(gold_quote)
    return bool(claim) and bool(gold) and (claim in gold or gold in claim)


def _build_outcome(
    question: EvalQuestion,
    gold: ResolvedGold | None,
    retrieval: RetrievalResult,
    response: AnswerResponse | None,
    *,
    mode: RetrievalMode,
    latency_ms: float,
    requested_model: str | None = None,
    error: str | None = None,
    model_failure: dict[str, object] | None = None,
) -> QuestionOutcome:
    context_ids = {chunk.chunk_id for chunk in retrieval.chunks}
    gold_id = gold.chunk_id if gold is not None else None
    gold_rank: int | None = None
    if gold_id is not None:
        for chunk in retrieval.chunks:
            if chunk.chunk_id == gold_id:
                gold_rank = chunk.final_rank
                break
    in_context = gold_id is not None and gold_id in context_ids

    if response is None:
        refused = False
        answered = False
        reason: str | None = "model_call_error"
        claims = 0
        citations = 0
        citations_correct = 0
        claims_faithful = 0
        schema_failures = 0
        probability = None
        model_requested = requested_model
        model_answered = None
        structured_mode: str | None = None
    else:
        refused = response.insufficient_evidence
        answered = not refused and bool(response.claims)
        reason = response.reason
        claims = len(response.claims)
        citations = sum(len(claim.citations) for claim in response.claims)
        citations_correct = sum(
            1
            for claim in response.claims
            for citation in claim.citations
            if gold_id is not None and citation.chunk_id == gold_id
        )
        claims_faithful = sum(
            1
            for claim in response.claims
            if gold is not None
            and claim.citations
            and all(
                citation.chunk_id == gold_id
                and _quotes_overlap(citation.quote, gold.span.quote)
                for citation in claim.citations
            )
        )
        schema_failures = response.validation.schema_failures
        probability = response.refusal_probability
        model_requested = response.model_requested or requested_model
        model_answered = response.model_answered
        structured_mode = response.structured_mode

    refused_for_validation = bool(
        response is not None
        and response.reason == "validation_failed"
        and response.validation.claims_dropped > 0
    )
    refused_for_schema = bool(
        response is not None
        and response.reason == "validation_failed"
        and not refused_for_validation
        and schema_failures > 0
    )
    trace = retrieval.trace
    stage = attribute_stage(
        mode=mode,
        answerable=question.answerable,
        in_context=in_context,
        in_vector=gold_id in {hit.chunk_id for hit in trace.vector_hits},
        in_lexical=gold_id in {hit.chunk_id for hit in trace.lexical_hits},
        in_fusion=gold_id in {hit.chunk_id for hit in trace.fusion_hits},
        refused=refused,
        refused_for_schema=refused_for_schema,
        refused_for_validation=refused_for_validation,
        has_claims=answered,
    )
    top_score = retrieval.chunks[0].rerank_score if retrieval.chunks else None
    failure = _stage_failure(
        stage=stage,
        response=response,
        model_failure=model_failure,
        gold_id=gold_id,
        gold_rank=gold_rank,
        in_context=in_context,
    )
    gold_payload: dict[str, object] | None = None
    if gold is not None:
        gold_payload = {
            "doc_id": gold.span.doc_id,
            "page_no": gold.span.page_no,
            "chunk_index": gold.span.chunk_index,
            "chunk_id": gold.chunk_id,
            "quote": gold.span.quote,
        }
    return QuestionOutcome(
        id=question.id,
        question=question.question,
        kind=question.kind,
        answerable=question.answerable,
        gold=gold_payload,
        retrieved=in_context if question.answerable else None,
        gold_rank=gold_rank if question.answerable else None,
        refused=refused,
        answered=answered,
        refusal_reason=reason,
        claims=claims,
        citations=citations,
        citations_correct=citations_correct,
        claims_faithful=claims_faithful,
        stage_failed=stage,
        schema_failures=schema_failures,
        top_rerank_score=top_score,
        refusal_probability=probability,
        refusal_threshold=response.calibration.threshold if response is not None else None,
        latency_ms=round(latency_ms, 1),
        error=error,
        model_requested=model_requested,
        model_answered=model_answered,
        structured_mode=structured_mode,
        failure=failure,
    )


def _stage_failure(
    *,
    stage: str | None,
    response: AnswerResponse | None,
    model_failure: dict[str, object] | None,
    gold_id: int | None,
    gold_rank: int | None,
    in_context: bool,
) -> dict[str, object] | None:
    """Record a measured reason for every stage that failed.

    Retrieval/fusion/rerank losses are structural (the gold chunk is absent);
    generation and schema losses carry the wire and validation diagnostics from
    :class:`~docqa.answer.FailureDetail` rather than a bare ``error: null``.
    """
    if stage is None:
        return None
    if stage in {"retrieval", "fusion", "rerank"}:
        structural: dict[str, object] = {
            "stage": stage,
            "reason": "gold_chunk_absent",
            "gold_chunk_id": gold_id,
            "gold_rank": gold_rank,
            "in_context": in_context,
        }
        return structural
    if stage in {"schema", "validation"}:
        failure: dict[str, object] = (
            dict(response.failure.as_dict())
            if response is not None and response.failure is not None
            else {"reason": f"{stage}_failure_unrecorded"}
        )
        failure["stage"] = stage
        return failure
    if response is None:
        failure = (
            dict(model_failure)
            if model_failure is not None
            else {"reason": "model_call_error"}
        )
        failure["stage"] = stage
        return failure
    generation: dict[str, object] = {
        "stage": stage,
        "reason": response.reason or "no_claims",
        "model_reason": response.model_reason,
    }
    return generation


# --------------------------------------------------------------------------- #
# Aggregation and gates
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Aggregate:
    """Aggregate metrics, each a binomial rate with its denominator."""

    retrieval_recall: Rate
    answer_rate: Rate
    unanswerable_refusal_rate: Rate
    citation_precision: Rate
    faithfulness: Rate
    stage_counts: dict[str, int]
    structured_mode_counts: dict[str, int] = field(default_factory=dict)
    failure_reason_counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self, *, k: int = 8) -> dict[str, object]:
        """JSON-safe aggregate block."""
        return {
            f"retrieval_recall_at_{k}": self.retrieval_recall.as_dict(),
            "answer_rate_answerable": self.answer_rate.as_dict(),
            "unanswerable_refusal_rate": self.unanswerable_refusal_rate.as_dict(),
            "citation_precision": {
                **self.citation_precision.as_dict(),
                "kind": RULE_BASED_LABEL,
            },
            "faithfulness": {
                **self.faithfulness.as_dict(),
                "kind": RULE_BASED_LABEL,
            },
            "stage_failed_counts": dict(self.stage_counts),
            "structured_mode_counts": dict(self.structured_mode_counts),
            "stage_failure_reasons": dict(self.failure_reason_counts),
        }


def aggregate(outcomes: Sequence[QuestionOutcome]) -> Aggregate:
    """Aggregate per-question rows into rates and stage counts."""
    answerable = [row for row in outcomes if row.answerable]
    unanswerable = [row for row in outcomes if not row.answerable]
    return Aggregate(
        retrieval_recall=Rate(
            sum(1 for row in answerable if row.retrieved is True), len(answerable)
        ),
        answer_rate=Rate(sum(1 for row in answerable if row.answered), len(answerable)),
        unanswerable_refusal_rate=Rate(
            sum(1 for row in unanswerable if row.refused), len(unanswerable)
        ),
        citation_precision=Rate(
            sum(row.citations_correct for row in outcomes),
            sum(row.citations for row in outcomes),
        ),
        faithfulness=Rate(
            sum(row.claims_faithful for row in outcomes),
            sum(row.claims for row in outcomes),
        ),
        stage_counts={stage: 0 for stage in STAGES}
        | dict(Counter(row.stage_failed for row in outcomes if row.stage_failed is not None)),
        structured_mode_counts=dict(
            Counter(
                row.structured_mode
                for row in outcomes
                if row.structured_mode is not None
            )
        ),
        failure_reason_counts=dict(
            Counter(
                str(row.failure.get("reason"))
                for row in outcomes
                if row.failure is not None and row.failure.get("reason")
            )
        ),
    )


@dataclass(frozen=True)
class GateResult:
    """One enforced threshold."""

    name: str
    rate: Rate
    threshold: float
    passed: bool
    detail: str

    def as_dict(self) -> dict[str, object]:
        """JSON-safe gate row."""
        return {
            "name": self.name,
            **self.rate.as_dict(),
            "threshold": self.threshold,
            "passed": self.passed,
            "detail": self.detail,
        }


def _gate(name: str, rate: Rate, threshold: float) -> GateResult:
    if rate.n == 0:
        return GateResult(
            name=name,
            rate=rate,
            threshold=threshold,
            passed=False,
            detail="no measurable denominator; a gate cannot pass vacuously",
        )
    value = rate.value
    if value is None:  # pragma: no cover - guarded by the empty-denominator branch.
        raise EvalError("rate has a denominator but no value")
    passed = value >= threshold
    return GateResult(
        name=name,
        rate=rate,
        threshold=threshold,
        passed=passed,
        detail=f"{value:.3f} {'>=' if passed else '<'} {threshold:.3f}",
    )


def build_gates(metrics: Aggregate, *, k: int = 8) -> tuple[GateResult, ...]:
    """Apply every per-metric gate to an aggregate."""
    return (
        _gate(
            f"retrieval_recall_at_{k}", metrics.retrieval_recall, GATE_RETRIEVAL_RECALL
        ),
        _gate("answer_rate_answerable", metrics.answer_rate, GATE_ANSWER_RATE),
        _gate(
            "unanswerable_refusal_rate",
            metrics.unanswerable_refusal_rate,
            GATE_UNANSWERABLE_REFUSAL,
        ),
        _gate(
            "citation_precision", metrics.citation_precision, GATE_CITATION_PRECISION
        ),
        _gate("faithfulness", metrics.faithfulness, GATE_FAITHFULNESS),
    )


def exit_code(gates: Sequence[GateResult]) -> int:
    """0 when every gate passes, ``EXIT_GATE`` otherwise."""
    return 0 if all(gate.passed for gate in gates) else EXIT_GATE


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EvalConfig:
    """What the harness is measuring in this run."""

    mode: RetrievalMode = "hybrid_rerank"
    k: int = 8
    label: str = "production"
    ablated: bool = False
    # Diagnostic: bypass the calibrated refusal so the model's own answer rate
    # is measurable. The report records ``gate_run: false`` and must not be
    # quoted as a production gate outcome.
    force_generation: bool = False


@dataclass(frozen=True)
class EvalReport:
    """The written report plus the gates that decide the process exit code."""

    payload: dict[str, object]
    gates: tuple[GateResult, ...]

    @property
    def passed(self) -> bool:
        """True when every gate passed."""
        return all(gate.passed for gate in self.gates)


def _judge_provenance(
    judge_status: JudgeStatus, *, n_questions: int
) -> dict[str, object]:
    """Describe how citation precision and faithfulness were computed.

    The rule-based checker is always the reported metric. The judge block says
    whether the pinned judge id answered, with every attempt's HTTP status, and
    why no Cohen's kappa is computed.
    """
    return {
        "kind": RULE_BASED_LABEL,
        "model_judge_used": False,
        "status": judge_status.status,
        "model": judge_status.model,
        "attempts": [attempt.as_dict() for attempt in judge_status.attempts],
        "retries": judge_status.retries,
        "reason": judge_status.reason,
        "kappa": kappa_decision(n_questions),
        "metrics": ["citation_precision", "faithfulness"],
        "rule": (
            "citation precision: a citation is correct when it names the gold "
            "chunk; faithfulness: every citation on a claim names the gold chunk "
            "and its verbatim quote contains, or is contained in, the gold quote"
        ),
    }


def _judge_model(settings: Settings) -> tuple[str | None, str | None]:
    """Read the pinned judge id; return ``(model, problem)``."""
    judge_artifact = settings.calibration_path.parent / "judge.yaml"
    try:
        protocol = read_flat_yaml(judge_artifact)
    except OSError as exc:
        return None, f"judge protocol is not readable: {exc}"
    except ValueError as exc:
        return None, f"judge protocol is invalid: {exc}"
    model = protocol.get("model", "").strip()
    if not model:
        return None, "eval/judge.yaml carries no model id"
    return model, None


async def _judge_status(settings: Settings, n_questions: int) -> JudgeStatus:
    """Probe the pinned judge id, or record why it could not be probed."""
    model, problem = _judge_model(settings)
    if problem is not None:
        return JudgeStatus(
            status="judge_protocol_invalid",
            model=None,
            attempts=(),
            retries=0,
            reason=f"{problem}; the rule-based metric stays the reported one",
        )
    assert model is not None
    if not settings.model_configured:
        return JudgeStatus(
            status="judge_not_configured",
            model=model,
            attempts=(),
            retries=0,
            reason=(
                "the judge is pinned but no model endpoint is configured; the "
                "rule-based metric stays the reported one"
            ),
        )
    return await probe_judge(settings, model)


async def _selected_retrieval(
    question: str,
    config: EvalConfig,
    *,
    cached: RetrievalResult | None,
    embedder: Embedder,
    source: PgCandidateSource,
    reranker: Reranker,
) -> RetrievalResult:
    """Reuse the top-8 +rerank result when the selected path is exactly that."""
    if cached is not None:
        return cached
    return await retrieve_with_trace(
        question,
        config.k,
        embedder=embedder,
        source=source,
        reranker=reranker,
        mode=config.mode,
    )


async def evaluate(
    config: EvalConfig | None = None,
    *,
    settings: Settings | None = None,
    gold_path: Path = DEFAULT_GOLD_PATH,
) -> EvalReport:
    """Run the fixed question set and return the report payload and gates."""
    resolved_config = config or EvalConfig()
    resolved_settings = settings or get_settings()
    force_generation = (
        resolved_config.force_generation or resolved_settings.force_generation
    )
    questions = load_gold(gold_path)
    choice = build_embedder(resolved_settings)
    reranker_choice = build_reranker(
        mode=os.environ.get("DOCQA_RERANKER", "auto"),
        cache_dir=(
            str(resolved_settings.hf_cache_dir)
            if resolved_settings.hf_cache_dir
            else None
        ),
    )
    calibration = load_calibration(resolved_settings.calibration_path)
    model = build_model(resolved_settings)
    judge_status = await _judge_status(resolved_settings, len(questions))
    write_judge_status(
        resolved_settings.calibration_path.parent / "judge-status.json",
        judge_status,
    )
    requested_models = tuple(getattr(model, "model_chain", (resolved_settings.model_name,)))
    endpoint_host = (
        urlparse(resolved_settings.model_base_url).hostname
        if resolved_settings.model_base_url
        else None
    )

    engine = make_engine(resolved_settings)
    factory = make_session_factory(engine)
    outcomes: list[QuestionOutcome] = []
    ablation_hits: dict[str, int] = {label: 0 for label, _ in RETRIEVAL_ABLATION}
    run_started = datetime.now(UTC)
    monotonic_start = time.monotonic()
    rate_limit_failures = 0
    blocked_reason: str | None = None
    try:
        async with factory() as session:
            gold = await resolve_gold(questions, session)
            source = PgCandidateSource(session)
            for question in questions:
                retrieval_by_mode: dict[RetrievalMode, RetrievalResult] = {}
                for _, mode in RETRIEVAL_ABLATION:
                    retrieval_by_mode[mode] = await retrieve_with_trace(
                        question.question,
                        8,
                        embedder=choice.embedder,
                        source=source,
                        reranker=reranker_choice.reranker,
                        mode=mode,
                    )
                for label, mode in RETRIEVAL_ABLATION:
                    if question.answerable and question.id in gold:
                        target = gold[question.id].chunk_id
                        context = {chunk.chunk_id for chunk in retrieval_by_mode[mode].chunks}
                        ablation_hits[label] += int(target in context)

                started = datetime.now(UTC)
                selected = await _selected_retrieval(
                    question.question,
                    resolved_config,
                    cached=(
                        retrieval_by_mode.get(resolved_config.mode)
                        if resolved_config.k == 8
                        else None
                    ),
                    embedder=choice.embedder,
                    source=source,
                    reranker=reranker_choice.reranker,
                )
                error: str | None = None
                model_failure: dict[str, object] | None = None
                response: AnswerResponse | None
                try:
                    response = await _answer(
                        question,
                        selected,
                        model,
                        calibration,
                        force_generation=force_generation,
                    )
                except RateLimitedError as exc:
                    response = None
                    error = str(exc)
                    model_failure = (
                        exc.failure.as_dict() if exc.failure is not None else None
                    )
                    rate_limit_failures += 1
                    if rate_limit_failures >= 3:
                        blocked_reason = (
                            "three exhausted HTTP 429 responses in a row; the run "
                            "is marked blocked rather than reported as a model result"
                        )
                except ModelCallError as exc:
                    response = None
                    error = str(exc)
                    model_failure = (
                        exc.failure.as_dict() if exc.failure is not None else None
                    )
                else:
                    rate_limit_failures = 0
                elapsed_ms = (datetime.now(UTC) - started).total_seconds() * 1000.0
                outcomes.append(
                    _build_outcome(
                        question,
                        gold.get(question.id),
                        selected,
                        response,
                        mode=resolved_config.mode,
                        latency_ms=elapsed_ms,
                        requested_model=getattr(
                            model, "requested_model", resolved_settings.model_name
                        ),
                        error=error,
                        model_failure=model_failure,
                    )
                )
                if blocked_reason is not None:
                    break
    finally:
        await engine.dispose()

    finished_at = datetime.now(UTC)
    wall_clock_s = time.monotonic() - monotonic_start
    stats = getattr(model, "stats", None)
    model_stats = None if stats is None else stats.as_dict()

    metrics = aggregate(outcomes)
    gates = build_gates(metrics, k=resolved_config.k)
    retrieval_rows = [
        {
            "configuration": label,
            "mode": mode,
            "k": 8,
            "n": metrics.retrieval_recall.n,
            "recall": Rate(ablation_hits[label], metrics.retrieval_recall.n).as_dict(),
        }
        for label, mode in RETRIEVAL_ABLATION
    ]
    payload: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "provenance": {
            "questions": len(questions),
            "answerable": sum(1 for question in questions if question.answerable),
            "unanswerable": sum(1 for question in questions if not question.answerable),
            "gold_path": str(gold_path.relative_to(PROJECT_ROOT))
            if gold_path.is_relative_to(PROJECT_ROOT)
            else str(gold_path),
            "gold_sha256": gold_sha256(gold_path),
            "gold_provenance": (
                "hand-authored from the 260 ingested chunks; every quote is checked "
                "against the live chunk text before measurement"
            ),
            "judge": _judge_provenance(judge_status, n_questions=len(questions)),
            "generator": {
                "name": model.name,
                "structured_output": model.structured_output,
                "endpoint": resolved_settings.model_base_url,
                "endpoint_host": endpoint_host,
                "requested_models": list(requested_models),
                "model_min_interval_s": resolved_settings.model_min_interval_s,
                "model_timeout_s": resolved_settings.model_timeout_s,
                "model_max_retries": resolved_settings.model_max_retries,
                "model_retry_backoff_s": resolved_settings.model_retry_backoff_s,
                "observed_answering_models": sorted(
                    {
                        row.model_answered
                        for row in outcomes
                        if row.model_answered is not None
                    }
                ),
            },
            "embedder": choice.embedder.name,
            "reranker": reranker_choice.reranker.name,
            "reranker_fallback_reason": reranker_choice.fallback_reason,
            "calibration": {
                "threshold": calibration.threshold,
                "fit_set_sha256": calibration.fit_set_sha256,
                "provenance": calibration.provenance,
                "uncalibrated": calibration.uncalibrated,
            },
            "run": {
                "label": resolved_config.label,
                "mode": resolved_config.mode,
                "k": resolved_config.k,
                "ablated": resolved_config.ablated,
                "force_generation": force_generation,
                "gate_run": not force_generation,
                "started_at": run_started.isoformat(),
                "finished_at": finished_at.isoformat(),
                "wall_clock_s": round(wall_clock_s, 1),
                "wall_clock_min": round(wall_clock_s / 60.0, 2),
                "questions_measured": len(outcomes),
                "model_calls": model_stats,
                "model_min_interval_s": resolved_settings.model_min_interval_s,
                "rate_limit_failures": rate_limit_failures,
                "blocked": blocked_reason,
            },
        },
        "questions": [row.as_dict() for row in outcomes],
        "metrics": metrics.as_dict(k=resolved_config.k),
        "retrieval_ablation": retrieval_rows,
        "answer_path": {
            "configuration": (
                f"{resolved_config.label}: {resolved_config.mode} top-{resolved_config.k}"
            ),
            "answer_rate_answerable": metrics.answer_rate.as_dict(),
            "unanswerable_refusal_rate": metrics.unanswerable_refusal_rate.as_dict(),
            "citation_precision": metrics.citation_precision.as_dict(),
            "faithfulness": metrics.faithfulness.as_dict(),
            "kind": RULE_BASED_LABEL,
        },
        "gates": [gate.as_dict() for gate in gates],
        "passed": all(gate.passed for gate in gates),
    }
    return EvalReport(payload=payload, gates=gates)


async def _answer(
    question: EvalQuestion,
    retrieval: RetrievalResult,
    model: StructuredModel,
    calibration: Calibration,
    *,
    force_generation: bool = False,
) -> AnswerResponse:
    """Call the answer path with the retrieval result already in hand."""
    return await answer_question(
        question.question,
        retrieval.chunks,
        model=model,
        calibration=calibration,
        force_generation=force_generation,
    )


def write_report(report: EvalReport, path: Path) -> None:
    """Write the JSON report, creating the parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def render(report: EvalReport) -> str:
    """Human-readable summary with every rate, denominator and interval."""
    payload = report.payload
    provenance = cast(dict[str, object], payload["provenance"])
    metrics = cast(dict[str, object], payload["metrics"])
    lines = [
        (
            f"Questions: {provenance['questions']} "
            f"({provenance['answerable']} answerable, "
            f"{provenance['unanswerable']} unanswerable)"
        ),
        "Retrieval recall (answerable questions):",
    ]
    for row in cast(list[dict[str, object]], payload["retrieval_ablation"]):
        recall = cast(dict[str, object], row["recall"])
        lines.append(
            f"  {cast(str, row['configuration']):24s} "
            f"{recall['successes']}/{recall['n']} "
            f"({cast(float, recall['rate']):.1%}) "
            f"[{cast(list[float], recall['wilson_95'])[0]:.1%}, "
            f"{cast(list[float], recall['wilson_95'])[1]:.1%}]"
        )
    run = cast(dict[str, object], provenance["run"])
    gate_run = bool(run.get("gate_run", True))
    if not gate_run:
        lines.append(
            "DIAGNOSTIC RUN (not a gate run): the calibrated refusal was "
            "bypassed; the gates below describe this diagnostic only."
        )
    lines.append(
        f"Answer path ({run['label']}: {run['mode']} top-{run['k']}; "
        f"generator: {cast(dict[str, object], provenance['generator'])['name']}):"
    )
    lines.append(
        f"Wall clock: {run.get('wall_clock_s')} s "
        f"({run.get('wall_clock_min')} min); "
        f"questions measured: {run.get('questions_measured')}; "
        f"model calls: {run.get('model_calls')}"
    )
    if run.get("blocked"):
        lines.append(f"RUN BLOCKED: {run['blocked']}")
    for key, label in (
        ("answer_rate_answerable", "answer rate (answerable)"),
        ("unanswerable_refusal_rate", "refusal rate (unanswerable)"),
        ("citation_precision", f"citation precision ({RULE_BASED_LABEL})"),
        ("faithfulness", f"faithfulness ({RULE_BASED_LABEL})"),
    ):
        lines.append(f"  {label:44s} {format_rate(_rate_from(metrics[key]))}")
    stage_counts = cast(dict[str, int], metrics["stage_failed_counts"])
    lines.append(
        "Stage failures: "
        + ", ".join(f"{stage}={stage_counts.get(stage, 0)}" for stage in STAGES)
    )
    failure_reasons = cast(dict[str, int], metrics.get("stage_failure_reasons", {}))
    if failure_reasons:
        lines.append(
            "Stage failure reasons: "
            + ", ".join(
                f"{reason}={count}" for reason, count in sorted(failure_reasons.items())
            )
        )
    mode_counts = cast(dict[str, int], metrics.get("structured_mode_counts", {}))
    if mode_counts:
        lines.append(
            "Structured modes: "
            + ", ".join(
                f"{mode}={count}" for mode, count in sorted(mode_counts.items())
            )
        )
    lines.append("Gates:" if gate_run else "Gates (diagnostic only):")
    for gate in report.gates:
        state = "PASS" if gate.passed else "FAIL"
        lines.append(
            f"  {state} {gate.name:26s} {format_rate(gate.rate)} "
            f"threshold {gate.threshold:.0%}"
        )
    failed = sum(1 for gate in report.gates if not gate.passed)
    lines.append(
        f"Result: {'all gates passed' if failed == 0 else f'{failed} gate(s) failed'}"
    )
    return "\n".join(lines)


def _rate_from(block: object) -> Rate:
    record = cast(dict[str, object], block)
    return Rate(successes=cast(int, record["successes"]), n=cast(int, record["n"]))

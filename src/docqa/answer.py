"""Validated, cited answers with a calibrated refusal.

Generation is deliberately **not** streamed. The model produces one
``submit_answer`` payload; this module validates it against the retrieved
context before any claim can reach a caller:

* every claim must carry at least one citation id that exists in the context;
* every quote must be a normalised substring of every chunk it cites;
* a schema failure gets exactly one repair turn quoting the validation errors;
* a second failure becomes ``insufficient_evidence`` with the closest two
  passages.

The refusal threshold is read from ``eval/calibration.json`` at boot: a Platt
scaling ``p = sigmoid(a * score + b)`` and a balanced-accuracy threshold. The
loader refuses to construct a service when the file is missing unless
``allow_uncalibrated=True`` is passed explicitly, and that state is carried into
every response.

``force_generation=True`` is a diagnostic escape hatch: it bypasses only the
low-score refusal so the model's own answer rate can be measured. Citation and
schema validation still run unchanged, and every forced response says so.

With no ``DOCQA_MODEL_BASE_URL`` configured, :func:`build_model` returns
:class:`ExtractiveModel`: a deterministic, clearly labelled fallback that still
emits verbatim quotes through the same server-side validation.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import http.server
import json
import logging
import math
import re
import threading
import time
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar, Literal, Protocol, cast, runtime_checkable

import httpx
import json_repair

from docqa.config import Settings, StructuredOutputMode
from docqa.retrieval import RetrievedChunk

LOGGER = logging.getLogger(__name__)

SUBMIT_ANSWER_TOOL = "submit_answer"
MAX_QUOTE_CHARS = 240
MAX_REPAIR_TURNS = 1
CLOSEST_PASSAGES = 2

RefusalReason = Literal["low_score", "validation_failed", "insufficient_context"]
DropReason = Literal[
    "uncited", "unknown_citation", "quote_mismatch", "bad_quote"
]

_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_QUOTE_TRANSLATION = str.maketrans(
    {
        "‘": "'",  # noqa: RUF001 - typographic apostrophe
        "’": "'",  # noqa: RUF001 - typographic apostrophe
        "“": '"',
        "”": '"',
        "–": "-",  # noqa: RUF001 - en dash
        "—": "-",
    }
)

_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "can",
        "could", "did", "do", "does", "for", "from", "had", "has", "have", "how",
        "in", "into", "is", "it", "its", "may", "might", "of", "on", "or", "should",
        "than", "that", "the", "their", "them", "then", "there", "these", "they",
        "this", "those", "to", "was", "were", "what", "when", "where", "which",
        "who", "why", "will", "with", "would",
    }
)

_ANSWER_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "citation_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "quote": {"type": "string"},
                },
                "required": ["text", "citation_ids", "quote"],
                "additionalProperties": False,
            },
        },
        "insufficient_evidence": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["claims", "insufficient_evidence"],
    "additionalProperties": False,
}


class CalibrationError(RuntimeError):
    """The calibration artifact exists but is not usable."""


class CalibrationUnavailable(CalibrationError):
    """No calibration file exists and the caller did not allow an uncalibrated boot."""


class ModelCallError(RuntimeError):
    """The configured model endpoint could not be reached or returned no body."""


class RateLimitedError(ModelCallError):
    """The endpoint kept returning HTTP 429 after the configured retries.

    The API maps this to a readable ``rate_limited`` state; the eval records it
    per question rather than turning it into a 500 or a silent fallback.
    """

    def __init__(
        self,
        message: str,
        *,
        endpoint: str,
        attempts: int,
        retry_after_s: float | None = None,
    ) -> None:
        """Carry the endpoint, attempt count, and any server hint."""
        super().__init__(message)
        self.endpoint = endpoint
        self.attempts = attempts
        self.retry_after_s = retry_after_s


class AnswerSchemaError(RuntimeError):
    """A model payload does not match the answer schema; ``errors`` is quotable."""

    def __init__(self, errors: Sequence[str]) -> None:
        """Store the human-readable validation errors in order."""
        super().__init__("; ".join(errors))
        self.errors: tuple[str, ...] = tuple(errors)


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CalibrationInfo:
    """The calibration provenance copied into every response."""

    fit_set_sha256: str | None
    a: float | None
    b: float | None
    threshold: float | None
    uncalibrated: bool
    provenance: str


@dataclass(frozen=True)
class Calibration:
    """Platt scaling plus the refusal threshold from ``eval/calibration.json``."""

    a: float | None
    b: float | None
    threshold: float | None
    fit_set_sha256: str | None
    uncalibrated: bool
    provenance: str

    @property
    def calibrated(self) -> bool:
        """True when a fitted threshold is available."""
        return (
            not self.uncalibrated
            and self.a is not None
            and self.b is not None
            and self.threshold is not None
        )

    def probability(self, score: float) -> float | None:
        """Return the calibrated probability that the top passage answers the query."""
        if self.a is None or self.b is None:
            return None
        return _sigmoid(self.a * score + self.b)

    def info(self) -> CalibrationInfo:
        """Return the response-safe provenance record."""
        return CalibrationInfo(
            fit_set_sha256=self.fit_set_sha256,
            a=self.a,
            b=self.b,
            threshold=self.threshold,
            uncalibrated=self.uncalibrated,
            provenance=self.provenance,
        )

    @classmethod
    def uncalibrated_default(cls) -> Calibration:
        """Return the explicit uncalibrated mode: validation still gates."""
        return cls(
            a=None,
            b=None,
            threshold=None,
            fit_set_sha256=None,
            uncalibrated=True,
            provenance="uncalibrated: --uncalibrated was passed; no score threshold is applied",
        )


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationError(f"calibration field {name!r} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise CalibrationError(f"calibration field {name!r} must be finite")
    return number


def load_calibration(
    path: Path, *, allow_uncalibrated: bool = False
) -> Calibration:
    """Load and validate the calibration artifact.

    A missing file is fatal unless ``allow_uncalibrated`` is true. A malformed
    file is always fatal: silently substituting a guessed threshold would break
    invariant 4.
    """
    if not path.is_file():
        if allow_uncalibrated:
            LOGGER.warning(
                "no calibration at %s; starting uncalibrated because "
                "--uncalibrated was passed",
                path,
            )
            return Calibration.uncalibrated_default()
        raise CalibrationUnavailable(
            f"no calibration file at {path}; run scripts/fit_calibration.py or "
            "pass --uncalibrated (recorded in every response)"
        )
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CalibrationError(f"cannot read calibration at {path}: {exc}") from exc
    data = _as_dict(raw)
    if data is None:
        raise CalibrationError("calibration file must contain a JSON object")
    missing = [
        name
        for name in ("a", "b", "threshold", "fit_set_sha256")
        if name not in data
    ]
    if missing:
        raise CalibrationError(
            f"calibration is missing required field(s): {', '.join(missing)}"
        )
    a = _finite_number(data["a"], "a")
    b = _finite_number(data["b"], "b")
    threshold = _finite_number(data["threshold"], "threshold")
    if a == 0.0:
        raise CalibrationError("calibration slope 'a' must be non-zero")
    if not 0.0 <= threshold <= 1.0:
        raise CalibrationError("calibration threshold must be in [0, 1]")
    fit_hash = data["fit_set_sha256"]
    if not isinstance(fit_hash, str) or _SHA256_RE.fullmatch(fit_hash) is None:
        raise CalibrationError(
            "calibration fit_set_sha256 must be 64 lower-case hex characters"
        )
    provenance = data.get("provenance")
    return Calibration(
        a=a,
        b=b,
        threshold=threshold,
        fit_set_sha256=fit_hash,
        uncalibrated=False,
        provenance=provenance if isinstance(provenance, str) else "calibration file",
    )


# --------------------------------------------------------------------------- #
# Public answer types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Citation:
    """A resolved span: the chunk's provenance plus the verbatim quoted text."""

    chunk_id: int
    doc_id: str
    page_no: int
    char_start: int
    char_end: int
    page_text_sha256: str
    quote: str


@dataclass(frozen=True)
class AnswerClaim:
    """A claim that survived validation, with the citations supporting it."""

    text: str
    citations: tuple[Citation, ...]


@dataclass(frozen=True)
class Passage:
    """One of the closest passages shown when the answer is a refusal."""

    chunk_id: int
    doc_id: str
    page_no: int
    breadcrumb: str
    text: str
    score: float | None


@dataclass(frozen=True)
class ValidationReport:
    """Counts that make the citation gate observable rather than implicit."""

    claims_returned: int = 0
    claims_kept: int = 0
    dropped_uncited: int = 0
    dropped_unknown_citation: int = 0
    dropped_quote_mismatch: int = 0
    dropped_bad_quote: int = 0
    schema_failures: int = 0
    repair_turns: int = 0

    @property
    def claims_dropped(self) -> int:
        """Total claims the server refused to show."""
        return (
            self.dropped_uncited
            + self.dropped_unknown_citation
            + self.dropped_quote_mismatch
            + self.dropped_bad_quote
        )


@dataclass(frozen=True)
class AnswerResponse:
    """The only shape the API returns; every claim here has passed validation."""

    claims: tuple[AnswerClaim, ...]
    insufficient_evidence: bool
    reason: RefusalReason | None
    passages: tuple[Passage, ...]
    model: str
    structured_output: Literal["tool", "json", "extractive"]
    validation: ValidationReport
    refusal_probability: float | None
    calibration: CalibrationInfo
    uncalibrated: bool
    model_reason: str | None = None
    forced_generation: bool = False
    # The id sent and the id the provider reported. They differ when a
    # ``-latest`` alias moved or when the documented retry model answered.
    model_requested: str | None = None
    model_answered: str | None = None


# --------------------------------------------------------------------------- #
# Answer schema
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RawClaim:
    """A claim as the model submitted it, before any context check."""

    text: str
    citation_ids: tuple[str, ...]
    quote: str


@dataclass(frozen=True)
class RawAnswer:
    """A schema-valid model payload, before citation validation."""

    claims: tuple[RawClaim, ...]
    insufficient_evidence: bool
    reason: str | None


def _as_dict(value: object) -> dict[str, object] | None:
    if isinstance(value, dict):
        return cast(dict[str, object], value)
    return None


def _as_list(value: object) -> list[object] | None:
    if isinstance(value, list):
        return cast(list[object], value)
    return None


def _citation_key(value: object) -> str | None:
    """Canonical citation id: integers and numeric strings name the same chunk."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("c", "C")) and stripped[1:].isdigit():
            stripped = stripped[1:]
        return stripped or None
    return None


def parse_answer_payload(payload: object) -> RawAnswer:
    """Validate the model payload against the answer schema.

    Raises :class:`AnswerSchemaError` with a list of errors suitable for quoting
    back to the model in the single repair turn.
    """
    errors: list[str] = []
    data = _as_dict(payload)
    if data is None:
        raise AnswerSchemaError(["the response is not a JSON object"])

    insufficient_value = data.get("insufficient_evidence")
    insufficient = insufficient_value if isinstance(insufficient_value, bool) else None
    if insufficient is None:
        errors.append("insufficient_evidence must be a boolean")

    reason_value = data.get("reason")
    reason: str | None = None
    if reason_value is not None:
        if isinstance(reason_value, str):
            reason = reason_value
        else:
            errors.append("reason, when present, must be a string")

    claims_value = data.get("claims")
    claims_list = _as_list(claims_value)
    if claims_list is None:
        errors.append("claims must be an array")
        claims_list = []

    claims: list[RawClaim] = []
    for index, item in enumerate(claims_list):
        label = f"claims[{index}]"
        claim = _as_dict(item)
        if claim is None:
            errors.append(f"{label} must be an object")
            continue
        text = claim.get("text")
        if not isinstance(text, str) or not text.strip():
            errors.append(f"{label}.text must be a non-empty string")
            continue
        quote = claim.get("quote")
        if not isinstance(quote, str) or not quote.strip():
            errors.append(f"{label}.quote must be a non-empty string")
            continue
        ids_value = _as_list(claim.get("citation_ids"))
        if ids_value is None:
            errors.append(f"{label}.citation_ids must be an array")
            continue
        ids: list[str] = []
        for position, value in enumerate(ids_value):
            key = _citation_key(value)
            if key is None:
                errors.append(
                    f"{label}.citation_ids[{position}] must be an integer or "
                    "numeric string"
                )
                break
            ids.append(key)
        if len(ids) != len(ids_value):
            continue
        claims.append(
            RawClaim(text=text.strip(), citation_ids=tuple(ids), quote=quote)
        )

    if insufficient is False and not claims:
        errors.append("claims must not be empty when insufficient_evidence is false")

    if errors:
        raise AnswerSchemaError(errors)
    if insufficient is None:  # pragma: no cover - guarded by the error above.
        raise AnswerSchemaError(["insufficient_evidence must be a boolean"])
    return RawAnswer(
        claims=tuple(claims), insufficient_evidence=insufficient, reason=reason
    )


# --------------------------------------------------------------------------- #
# Citation integrity
# --------------------------------------------------------------------------- #


def normalise_text(text: str) -> str:
    """Normalise for quote matching: NFKC, canonical quotes, collapsed space, casefold."""
    value = unicodedata.normalize("NFKC", text)
    value = value.translate(_QUOTE_TRANSLATION)
    return re.sub(r"\s+", " ", value).strip().casefold()


@dataclass(frozen=True)
class _ClaimCounts:
    kept: int = 0
    uncited: int = 0
    unknown_citation: int = 0
    quote_mismatch: int = 0
    bad_quote: int = 0


def _drop(reason: DropReason, claim: RawClaim) -> None:
    """Record a rejected claim; the counter is derived from the log and the report."""
    LOGGER.warning(
        "dropped claim reason=%s citation_ids=%s text=%r",
        reason,
        list(claim.citation_ids),
        claim.text[:120],
    )


def validate_claims(
    claims: Sequence[RawClaim], chunks: Sequence[RetrievedChunk]
) -> tuple[list[AnswerClaim], _ClaimCounts]:
    """Keep only claims whose ids exist in context and whose quote is verbatim."""
    if not chunks:
        for claim in claims:
            _drop("unknown_citation", claim)
        return [], _ClaimCounts(unknown_citation=len(claims))

    by_id = {str(chunk.chunk_id): chunk for chunk in chunks}
    kept: list[AnswerClaim] = []
    kept_count = uncited = unknown = mismatch = bad_quote = 0

    for claim in claims:
        if not claim.citation_ids:
            uncited += 1
            _drop("uncited", claim)
            continue

        cited: list[RetrievedChunk] = []
        missing = False
        for citation_id in claim.citation_ids:
            chunk = by_id.get(citation_id)
            if chunk is None:
                missing = True
                break
            cited.append(chunk)
        if missing:
            unknown += 1
            _drop("unknown_citation", claim)
            continue

        normalised_quote = normalise_text(claim.quote)
        if len(normalised_quote) > MAX_QUOTE_CHARS:
            bad_quote += 1
            _drop("bad_quote", claim)
            continue
        if not all(normalised_quote in normalise_text(chunk.text) for chunk in cited):
            mismatch += 1
            _drop("quote_mismatch", claim)
            continue

        kept.append(
            AnswerClaim(
                text=claim.text,
                citations=tuple(
                    Citation(
                        chunk_id=chunk.chunk_id,
                        doc_id=chunk.doc_id,
                        page_no=chunk.page_no,
                        char_start=chunk.char_start,
                        char_end=chunk.char_end,
                        page_text_sha256=chunk.page_text_sha256,
                        quote=claim.quote,
                    )
                    for chunk in cited
                ),
            )
        )
        kept_count += 1

    return kept, _ClaimCounts(
        kept=kept_count,
        uncited=uncited,
        unknown_citation=unknown,
        quote_mismatch=mismatch,
        bad_quote=bad_quote,
    )


# --------------------------------------------------------------------------- #
# Model interface
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelRequest:
    """Everything a model implementation may use for one answer attempt."""

    question: str
    system_prompt: str
    user_prompt: str
    chunks: tuple[RetrievedChunk, ...]
    repair_errors: tuple[str, ...] = ()
    previous_response: str = ""


@dataclass(frozen=True)
class ModelReply:
    """A raw model response, parsed only as far as a JSON payload."""

    payload: object | None
    raw_text: str
    via_tool_call: bool
    error: str | None = None
    # The id the provider reports in the response body. ``-latest`` aliases are
    # not stable, so the eval records this next to the id it requested.
    model: str | None = None
    http_status: int | None = None


@runtime_checkable
class StructuredModel(Protocol):
    """A model that answers through the ``submit_answer`` contract."""

    name: str
    structured_output: Literal["tool", "json", "extractive"]

    async def complete(self, request: ModelRequest) -> ModelReply:
        """Return one answer payload (or an error describing why none was produced)."""
        ...


def build_prompts(question: str, chunks: Sequence[RetrievedChunk]) -> tuple[str, str]:
    """Build the system and user prompts; passage ids are the citation ids."""
    system = (
        "You answer questions about a fixed PDF corpus. You must call the "
        f"{SUBMIT_ANSWER_TOOL} tool exactly once.\n"
        "Rules:\n"
        "1. Every claim must carry at least one passage id in citation_ids; "
        "ids are the integers in square brackets before each passage.\n"
        "2. quote must be copied character-for-character from every passage you "
        "cite. Never paraphrase, repair or reformat a quote.\n"
        f"3. A quote is at most {MAX_QUOTE_CHARS} characters; choose a contiguous span.\n"
        "4. If the passages do not support an answer, return empty claims with "
        "insufficient_evidence=true and a short reason.\n"
        "5. Do not use outside knowledge. Do not make a claim you cannot quote."
    )
    blocks: list[str] = []
    for chunk in chunks:
        blocks.append(
            f"[{chunk.chunk_id}] {chunk.doc_id} | page {chunk.page_no} | "
            f"{chunk.breadcrumb}\n{chunk.text}"
        )
    user = (
        f"Question: {question}\n\n"
        "Passages:\n\n" + "\n\n".join(blocks)
    )
    return system, user


def _repair_messages(request: ModelRequest) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": request.system_prompt},
        {"role": "user", "content": request.user_prompt},
    ]
    if request.repair_errors:
        messages.append(
            {"role": "assistant", "content": request.previous_response or "(empty)"}
        )
        errors = "\n".join(f"- {error}" for error in request.repair_errors)
        messages.append(
            {
                "role": "user",
                "content": (
                    "Your previous response failed validation:\n"
                    f"{errors}\n\n"
                    f"Correct it and call {SUBMIT_ANSWER_TOOL} again. Copy quotes "
                    "verbatim from a passage; do not invent text."
                ),
            }
        )
    return messages


class ExtractiveModel:
    """Deterministic fallback used when no model endpoint is configured.

    It emits the same payload shape as a model would, so the server-side quote
    and citation validation still runs. Its output is labelled
    ``extractive fallback (no model key)`` in every response.
    """

    name = "extractive fallback (no model key)"
    structured_output: Literal["tool", "json", "extractive"] = "extractive"

    async def complete(self, request: ModelRequest) -> ModelReply:
        """Select verbatim sentences that overlap the question, or refuse."""
        terms = {
            term
            for term in (word.lower() for word in _WORD_RE.findall(request.question))
            if term not in _STOPWORDS and len(term) > 2
        }
        candidates: list[tuple[int, int, str, int]] = []
        for chunk in request.chunks:
            for sentence in _SENTENCE_RE.split(chunk.text):
                span = _quote_span(sentence)
                if span is None:
                    continue
                overlap = len(terms & {word.lower() for word in _WORD_RE.findall(span)})
                if overlap:
                    candidates.append((-overlap, chunk.final_rank, span, chunk.chunk_id))
        candidates.sort()

        claims: list[dict[str, object]] = []
        used_chunks: set[int] = set()
        for _, _, span, chunk_id in candidates:
            if chunk_id in used_chunks or len(claims) >= 3:
                continue
            used_chunks.add(chunk_id)
            claims.append(
                {"text": span, "citation_ids": [str(chunk_id)], "quote": span}
            )

        if claims:
            payload: object = {"claims": claims, "insufficient_evidence": False}
        else:
            payload = {
                "claims": [],
                "insufficient_evidence": True,
                "reason": "no sentence in the retrieved passages overlaps the question",
            }
        return ModelReply(
            payload=payload,
            raw_text=json.dumps(payload, sort_keys=True),
            via_tool_call=False,
        )


def _quote_span(sentence: str) -> str | None:
    """Return a contiguous span of ``sentence`` that is at most the quote cap."""
    stripped = sentence.strip()
    if not stripped:
        return None
    if len(stripped) <= MAX_QUOTE_CHARS:
        return stripped
    clipped = stripped[:MAX_QUOTE_CHARS]
    boundary = clipped.rfind(" ")
    if boundary >= MAX_QUOTE_CHARS // 2:
        clipped = clipped[:boundary]
    return clipped.strip() or None


def parse_json_lenient(raw: str) -> object | None:
    """Parse JSON, repairing the common malformed shapes a gateway produces."""
    text = raw.strip()
    if not text:
        return None
    try:
        parsed: object = json.loads(text)
        return parsed
    except ValueError:
        pass
    try:
        repaired: object = json_repair.loads(text)
    except Exception:  # json-repair raises several unrelated exception types.
        return None
    if repaired in ("", None):
        return None
    return repaired


@dataclass
class ModelCallStats:
    """Wire-level call accounting copied into the eval report."""

    requests: int = 0
    retries: int = 0
    rate_limited: int = 0
    server_errors: int = 0
    timeouts: int = 0

    def as_dict(self) -> dict[str, int]:
        """JSON-safe counters."""
        return {
            "requests": self.requests,
            "retries": self.retries,
            "rate_limited": self.rate_limited,
            "server_errors": self.server_errors,
            "timeouts": self.timeouts,
        }

    def add(self, other: ModelCallStats) -> None:
        """Accumulate another client's counters (used by the failover chain)."""
        self.requests += other.requests
        self.retries += other.retries
        self.rate_limited += other.rate_limited
        self.server_errors += other.server_errors
        self.timeouts += other.timeouts


class _EndpointGate:
    """One lock and one start timestamp per endpoint.

    The gate is process-wide so concurrent API requests to the same free tier
    cannot overlap: one call in flight, and at least ``min_interval_s`` between
    request starts.
    """

    __slots__ = ("last_start", "lock")

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.last_start = 0.0


_GATES: dict[str, _EndpointGate] = {}
_GATES_LOCK = threading.Lock()


def _endpoint_gate(base_url: str) -> _EndpointGate:
    """Return the shared gate for ``base_url``, creating it if needed."""
    with _GATES_LOCK:
        gate = _GATES.get(base_url)
        if gate is None:
            gate = _EndpointGate()
            _GATES[base_url] = gate
        return gate


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """Read a numeric ``Retry-After`` header when the server sends one."""
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        value = float(raw.strip())
    except ValueError:
        return None
    return value if value >= 0 else None


class OpenAICompatibleModel:
    """Chat-completions client with a forced tool call and a JSON-only fallback.

    Calls to one endpoint are serialised and spaced by ``min_interval_s``
    (default 31 s for the free keyless tier). A 429 is retried ``max_retries``
    times with exponential backoff; an exhausted call raises
    :class:`RateLimitedError` instead of pretending the model answered.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        *,
        structured_output: StructuredOutputMode = "tool",
        api_key: str | None = None,
        timeout_s: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
        min_interval_s: float = 0.0,
        max_retries: int = 2,
        retry_backoff_s: float = 5.0,
    ) -> None:
        """Configure the endpoint; nothing is contacted until ``complete`` runs."""
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._transport = transport
        self._min_interval_s = max(0.0, min_interval_s)
        self._max_retries = max(0, max_retries)
        self._retry_backoff_s = max(0.0, retry_backoff_s)
        self.stats = ModelCallStats()
        self.name = f"openai-compatible:{model_name}"
        self.structured_output: Literal["tool", "json", "extractive"] = (
            "json" if structured_output == "json" else "tool"
        )

    @property
    def endpoint(self) -> str:
        """The endpoint this client writes to."""
        return self._base_url

    @property
    def model_name(self) -> str:
        """The exact model id sent in the request body."""
        return self._model_name

    @property
    def requested_model(self) -> str:
        """The primary id, for the eval report."""
        return self._model_name

    @property
    def model_chain(self) -> tuple[str, ...]:
        """Return the single id this client sends."""
        return (self._model_name,)

    def _body(self, request: ModelRequest) -> dict[str, object]:
        body: dict[str, object] = {
            "model": self._model_name,
            "messages": _repair_messages(request),
            "temperature": 0,
        }
        if self.structured_output == "tool":
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": SUBMIT_ANSWER_TOOL,
                        "description": (
                            "Submit the answer. Every claim must cite passage ids "
                            "and quote a verbatim span from each cited passage."
                        ),
                        "parameters": _ANSWER_SCHEMA,
                    },
                }
            ]
            body["tool_choice"] = {
                "type": "function",
                "function": {"name": SUBMIT_ANSWER_TOOL},
            }
        else:
            body["messages"] = [
                *cast(list[dict[str, str]], body["messages"]),
                {
                    "role": "user",
                    "content": (
                        "Return only one JSON object with keys claims, "
                        "insufficient_evidence and optionally reason. No prose, "
                        "no markdown fences."
                    ),
                },
            ]
        return body

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def _send(
        self, body: dict[str, object], headers: dict[str, str]
    ) -> tuple[httpx.Response | None, str | None, bool]:
        """One HTTP attempt; transport failures become a readable error string.

        The third value is True for a timeout, which — like HTTP 429 and 503 —
        is retried with exponential backoff instead of failing the question.
        """
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout_s, transport=self._transport
            ) as client:
                response = await client.post(
                    f"{self._base_url}/chat/completions", json=body, headers=headers
                )
                return response, None, False
        except httpx.TimeoutException as exc:
            return None, f"{type(exc).__name__}: {exc}", True
        except httpx.HTTPError as exc:
            return None, f"{type(exc).__name__}: {exc}", False

    async def post_once(
        self, body: dict[str, object]
    ) -> tuple[httpx.Response | None, str | None, bool]:
        """One attempt through the shared endpoint gate, with no retry logic.

        The judge availability probe needs the raw HTTP status of every
        attempt, so it calls this directly instead of :meth:`_post`.
        """
        gate = (
            _endpoint_gate(self._base_url) if self._min_interval_s > 0 else None
        )
        if gate is None:
            self.stats.requests += 1
            return await self._send(body, self._headers())
        async with gate.lock:
            delay = gate.last_start + self._min_interval_s - time.monotonic()
            if delay > 0:
                LOGGER.debug(
                    "waiting %.1fs for the %s rate limit", delay, self._base_url
                )
                await asyncio.sleep(delay)
            gate.last_start = time.monotonic()
            self.stats.requests += 1
            return await self._send(body, self._headers())

    @staticmethod
    def _retry_delay(
        attempt: int, backoff_s: float, retry_after: float | None
    ) -> float:
        exponential: float = backoff_s * float(2 ** (attempt - 1))
        return max(exponential, retry_after or 0.0)

    async def _post(self, body: dict[str, object]) -> object:
        """POST with serialisation, spacing, and exponential backoff.

        HTTP 429 and 503 and request timeouts are retried ``max_retries``
        times; an exhausted 429 raises :class:`RateLimitedError` and any other
        exhausted failure raises :class:`ModelCallError`.
        """
        attempt = 0
        while True:
            response, error, timed_out = await self.post_once(body)

            if error is not None:
                if timed_out and attempt < self._max_retries:
                    attempt += 1
                    self.stats.retries += 1
                    self.stats.timeouts += 1
                    delay = self._retry_delay(
                        attempt, self._retry_backoff_s, None
                    )
                    LOGGER.warning(
                        "timeout from %s; retry %d/%d in %.1fs",
                        self._base_url,
                        attempt,
                        self._max_retries,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                if timed_out:
                    self.stats.timeouts += 1
                raise ModelCallError(error)
            if response is None:  # pragma: no cover - _send never returns this.
                raise ModelCallError("empty HTTP response")

            retry_after = _retry_after_seconds(response)
            retryable_status = response.status_code in {429, 503}
            if retryable_status and attempt < self._max_retries:
                attempt += 1
                self.stats.retries += 1
                if response.status_code == 503:
                    self.stats.server_errors += 1
                delay = self._retry_delay(
                    attempt, self._retry_backoff_s, retry_after
                )
                LOGGER.warning(
                    "HTTP %d from %s; retry %d/%d in %.1fs",
                    response.status_code,
                    self._base_url,
                    attempt,
                    self._max_retries,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            if response.status_code == 429:
                self.stats.rate_limited += 1
                raise RateLimitedError(
                    "rate_limited: "
                    f"{self._base_url} returned HTTP 429 after "
                    f"{attempt + 1} attempt(s)"
                    + (
                        f"; server asked for {retry_after:g}s"
                        if retry_after is not None
                        else ""
                    ),
                    endpoint=self._base_url,
                    attempts=attempt + 1,
                    retry_after_s=retry_after,
                )
            if response.status_code >= 400:
                if response.status_code == 503:
                    self.stats.server_errors += 1
                raise ModelCallError(
                    f"HTTP {response.status_code}: {response.text[:300]}"
                )
            try:
                parsed: object = response.json()
            except ValueError as exc:
                raise ModelCallError(f"response was not JSON: {exc}") from exc
            return parsed

    async def complete(self, request: ModelRequest) -> ModelReply:
        """Post one chat-completions request and extract the answer payload."""
        try:
            body = await self._post(self._body(request))
        except RateLimitedError:
            raise
        except ModelCallError as exc:
            return ModelReply(
                payload=None,
                raw_text="",
                via_tool_call=False,
                error=str(exc),
            )
        return _reply_from_body(body, expect_tool=self.structured_output == "tool")


class FailoverModel:
    """Try the documented model chain in order, then fail loudly.

    The generator has one documented retry id (the free tier retires and
    overloads model ids independently). Each entry in the chain is a complete
    client call with its own 429/503/timeout retries; the reply keeps the id the
    provider reported so the eval records which model actually answered.
    """

    def __init__(self, models: Sequence[OpenAICompatibleModel]) -> None:
        """Build a chain from at least one client; order is the retry order."""
        if not models:
            raise ValueError("a model chain needs at least one client")
        self._models = tuple(models)
        self.structured_output: Literal["tool", "json", "extractive"] = (
            self._models[0].structured_output
        )
        primary = self._models[0].model_name
        retries = [model.model_name for model in self._models[1:]]
        self.name = f"openai-compatible:{primary}" + (
            f" (retry {', '.join(retries)})" if retries else ""
        )

    @property
    def requested_model(self) -> str:
        """The primary model id sent first."""
        return self._models[0].model_name

    @property
    def model_chain(self) -> tuple[str, ...]:
        """Every model id in the chain, in order."""
        return tuple(model.model_name for model in self._models)

    @property
    def endpoint(self) -> str:
        """The shared endpoint the chain writes to."""
        return self._models[0].endpoint

    @property
    def stats(self) -> ModelCallStats:
        """Aggregate wire counters over every model in the chain."""
        total = ModelCallStats()
        for model in self._models:
            total.add(model.stats)
        return total

    async def complete(self, request: ModelRequest) -> ModelReply:
        """Return the first usable reply, or raise with every failure named."""
        failures: list[str] = []
        last_rate_limited = False
        for model in self._models:
            try:
                reply = await model.complete(request)
            except RateLimitedError as exc:
                last_rate_limited = True
                failures.append(f"{model.model_name}: {exc}")
                continue
            last_rate_limited = False
            if reply.error is None and reply.payload is not None:
                return reply
            failures.append(
                f"{model.model_name}: {reply.error or 'no usable payload'}"
            )
        message = "all models in the chain failed: " + "; ".join(failures)
        if last_rate_limited:
            raise RateLimitedError(
                message,
                endpoint=self.endpoint,
                attempts=len(self._models),
            )
        raise ModelCallError(message)


def _string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _reply_from_body(body: object, *, expect_tool: bool) -> ModelReply:
    """Extract ``submit_answer`` arguments (or content JSON) from a chat response."""
    data = _as_dict(body)
    if data is None:
        return ModelReply(None, "", False, "response body is not a JSON object")
    reported_model = _string(data.get("model"))
    error_field = _string(data.get("error"))
    if error_field is not None:
        return ModelReply(None, "", False, f"gateway error: {error_field}")
    choices = _as_list(data.get("choices"))
    if not choices:
        return ModelReply(None, "", False, "response has no choices")
    first = _as_dict(choices[0])
    message = _as_dict(first.get("message")) if first is not None else None
    if message is None:
        return ModelReply(None, "", False, "response choice has no message")

    tool_calls = _as_list(message.get("tool_calls"))
    if tool_calls:
        call = _as_dict(tool_calls[0])
        function = _as_dict(call.get("function")) if call is not None else None
        if function is None:
            return ModelReply(None, "", False, "tool call has no function object")
        name = _string(function.get("name")) or ""
        arguments = function.get("arguments")
        raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
        payload = parse_json_lenient(raw)
        error = None if payload is not None else "tool arguments were not valid JSON"
        return ModelReply(
            payload=payload,
            raw_text=raw,
            via_tool_call=name == SUBMIT_ANSWER_TOOL,
            error=error,
            model=reported_model,
            http_status=200,
        )

    content = _string(message.get("content"))
    if content is None or not content.strip():
        expectation = "tool call" if expect_tool else "JSON content"
        return ModelReply(None, "", False, f"response contained no {expectation}")
    payload = parse_json_lenient(content)
    error = None if payload is not None else "message content was not valid JSON"
    return ModelReply(
        payload=payload,
        raw_text=content,
        via_tool_call=False,
        error=error,
        model=reported_model,
        http_status=200,
    )


# --------------------------------------------------------------------------- #
# Answer orchestration
# --------------------------------------------------------------------------- #


async def answer_question(
    question: str,
    chunks: Sequence[RetrievedChunk],
    *,
    model: StructuredModel,
    calibration: Calibration,
    force_generation: bool = False,
) -> AnswerResponse:
    """Retrieve-to-answer validation: refuse, call the model, drop bad claims.

    ``force_generation`` skips the calibrated low-score refusal only; an empty
    context is still a refusal, and the model's claims still face the same
    citation contract.
    """
    context = tuple(chunks)
    closest = tuple(_passage(chunk) for chunk in context[:CLOSEST_PASSAGES])
    top_score = context[0].rerank_score if context else None
    probability = (
        calibration.probability(top_score) if top_score is not None else None
    )
    info = calibration.info()
    requested_model = getattr(model, "requested_model", model.name)
    answered_model: str | None = None

    if not force_generation and calibration.calibrated and probability is not None:
        threshold = calibration.threshold
        if threshold is not None and probability < threshold:
            return _refusal(
                "low_score",
                closest,
                model=model,
                info=info,
                probability=probability,
                validation=ValidationReport(),
                force_generation=force_generation,
            )
    if not context:
        return _refusal(
            "low_score",
            (),
            model=model,
            info=info,
            probability=probability,
            validation=ValidationReport(),
            force_generation=force_generation,
        )

    system, user = build_prompts(question, context)
    request = ModelRequest(
        question=question,
        system_prompt=system,
        user_prompt=user,
        chunks=context,
    )
    reply = await _call(model, request)
    answered_model = reply.model
    raw, errors = _parse_reply(reply)
    schema_failures = 1 if errors else 0
    repair_turns = 0

    if errors:
        repair_turns = MAX_REPAIR_TURNS
        repair_request = dataclasses.replace(
            request,
            repair_errors=errors,
            previous_response=reply.raw_text,
        )
        reply = await _call(model, repair_request)
        answered_model = reply.model
        raw, errors = _parse_reply(reply)
        if errors:
            schema_failures = 2
            LOGGER.warning(
                "answer schema failed twice; refusing with closest passages: %s",
                "; ".join(errors),
            )

    if raw is None:
        return _refusal(
            "validation_failed",
            closest,
            model=model,
            info=info,
            probability=probability,
            validation=ValidationReport(
                schema_failures=schema_failures, repair_turns=repair_turns
            ),
            force_generation=force_generation,
            model_requested=requested_model,
            model_answered=answered_model,
        )

    if raw.insufficient_evidence:
        return _refusal(
            "insufficient_context",
            closest,
            model=model,
            info=info,
            probability=probability,
            validation=ValidationReport(
                claims_returned=len(raw.claims),
                schema_failures=schema_failures,
                repair_turns=repair_turns,
            ),
            model_reason=raw.reason,
            force_generation=force_generation,
            model_requested=requested_model,
            model_answered=answered_model,
        )

    kept, counts = validate_claims(raw.claims, context)
    report = ValidationReport(
        claims_returned=len(raw.claims),
        claims_kept=counts.kept,
        dropped_uncited=counts.uncited,
        dropped_unknown_citation=counts.unknown_citation,
        dropped_quote_mismatch=counts.quote_mismatch,
        dropped_bad_quote=counts.bad_quote,
        schema_failures=schema_failures,
        repair_turns=repair_turns,
    )
    if not kept:
        return _refusal(
            "validation_failed",
            closest,
            model=model,
            info=info,
            probability=probability,
            validation=report,
            force_generation=force_generation,
            model_requested=requested_model,
            model_answered=answered_model,
        )
    return AnswerResponse(
        claims=tuple(kept),
        insufficient_evidence=False,
        reason=None,
        passages=(),
        model=model.name,
        structured_output=model.structured_output,
        validation=report,
        refusal_probability=probability,
        calibration=info,
        uncalibrated=calibration.uncalibrated,
        forced_generation=force_generation,
        model_requested=requested_model,
        model_answered=answered_model,
    )


async def _call(model: StructuredModel, request: ModelRequest) -> ModelReply:
    reply = await model.complete(request)
    if reply.error is not None:
        raise ModelCallError(reply.error)
    return reply


def _parse_reply(reply: ModelReply) -> tuple[RawAnswer | None, tuple[str, ...]]:
    try:
        return parse_answer_payload(reply.payload), ()
    except AnswerSchemaError as exc:
        return None, exc.errors


def _passage(chunk: RetrievedChunk) -> Passage:
    score = chunk.rerank_score if chunk.rerank_score is not None else chunk.rrf_score
    return Passage(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        page_no=chunk.page_no,
        breadcrumb=chunk.breadcrumb,
        text=chunk.text,
        score=score,
    )


def _refusal(
    reason: RefusalReason,
    passages: tuple[Passage, ...],
    *,
    model: StructuredModel,
    info: CalibrationInfo,
    probability: float | None,
    validation: ValidationReport,
    model_reason: str | None = None,
    force_generation: bool = False,
    model_requested: str | None = None,
    model_answered: str | None = None,
) -> AnswerResponse:
    return AnswerResponse(
        claims=(),
        insufficient_evidence=True,
        reason=reason,
        passages=passages,
        model=model.name,
        structured_output=model.structured_output,
        validation=validation,
        refusal_probability=probability,
        calibration=info,
        uncalibrated=info.uncalibrated,
        model_reason=model_reason,
        forced_generation=force_generation,
        model_requested=model_requested,
        model_answered=model_answered,
    )


def build_model(settings: Settings) -> StructuredModel:
    """Pick the configured model chain, or the labelled extractive fallback."""
    if not settings.model_configured or settings.model_base_url is None:
        LOGGER.warning(
            "no DOCQA_MODEL_BASE_URL configured; using the extractive fallback "
            "(no model key)"
        )
        return ExtractiveModel()
    clients = [
        OpenAICompatibleModel(
            settings.model_base_url,
            name,
            structured_output=settings.structured_output,
            api_key=settings.model_api_key,
            timeout_s=settings.model_timeout_s,
            min_interval_s=settings.model_min_interval_s,
            max_retries=settings.model_max_retries,
            retry_backoff_s=settings.model_retry_backoff_s,
        )
        for name in settings.model_chain
    ]
    if len(clients) == 1:
        return clients[0]
    return FailoverModel(clients)


# --------------------------------------------------------------------------- #
# tool_choice probe
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProbeVerdict:
    """What an actual HTTP round trip said about structured-output support."""

    endpoint: str
    made_request: bool
    tool_choice_sent: bool
    tool_call_returned: bool | None
    detail: str

    @property
    def honoured(self) -> bool | None:
        """True when the gateway returned the forced call; None when unverified."""
        if not self.made_request:
            return None
        return self.tool_call_returned


async def probe_tool_choice(model: OpenAICompatibleModel) -> ProbeVerdict:
    """Force ``submit_answer`` with a real request and report what came back."""
    request = ModelRequest(
        question="Call submit_answer with no claims.",
        system_prompt=(
            f"Call {SUBMIT_ANSWER_TOOL} with claims=[] and "
            "insufficient_evidence=true."
        ),
        user_prompt="Probe only.",
        chunks=(),
    )
    body = model._body(request)  # the probe inspects the wire body it will send.
    tool_choice_sent = isinstance(body.get("tool_choice"), dict)
    try:
        payload = await model._post(body)
    except ModelCallError as exc:
        return ProbeVerdict(
            endpoint=model._base_url,
            made_request=True,
            tool_choice_sent=tool_choice_sent,
            tool_call_returned=None,
            detail=f"request failed: {exc}",
        )
    data = _as_dict(payload) or {}
    received = data.get("received_tool_choice")
    if received is not None:
        tool_choice_sent = isinstance(received, dict) and received.get(
            "type"
        ) == "function"
    choices = _as_list(data.get("choices")) or []
    message: dict[str, object] | None = None
    if choices:
        choice = _as_dict(choices[0])
        if choice is not None:
            message = _as_dict(choice.get("message"))
    tool_calls = _as_list(message.get("tool_calls")) if message is not None else None
    name: str | None = None
    if tool_calls:
        call = _as_dict(tool_calls[0])
        function = _as_dict(call.get("function")) if call is not None else None
        name = _string(function.get("name")) if function is not None else None
    if name == SUBMIT_ANSWER_TOOL:
        return ProbeVerdict(
            endpoint=model._base_url,
            made_request=True,
            tool_choice_sent=tool_choice_sent,
            tool_call_returned=True,
            detail=f"gateway returned the forced {SUBMIT_ANSWER_TOOL} call",
        )
    content = _string(message.get("content")) if message is not None else None
    detail = (
        "gateway ignored tool_choice and returned message content"
        if content
        else "gateway returned neither the forced call nor content"
    )
    return ProbeVerdict(
        endpoint=model._base_url,
        made_request=True,
        tool_choice_sent=tool_choice_sent,
        tool_call_returned=False,
        detail=detail,
    )


class _ProbeStubHandler(http.server.BaseHTTPRequestHandler):
    """Loopback OpenAI-compatible stub used only when no gateway is configured."""

    received: ClassVar[list[dict[str, object]]] = []

    def do_POST(self) -> None:
        """Record the request body and answer with a forced tool call."""
        length = int(self.headers.get("Content-Length", "0"))
        body: object = json.loads(self.rfile.read(length) or b"{}")
        parsed = _as_dict(body) or {}
        _ProbeStubHandler.received.append(parsed)
        reply = {
            "received_tool_choice": parsed.get("tool_choice"),
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "probe",
                                "type": "function",
                                "function": {
                                    "name": SUBMIT_ANSWER_TOOL,
                                    "arguments": (
                                        '{"claims": [], '
                                        '"insufficient_evidence": true}'
                                    ),
                                },
                            }
                        ],
                    }
                }
            ],
        }
        encoded = json.dumps(reply).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:
        """Silence the stub's access log."""


async def probe_local_client() -> ProbeVerdict:
    """Exercise the client against a real loopback HTTP server.

    This verifies serialisation and response parsing, not a hosted gateway. The
    verdict says so; ``docs/ENV.md`` records that the gateway itself is
    unverified until ``DOCQA_MODEL_BASE_URL`` points at one.
    """
    _ProbeStubHandler.received = []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _ProbeStubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host = str(server.server_address[0])
    port = int(server.server_address[1])
    base_url = f"http://{host}:{port}/v1"
    model = OpenAICompatibleModel(base_url, "probe-stub", structured_output="tool")
    try:
        verdict = await probe_tool_choice(model)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    sent = bool(_ProbeStubHandler.received) and isinstance(
        _ProbeStubHandler.received[0].get("tool_choice"), dict
    )
    return ProbeVerdict(
        endpoint="local loopback stub",
        made_request=verdict.made_request,
        tool_choice_sent=sent,
        tool_call_returned=verdict.tool_call_returned,
        detail=(
            "client transmitted tool_choice and parsed a submit_answer call over "
            "real HTTP; no hosted gateway is configured, so the gateway itself is "
            "unverified"
        ),
    )


@dataclass(frozen=True)
class StructuredProbeVerdict:
    """The recorded outcome of probing tool mode, then JSON mode if needed."""

    endpoint: str
    model: str
    mode: StructuredOutputMode
    tool_choice_sent: bool
    tool_call_returned: bool | None
    json_mode_ok: bool | None
    detail: str
    checked_at: str
    wall_clock_s: float

    def as_dict(self) -> dict[str, object]:
        """JSON-safe artifact, including the mode the service will use."""
        return {
            "endpoint": self.endpoint,
            "model": self.model,
            "mode": self.mode,
            "tool_choice_sent": self.tool_choice_sent,
            "tool_call_returned": self.tool_call_returned,
            "json_mode_ok": self.json_mode_ok,
            "detail": self.detail,
            "checked_at": self.checked_at,
            "wall_clock_s": round(self.wall_clock_s, 3),
        }


async def probe_structured_output(
    settings: Settings,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    artifact_path: Path | None = None,
) -> StructuredProbeVerdict:
    """Probe the configured endpoint and record tool or JSON mode.

    Tool calling is tried first with a real forced call. If the endpoint
    returns content instead, JSON mode is probed with a second real request and
    becomes the recorded mode. The artifact is what ``DOCQA_STRUCTURED_OUTPUT=auto``
    reads; a stale artifact for another endpoint or model is ignored.
    """
    if settings.model_base_url is None:
        raise ModelCallError(
            "no DOCQA_MODEL_BASE_URL configured; there is nothing to probe"
        )
    started = time.monotonic()
    tool_model = OpenAICompatibleModel(
        settings.model_base_url,
        settings.model_name,
        structured_output="tool",
        api_key=settings.model_api_key,
        timeout_s=settings.model_timeout_s,
        transport=transport,
        min_interval_s=settings.model_min_interval_s,
        max_retries=settings.model_max_retries,
        retry_backoff_s=settings.model_retry_backoff_s,
    )
    tool_verdict = await probe_tool_choice(tool_model)
    mode: StructuredOutputMode = "tool"
    json_mode_ok: bool | None = None
    detail = tool_verdict.detail

    if tool_verdict.honoured is not True:
        mode = "json"
        json_model = OpenAICompatibleModel(
            settings.model_base_url,
            settings.model_name,
            structured_output="json",
            api_key=settings.model_api_key,
            timeout_s=settings.model_timeout_s,
            transport=transport,
            min_interval_s=settings.model_min_interval_s,
            max_retries=settings.model_max_retries,
            retry_backoff_s=settings.model_retry_backoff_s,
        )
        request = ModelRequest(
            question="Probe JSON mode only.",
            system_prompt=(
                "Return one JSON object with claims=[] and "
                "insufficient_evidence=true. No prose."
            ),
            user_prompt="Probe only.",
            chunks=(),
        )
        try:
            reply = await json_model.complete(request)
            json_mode_ok = reply.error is None and isinstance(reply.payload, dict)
            detail = (
                f"{tool_verdict.detail}; JSON fallback "
                + (
                    "returned a parseable payload"
                    if json_mode_ok
                    else f"failed: {reply.error or 'no JSON payload'}"
                )
            )
        except RateLimitedError as exc:
            json_mode_ok = False
            detail = f"{tool_verdict.detail}; JSON fallback failed: {exc}"

    verdict = StructuredProbeVerdict(
        endpoint=tool_model.endpoint,
        model=settings.model_name,
        mode=mode,
        tool_choice_sent=tool_verdict.tool_choice_sent,
        tool_call_returned=tool_verdict.tool_call_returned,
        json_mode_ok=json_mode_ok,
        detail=detail,
        checked_at=datetime.now(UTC).isoformat(),
        wall_clock_s=time.monotonic() - started,
    )
    path = artifact_path or settings.structured_output_probe_path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(verdict.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return verdict


def calibration_fit_hash(pairs_path: Path) -> str:
    """Recompute the canonical fit-set hash written by ``fit_calibration.py``."""
    raw: object = json.loads(pairs_path.read_text(encoding="utf-8"))
    rows = _as_list(raw) or []
    canonical: list[dict[str, object]] = []
    for row in rows:
        record = _as_dict(row)
        if record is None:
            continue
        canonical.append(
            {
                "query": record.get("query"),
                "chunk_id": record.get("chunk_id"),
                "label": record.get("label"),
            }
        )
    ordered = sorted(
        canonical, key=lambda row: (str(row["query"]), str(row["chunk_id"]))
    )
    payload = json.dumps(
        ordered, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

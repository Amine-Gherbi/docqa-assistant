"""Judge availability: probe the pinned judge id, never substitute the generator.

Invariant 6 requires a judge model with a different exact id from the
generator. On the keyed free tier the *second* available id is the only
candidate, so this module asks it one real question and records the exact HTTP
status of every attempt. If it does not answer, the status is
``judge_model_unavailable`` and the deterministic rule-based metric stays the
reported one — a same-model judge is worse than an honest gap.

No score is produced here. Cohen's kappa needs real human labels of the
judge's binary decision; the 30 hand-labelled gold rows label *questions*, not
per-claim support decisions, so :func:`kappa_decision` records why no number is
computed instead of inventing one.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx

from docqa.answer import OpenAICompatibleModel, _retry_after_seconds
from docqa.config import Settings

LOGGER = logging.getLogger(__name__)

JudgeState = Literal[
    "judge_model_available",
    "judge_model_unavailable",
    "judge_model_conflict",
    "judge_protocol_invalid",
    "judge_not_configured",
]

DEFAULT_JUDGE_PROMPT = (
    "You are a blinded citation checker. You see only a claim and the text of "
    "the span it cites. Decide whether the cited span supports the claim. "
    "Reply with exactly one token: support or no_support."
)
DEFAULT_JUDGE_USER = (
    "Claim: The encoder is composed of six identical layers.\n"
    "Span: The encoder is composed of a stack of N = 6 identical layers.\n"
    "Reply with one token."
)


@dataclass(frozen=True)
class JudgeAttempt:
    """One real HTTP attempt against the judge id."""

    attempt: int
    http_status: int | None
    error: str | None

    def as_dict(self) -> dict[str, object]:
        """JSON-safe attempt row."""
        return {
            "attempt": self.attempt,
            "http_status": self.http_status,
            "error": self.error,
        }


@dataclass(frozen=True)
class JudgeStatus:
    """The recorded judge state and the evidence behind it."""

    status: JudgeState
    model: str | None
    attempts: tuple[JudgeAttempt, ...]
    retries: int
    reason: str

    @property
    def available(self) -> bool:
        """True only when the judge id answered a real request."""
        return self.status == "judge_model_available"

    def as_dict(self) -> dict[str, object]:
        """JSON-safe status block for the eval report and the artifact."""
        return {
            "status": self.status,
            "model": self.model,
            "attempts": [attempt.as_dict() for attempt in self.attempts],
            "retries": self.retries,
            "reason": self.reason,
        }


def kappa_decision(n_questions: int, *, n_judge_labels: int = 0) -> dict[str, object]:
    """State whether Cohen's kappa can be computed, and why.

    The gold file carries question-level labels (answerable or not, plus a
    supporting span). It carries no human judgement of *claim supported by the
    cited span*, which is the quantity a judge produces. Pairing the two would
    be a fabricated comparison, so no kappa is computed.
    """
    return {
        "computed": False,
        "n_hand_labelled_questions": n_questions,
        "n_human_judge_labels": n_judge_labels,
        "reason": (
            "Cohen's kappa needs human labels of the judge's binary decision "
            "(claim supported by the cited span). eval/gold.jsonl labels "
            f"{n_questions} questions, not per-claim support decisions; "
            "0 human judge labels exist, so no kappa is computed and no label "
            "was invented."
        ),
    }


def _status(
    state: JudgeState,
    model: str | None,
    attempts: list[JudgeAttempt],
    reason: str,
) -> JudgeStatus:
    return JudgeStatus(
        status=state,
        model=model,
        attempts=tuple(attempts),
        retries=max(0, len(attempts) - 1),
        reason=reason,
    )


def read_flat_yaml(path: Path) -> dict[str, str]:
    """Read the flat ``key: value`` judge protocol file.

    The judge artifact is intentionally a small flat file; rejecting nesting is
    better than silently mis-parsing a protocol that must be exact.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise OSError(f"cannot read {path}: {exc}") from exc
    values: dict[str, str] = {}
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        key, separator, value = line.partition(":")
        if not separator or not key.strip():
            raise ValueError(f"{path}:{number}: expected 'key: value'")
        key = key.strip()
        if key in values:
            raise ValueError(f"{path}:{number}: duplicate key {key!r}")
        values[key] = value.strip().strip("'\"")
    return values


def write_judge_status(path: Path, status: JudgeStatus) -> None:
    """Write the judge status artifact atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(status.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


async def probe_judge(
    settings: Settings,
    judge_model: str,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    system_prompt: str | None = None,
) -> JudgeStatus:
    """Ask the judge id one real question, recording every attempt's status.

    Retries use the same exponential backoff as the generator: only timeouts,
    HTTP 429 and HTTP 503 are retried. A 404 or 400 stops immediately because
    repeating it cannot change the answer.
    """
    if settings.model_base_url is None:
        return _status(
            "judge_not_configured",
            judge_model,
            [],
            "no DOCQA_MODEL_BASE_URL is configured, so no judge can be reached; "
            "citation precision and faithfulness stay rule-based",
        )
    if judge_model == settings.model_name:
        return _status(
            "judge_model_conflict",
            judge_model,
            [],
            f"judge model {judge_model!r} equals the generator model; "
            "invariant 6 forbids a same-model judge, so it was not probed and "
            "the generator is never substituted",
        )

    model = OpenAICompatibleModel(
        settings.model_base_url,
        judge_model,
        structured_output="json",
        api_key=settings.judge_api_key,
        timeout_s=settings.model_timeout_s,
        transport=transport,
        min_interval_s=settings.model_min_interval_s,
        max_retries=0,
        retry_backoff_s=settings.model_retry_backoff_s,
    )
    body: dict[str, object] = {
        "model": judge_model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt or DEFAULT_JUDGE_PROMPT,
            },
            {"role": "user", "content": DEFAULT_JUDGE_USER},
        ],
        "temperature": 0,
    }

    attempts: list[JudgeAttempt] = []
    retry_ids = settings.model_max_retries
    note = ""
    if judge_model in settings.model_fallback_names:
        note = (
            " Note: this id is also the generator's documented retry, so any "
            "answer served by that retry would share an id with the judge."
        )

    for number in range(1, retry_ids + 2):
        response, error, timed_out = await model.post_once(body)
        if error is not None:
            attempts.append(JudgeAttempt(number, None, error))
            retryable = timed_out
            if retryable and number <= retry_ids:
                delay = settings.model_retry_backoff_s * (2 ** (number - 1))
                LOGGER.warning(
                    "judge probe %s attempt %d failed (%s); retrying in %.1fs",
                    judge_model,
                    number,
                    error,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            break

        assert response is not None
        status_code = response.status_code
        attempts.append(JudgeAttempt(number, status_code, None))
        if status_code == 200:
            return _status(
                "judge_model_available",
                judge_model,
                attempts,
                f"judge model {judge_model} answered HTTP 200 on attempt "
                f"{number}. Reliability is still unmeasured: no human-labelled "
                "(claim, span) pairs exist, so Cohen's kappa is not computed "
                f"and the rule-based metric stays the reported one.{note}",
            )
        retryable = status_code in {429, 503}
        if not retryable:
            break
        if number > retry_ids:
            break
        retry_after = _retry_after_seconds(response)
        delay = max(
            settings.model_retry_backoff_s * (2 ** (number - 1)),
            retry_after or 0.0,
        )
        LOGGER.warning(
            "judge probe %s attempt %d returned HTTP %d; retrying in %.1fs",
            judge_model,
            number,
            status_code,
            delay,
        )
        await asyncio.sleep(delay)

    statuses = [str(attempt.http_status) for attempt in attempts]
    return _status(
        "judge_model_unavailable",
        judge_model,
        attempts,
        f"judge model {judge_model} did not answer after {len(attempts)} "
        f"attempt(s); HTTP statuses: [{', '.join(statuses)}]. The generator is "
        "never substituted as judge; citation precision and faithfulness stay "
        f"rule-based.{note}",
    )

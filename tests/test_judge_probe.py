"""The judge availability probe: exact per-attempt HTTP status, no substitution.

The probe only answers "does this judge model id answer on this key?". It never
produces a score, and it never falls back to the generator: an unavailable
judge is recorded as ``judge_model_unavailable`` and the rule-based metric
stays the reported one.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from docqa.config import get_settings
from docqa.judge import probe_judge


def _settings(monkeypatch: pytest.MonkeyPatch, *, base_url: str) -> object:
    monkeypatch.setenv("DOCQA_MODEL_BASE_URL", base_url)
    monkeypatch.setenv("DOCQA_MODEL_NAME", "gemini-flash-lite-latest")
    monkeypatch.setenv("DOCQA_MODEL_FALLBACK_NAMES", "gemini-3.8-flash")
    monkeypatch.setenv("DOCQA_MODEL_API_KEY", "test-key")
    monkeypatch.setenv("DOCQA_MODEL_MIN_INTERVAL_S", "0")
    monkeypatch.setenv("DOCQA_MODEL_MAX_RETRIES", "2")
    monkeypatch.setenv("DOCQA_MODEL_RETRY_BACKOFF_S", "0.001")
    return get_settings()


def test_judge_unavailable_records_every_http_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(monkeypatch, base_url="http://judge.invalid/v1")

    def handler(request: httpx.Request) -> httpx.Response:
        assert "test-key" not in str(request.url)
        return httpx.Response(503, text="high demand")

    status = asyncio.run(
        probe_judge(
            settings,  # type: ignore[arg-type]
            "gemini-3.8-flash",
            transport=httpx.MockTransport(handler),
        )
    )

    assert status.status == "judge_model_unavailable"
    assert [attempt.http_status for attempt in status.attempts] == [503, 503, 503]
    assert status.retries == 2
    assert "503" in status.reason


def test_judge_available_after_a_503(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, base_url="http://judge.invalid/v1")
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, text="high demand")
        return httpx.Response(
            200,
            json={
                "model": "gemini-3.8-flash",
                "choices": [
                    {"message": {"content": "support"}}
                ],
            },
        )

    status = asyncio.run(
        probe_judge(
            settings,  # type: ignore[arg-type]
            "gemini-3.8-flash",
            transport=httpx.MockTransport(handler),
        )
    )

    assert status.status == "judge_model_available"
    assert [attempt.http_status for attempt in status.attempts] == [503, 200]


def test_judge_equal_to_the_generator_is_never_probed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(monkeypatch, base_url="http://judge.invalid/v1")

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("a same-model judge must not be probed")

    status = asyncio.run(
        probe_judge(
            settings,  # type: ignore[arg-type]
            "gemini-flash-lite-latest",
            transport=httpx.MockTransport(handler),
        )
    )

    assert status.status == "judge_model_conflict"
    assert status.attempts == ()
    assert "invariant 6" in status.reason


def test_no_endpoint_reports_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DOCQA_MODEL_BASE_URL", raising=False)
    monkeypatch.setenv("DOCQA_MODEL_NAME", "gemini-flash-lite-latest")
    settings = get_settings()

    status = asyncio.run(probe_judge(settings, "gemini-3.8-flash"))

    assert status.status == "judge_not_configured"
    assert status.attempts == ()

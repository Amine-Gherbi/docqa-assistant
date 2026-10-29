"""The free-endpoint client, probe, cache, and rate-limit path.

No test here contacts the network. HTTP behaviour is exercised through
``httpx.MockTransport`` or the existing loopback stub, and every sleep is short
or replaced by mock transport.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

import httpx
import pytest
from fastapi.testclient import TestClient

from docqa.answer import (
    AnswerResponse,
    Calibration,
    ModelReply,
    ModelRequest,
    OpenAICompatibleModel,
    RateLimitedError,
    ValidationReport,
    probe_structured_output,
)
from docqa.api import PgRetriever, create_app
from docqa.cache import (
    AnswerCache,
    CachedAnswer,
    DemoQuestion,
    load_demo_questions,
    warm_demo_answers,
)
from docqa.config import get_settings, resolve_structured_output
from docqa.retrieval import RetrievedChunk


def _request() -> ModelRequest:
    return ModelRequest(question="q", system_prompt="s", user_prompt="u", chunks=())


def _tool_body() -> dict[str, object]:
    return {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "submit_answer",
                                "arguments": (
                                    '{"claims": [], "insufficient_evidence": true}'
                                ),
                            }
                        }
                    ]
                }
            }
        ]
    }


def _calibration(*, threshold: float = 0.5) -> Calibration:
    return Calibration(
        a=1.0,
        b=0.0,
        threshold=threshold,
        fit_set_sha256="b" * 64,
        uncalibrated=False,
        provenance="test fixture",
    )


def _chunk(chunk_id: int = 7, *, score: float = 4.0) -> RetrievedChunk:
    text = "The encoder maps tokens to continuous representations."
    return RetrievedChunk(
        chunk_id=chunk_id,
        chunk_index=chunk_id,
        doc_id="attention-is-all-you-need",
        page_no=3,
        char_start=100,
        char_end=100 + len(text),
        page_text_sha256="a" * 64,
        breadcrumb="3 Model Architecture",
        text=text,
        vector_rank=1,
        lexical_rank=None,
        rrf_score=0.03,
        rerank_score=score,
        final_rank=1,
    )


class NeverCalledModel:
    """A model that fails the test if a cache hit reaches it."""

    name = "fake:never-called"
    structured_output: Literal["tool", "json", "extractive"] = "tool"

    async def complete(self, request: ModelRequest) -> ModelReply:
        raise AssertionError("a cached question must not reach the model")


class RateLimitedModel:
    """A model whose endpoint has exhausted its 429 retries."""

    name = "openai-compatible:rate-limited"
    structured_output: Literal["tool", "json", "extractive"] = "tool"

    async def complete(self, request: ModelRequest) -> ModelReply:
        raise RateLimitedError(
            "rate_limited: https://example.invalid returned HTTP 429 after 3 attempt(s)",
            endpoint="https://example.invalid",
            attempts=3,
            retry_after_s=30.0,
        )


async def _retriever(question: str) -> list[RetrievedChunk]:
    return [_chunk()]


def test_calls_are_serialised_and_spaced() -> None:
    events: list[tuple[str, float]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        events.append(("start", time.monotonic()))
        await asyncio.sleep(0.05)
        events.append(("end", time.monotonic()))
        return httpx.Response(200, json=_tool_body())

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        first = OpenAICompatibleModel(
            "http://serialise.invalid/v1",
            "m",
            transport=transport,
            min_interval_s=0.1,
        )
        second = OpenAICompatibleModel(
            "http://serialise.invalid/v1",
            "m",
            transport=transport,
            min_interval_s=0.1,
        )
        await asyncio.gather(first.complete(_request()), second.complete(_request()))

    asyncio.run(run())

    assert [kind for kind, _ in events] == ["start", "end", "start", "end"]
    # One call in flight at a time, and request starts are at least the
    # configured interval apart (31 s in production).
    assert events[2][1] - events[0][1] >= 0.09


async def test_429_is_retried_twice_then_reported_as_rate_limited() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(429, headers={"Retry-After": "0"}, text="slow down")

    model = OpenAICompatibleModel(
        "http://limit.invalid/v1",
        "m",
        transport=httpx.MockTransport(handler),
        min_interval_s=0,
        max_retries=2,
        retry_backoff_s=0.001,
    )

    with pytest.raises(RateLimitedError) as raised:
        await model.complete(_request())

    assert calls == 3
    assert raised.value.attempts == 3
    assert raised.value.retry_after_s == 0.0
    assert "429" in str(raised.value)
    assert "3 attempt" in str(raised.value)
    assert model.stats.retries == 2
    assert model.stats.rate_limited == 1


async def test_429_then_success_returns_the_tool_call() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, text="slow down")
        return httpx.Response(200, json=_tool_body())

    model = OpenAICompatibleModel(
        "http://recover.invalid/v1",
        "m",
        transport=httpx.MockTransport(handler),
        min_interval_s=0,
        retry_backoff_s=0.001,
    )
    reply = await model.complete(_request())

    assert calls == 2
    assert reply.via_tool_call is True
    assert reply.payload == {"claims": [], "insufficient_evidence": True}


def test_probe_records_tool_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCQA_MODEL_BASE_URL", "https://probe.invalid/openai")
    monkeypatch.setenv("DOCQA_MODEL_NAME", "probe-model")
    monkeypatch.setenv("DOCQA_MODEL_MIN_INTERVAL_S", "0")
    settings = get_settings()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_tool_body())

    artifact = tmp_path / "structured-output.json"
    result = asyncio.run(
        probe_structured_output(
            settings,
            transport=httpx.MockTransport(handler),
            artifact_path=artifact,
        )
    )

    assert result.mode == "tool"
    assert result.tool_call_returned is True
    assert result.json_mode_ok is None
    stored = json.loads(artifact.read_text(encoding="utf-8"))
    assert stored["mode"] == "tool"
    assert stored["endpoint"] == "https://probe.invalid/openai"
    assert stored["model"] == "probe-model"


def test_probe_falls_back_to_json_and_records_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DOCQA_MODEL_BASE_URL", "https://json.invalid/openai")
    monkeypatch.setenv("DOCQA_MODEL_NAME", "probe-model")
    monkeypatch.setenv("DOCQA_MODEL_MIN_INTERVAL_S", "0")
    settings = get_settings()

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if "tools" in body:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": "I cannot call tools."}}
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"claims": [], "insufficient_evidence": true}'
                            )
                        }
                    }
                ]
            },
        )

    artifact = tmp_path / "structured-output.json"
    result = asyncio.run(
        probe_structured_output(
            settings,
            transport=httpx.MockTransport(handler),
            artifact_path=artifact,
        )
    )

    assert result.mode == "json"
    assert result.tool_call_returned is False
    assert result.json_mode_ok is True
    stored = json.loads(artifact.read_text(encoding="utf-8"))
    assert stored["mode"] == "json"
    assert resolve_structured_output(
        artifact, "https://json.invalid/openai", "probe-model"
    ) == ("json", f"probe:{artifact.name}")
    other = resolve_structured_output(
        artifact, "https://other.invalid/openai", "probe-model"
    )
    assert other[0] == "tool"


def test_force_generation_environment_flag_is_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DOCQA_FORCE_GENERATION", "1")
    assert get_settings().force_generation is True
    monkeypatch.setenv("DOCQA_FORCE_GENERATION", "off")
    assert get_settings().force_generation is False
    monkeypatch.setenv("DOCQA_FORCE_GENERATION", "sometimes")
    with pytest.raises(ValueError, match="DOCQA_FORCE_GENERATION"):
        get_settings()


def test_explicit_environment_mode_beats_the_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "structured-output.json"
    artifact.write_text(
        json.dumps(
            {
                "endpoint": "https://json.invalid/openai",
                "model": "probe-model",
                "mode": "json",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DOCQA_STRUCTURED_OUTPUT", "tool")
    assert resolve_structured_output(
        artifact, "https://json.invalid/openai", "probe-model"
    ) == ("tool", "environment")


def test_api_serves_a_cached_answer_with_model_and_date(tmp_path: Path) -> None:
    timestamp = "2026-10-23T12:00:00+00:00"
    entry = CachedAnswer(
        question="How many layers?",
        model="openai-compatible:openai-fast",
        generated_at=timestamp,
        structured_output="tool",
        response={
            "claims": [],
            "insufficient_evidence": True,
            "reason": "insufficient_context",
            "passages": [],
        },
        trace=None,
    )
    cache = AnswerCache(
        path=tmp_path / "demo-answers.json",
        generated_at=timestamp,
        model=entry.model,
        entries={entry.question: entry},
    )

    async def forbidden_retriever(question: str) -> list[RetrievedChunk]:
        raise AssertionError("a cache hit must not retrieve")

    app = create_app(
        calibration=_calibration(),
        model=NeverCalledModel(),
        retriever=forbidden_retriever,
        answer_cache=cache,
    )
    with TestClient(app) as client:
        response = client.post("/ask", json={"question": entry.question})

    assert response.status_code == 200
    body = response.json()
    assert body["cached"] is True
    assert body["cache"] == {"model": entry.model, "generated_at": timestamp}
    assert body["insufficient_evidence"] is True


def test_pg_retriever_passes_the_model_cache_to_the_reranker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The hosted image prefetches both models into DOCQA_HF_CACHE.

    The reranker is loaded on the first retrieval, not at construction, so the
    test asks the retriever to load it explicitly.
    """
    import asyncio

    from docqa.ingest.embedder import EmbedderChoice
    from docqa.rerank import RerankerChoice

    class FakeEmbedder:
        name = "fake:embedder"
        is_dummy = False
        dim = 384

        async def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] * 384 for _ in texts]

        async def embed_query(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] * 384 for _ in texts]

    class FakeReranker:
        name = "fake:reranker"
        is_dummy = False

        async def score(self, query: str, passages: list[str]) -> list[float]:
            return [0.0 for _ in passages]

    captured: dict[str, object] = {}

    def fake_build_reranker(**kwargs: object) -> RerankerChoice:
        captured.update(kwargs)
        return RerankerChoice(FakeReranker(), requested="fastembed", fallback_reason=None)

    monkeypatch.setenv("DOCQA_HF_CACHE", str(tmp_path))
    monkeypatch.delenv("DOCQA_RERANK_BATCH_SIZE", raising=False)
    monkeypatch.delenv("DOCQA_ONNX_MEMORY_ARENA", raising=False)
    monkeypatch.setattr(
        "docqa.api.build_embedder",
        lambda settings: EmbedderChoice(
            FakeEmbedder(), requested="fastembed", fallback_reason=None
        ),
    )
    monkeypatch.setattr("docqa.api.build_reranker", fake_build_reranker)
    monkeypatch.setattr("docqa.api.make_engine", lambda settings: object())
    monkeypatch.setattr("docqa.api.make_session_factory", lambda engine: object())

    retriever = PgRetriever(get_settings())
    assert captured == {}
    asyncio.run(retriever._ensure_reranker())

    assert captured["cache_dir"] == str(tmp_path)
    assert captured["batch_size"] == 1
    assert captured["memory_arena"] is False


def test_api_cors_uses_the_configured_ui_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hosted UI origin is admitted explicitly, not by a wildcard."""
    monkeypatch.setenv("DOCQA_CORS_ORIGINS", "https://marginalia-ui.onrender.com")
    app = create_app(
        calibration=_calibration(),
        model=NeverCalledModel(),
        retriever=_retriever,
        answer_cache=AnswerCache.empty(Path("missing-demo-answers.json")),
    )
    with TestClient(app) as client:
        response = client.options(
            "/ask",
            headers={
                "Origin": "https://marginalia-ui.onrender.com",
                "Access-Control-Request-Method": "POST",
            },
        )

    assert response.status_code == 200
    assert (
        response.headers["access-control-allow-origin"]
        == "https://marginalia-ui.onrender.com"
    )


def test_api_cache_miss_reports_rate_limited_not_500() -> None:
    app = create_app(
        calibration=_calibration(),
        model=RateLimitedModel(),
        retriever=_retriever,
        answer_cache=AnswerCache.empty(Path("missing-demo-answers.json")),
    )
    with TestClient(app) as client:
        response = client.post("/ask", json={"question": "Not a demo question"})

    assert response.status_code == 429
    body = response.json()
    assert body["state"] == "rate_limited"
    assert body["cached"] is False
    assert body["retry_after_s"] == 30.0
    assert "rate_limited" in body["detail"]
    assert response.headers["Retry-After"] == "30"


async def test_warm_writes_six_entries_and_reuses_them(tmp_path: Path) -> None:
    questions = tuple(
        DemoQuestion(id=f"Q{index}", question=f"question {index}", kind="answerable")
        for index in range(6)
    )
    calls = 0

    async def answer(question: str) -> tuple[AnswerResponse, None]:
        nonlocal calls
        calls += 1
        return (
            AnswerResponse(
                claims=(),
                insufficient_evidence=True,
                reason="insufficient_context",
                passages=(),
                model="fake:model",
                structured_output="tool",
                validation=ValidationReport(),
                refusal_probability=None,
                calibration=_calibration().info(),
                uncalibrated=False,
            ),
            None,
        )

    out = tmp_path / "demo-answers.json"
    await warm_demo_answers(
        questions,
        answer=answer,
        out_path=out,
        model_name="fake:model",
        calibration=_calibration(),
    )
    assert calls == 6
    stored = json.loads(out.read_text(encoding="utf-8"))
    assert len(stored["answers"]) == 6
    assert stored["model"] == "fake:model"
    for entry in stored["answers"]:
        assert entry["question"].startswith("question ")
        assert entry["model"] == "fake:model"
        datetime.fromisoformat(entry["generated_at"])

    async def forbidden(question: str) -> tuple[AnswerResponse, None]:
        raise AssertionError("a reuse run must not call the model")

    await warm_demo_answers(
        questions,
        answer=forbidden,
        out_path=out,
        model_name="fake:model",
        calibration=_calibration(),
        reuse=True,
    )
    assert len(json.loads(out.read_text(encoding="utf-8"))["answers"]) == 6


def test_demo_question_loader_requires_six(tmp_path: Path) -> None:
    path = tmp_path / "demo-questions.json"
    path.write_text(
        json.dumps(
            {
                "questions": [
                    {"id": "Q1", "question": "q", "kind": "answerable"}
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly 6"):
        load_demo_questions(path)

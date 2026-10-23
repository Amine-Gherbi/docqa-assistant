"""Refusal behaviour and the calibration loader.

The threshold is never typed into code: these tests prove it is read from the
artifact, that a missing artifact refuses to boot unless ``--uncalibrated`` is
explicit, and that the low-score refusal happens before the model is called.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

import httpx
import pytest
from fastapi.testclient import TestClient

from docqa.answer import (
    Calibration,
    CalibrationError,
    CalibrationUnavailable,
    ModelReply,
    ModelRequest,
    OpenAICompatibleModel,
    answer_question,
    calibration_fit_hash,
    load_calibration,
    probe_local_client,
)
from docqa.api import create_app
from docqa.config import PROJECT_ROOT
from docqa.retrieval import RetrievedChunk

QUOTE = "The encoder maps tokens to continuous representations."
CHUNK_TEXT = f"Introduction. {QUOTE} A decoder then attends to them."
SECOND_TEXT = "Attention weights are computed with scaled dot products."


def _chunk(
    chunk_id: int,
    *,
    text: str = CHUNK_TEXT,
    final_rank: int = 1,
    score: float = 4.0,
) -> RetrievedChunk:
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
        final_rank=final_rank,
    )


def _calibration(*, a: float = 1.0, b: float = 0.0, threshold: float = 0.5) -> Calibration:
    return Calibration(
        a=a,
        b=b,
        threshold=threshold,
        fit_set_sha256="b" * 64,
        uncalibrated=False,
        provenance="test fixture",
    )


class FixedModel:
    """Returns one payload and counts calls."""

    name = "fake:refusal"
    structured_output: Literal["tool", "json", "extractive"] = "tool"

    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.calls = 0

    async def complete(self, request: ModelRequest) -> ModelReply:
        self.calls += 1
        return ModelReply(
            payload=self.payload, raw_text=json.dumps(self.payload), via_tool_call=True
        )


class NeverCalledModel:
    """Fails the test if generation starts."""

    name = "fake:never-called"
    structured_output: Literal["tool", "json", "extractive"] = "tool"

    async def complete(self, request: ModelRequest) -> ModelReply:
        raise AssertionError("the model must not be called past a refusal gate")


async def test_low_score_refuses_before_generation() -> None:
    model = NeverCalledModel()
    response = await answer_question(
        "What is the loss function?",
        [_chunk(7, score=-2.0)],
        model=model,
        calibration=_calibration(a=1.0, b=0.0, threshold=0.9),
    )

    assert response.insufficient_evidence is True
    assert response.reason == "low_score"
    assert response.claims == ()
    assert response.refusal_probability is not None
    assert response.refusal_probability < 0.9
    assert [passage.chunk_id for passage in response.passages] == [7]
    assert response.calibration.threshold == 0.9


def test_probability_comes_from_the_file_coefficients() -> None:
    calibration = _calibration(a=2.0, b=-3.0, threshold=0.4)
    expected = 1.0 / (1.0 + math.exp(1.0))
    assert calibration.probability(1.0) == pytest.approx(expected)
    assert calibration.probability(3.0) > calibration.probability(1.0)


def test_missing_calibration_refuses_to_load(tmp_path: Path) -> None:
    missing = tmp_path / "calibration.json"
    with pytest.raises(CalibrationUnavailable, match="--uncalibrated"):
        load_calibration(missing)


async def test_uncalibrated_flag_is_allowed_and_travels_into_the_response() -> None:
    missing = Path("does-not-exist") / "calibration.json"
    calibration = load_calibration(missing, allow_uncalibrated=True)
    assert calibration.uncalibrated is True
    assert calibration.threshold is None

    response_payload: dict[str, object] = {
        "claims": [],
        "insufficient_evidence": True,
        "reason": "no answer in the passages",
    }
    response = await answer_question(
        "Is the moon made of cheese?",
        [_chunk(7)],
        model=FixedModel(response_payload),
        calibration=calibration,
    )
    assert response.uncalibrated is True
    assert response.calibration.uncalibrated is True
    assert response.calibration.fit_set_sha256 is None


@pytest.mark.parametrize(
    "document",
    [
        {"a": 1.0, "b": 0.0, "fit_set_sha256": "a" * 64},
        {"a": 0.0, "b": 0.0, "threshold": 0.5, "fit_set_sha256": "a" * 64},
        {"a": 1.0, "b": 0.0, "threshold": 1.5, "fit_set_sha256": "a" * 64},
        {"a": 1.0, "b": 0.0, "threshold": 0.5, "fit_set_sha256": "not-a-hash"},
        {"a": float("nan"), "b": 0.0, "threshold": 0.5, "fit_set_sha256": "a" * 64},
    ],
)
def test_malformed_calibration_is_fatal(tmp_path: Path, document: dict[str, object]) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(CalibrationError):
        load_calibration(path)


def test_committed_calibration_matches_its_fit_set() -> None:
    calibration = load_calibration(PROJECT_ROOT / "eval" / "calibration.json")
    assert calibration.calibrated is True
    assert calibration.threshold is not None
    assert 0.0 <= calibration.threshold <= 1.0
    assert calibration.a is not None and calibration.a > 0
    assert calibration.fit_set_sha256 == calibration_fit_hash(
        PROJECT_ROOT / "eval" / "calibration_fit_set.json"
    )


async def test_unanswerable_question_returns_reason_and_two_passages() -> None:
    payload: dict[str, object] = {
        "claims": [],
        "insufficient_evidence": True,
        "reason": "the passages discuss architecture, not the loss",
    }
    response = await answer_question(
        "What learning rate schedule was used?",
        [_chunk(7), _chunk(8, text=SECOND_TEXT, final_rank=2)],
        model=FixedModel(payload),
        calibration=_calibration(),
    )

    assert response.insufficient_evidence is True
    assert response.reason == "insufficient_context"
    assert response.model_reason == "the passages discuss architecture, not the loss"
    assert [passage.chunk_id for passage in response.passages] == [7, 8]
    assert response.claims == ()


async def test_no_retrieved_chunks_is_a_refusal() -> None:
    response = await answer_question(
        "anything",
        [],
        model=NeverCalledModel(),
        calibration=_calibration(),
    )
    assert response.insufficient_evidence is True
    assert response.reason == "low_score"
    assert response.passages == ()


def test_create_app_refuses_to_boot_without_calibration(tmp_path: Path) -> None:
    with pytest.raises(CalibrationUnavailable):
        create_app(
            calibration_path=tmp_path / "missing.json",
            model=NeverCalledModel(),
            retriever=lambda question: _empty(),
        )


async def _empty() -> list[RetrievedChunk]:
    return []


def test_create_app_boots_uncalibrated_and_health_records_it(tmp_path: Path) -> None:
    app = create_app(
        calibration_path=tmp_path / "missing.json",
        allow_uncalibrated=True,
        model=NeverCalledModel(),
        retriever=lambda question: _empty(),
    )
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["uncalibrated"] is True
    assert body["calibration"]["threshold"] is None


def test_post_ask_low_score_returns_a_refusal_without_calling_the_model() -> None:
    async def retriever(question: str) -> list[RetrievedChunk]:
        return [_chunk(7, score=-2.0)]

    app = create_app(
        calibration=_calibration(threshold=0.9),
        model=NeverCalledModel(),
        retriever=retriever,
    )
    with TestClient(app) as client:
        response = client.post("/ask", json={"question": "What is the loss?"})

    assert response.status_code == 200
    body = response.json()
    assert body["insufficient_evidence"] is True
    assert body["reason"] == "low_score"
    assert body["passages"][0]["chunk_id"] == 7
    assert body["uncalibrated"] is False


async def test_tool_mode_puts_the_forced_tool_choice_on_the_wire() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "submit_answer",
                                        "arguments": (
                                            '{"claims": [], '
                                            '"insufficient_evidence": true}'
                                        ),
                                    }
                                }
                            ]
                        }
                    }
                ]
            },
        )

    model = OpenAICompatibleModel(
        "http://stub.invalid/v1",
        "stub-model",
        structured_output="tool",
        transport=httpx.MockTransport(handler),
    )
    reply = await model.complete(
        ModelRequest(
            question="q",
            system_prompt="s",
            user_prompt="u",
            chunks=(),
        )
    )

    assert captured["tool_choice"] == {
        "type": "function",
        "function": {"name": "submit_answer"},
    }
    assert isinstance(captured["tools"], list)
    assert reply.via_tool_call is True
    assert reply.payload == {"claims": [], "insufficient_evidence": True}


async def test_json_mode_omits_tools_and_repairs_the_content() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '```json\n{"claims": [], '
                                '"insufficient_evidence": true}\n```'
                            )
                        }
                    }
                ]
            },
        )

    model = OpenAICompatibleModel(
        "http://stub.invalid/v1",
        "stub-model",
        structured_output="json",
        transport=httpx.MockTransport(handler),
    )
    reply = await model.complete(
        ModelRequest(
            question="q",
            system_prompt="s",
            user_prompt="u",
            chunks=(),
        )
    )

    assert "tools" not in captured
    assert "tool_choice" not in captured
    assert reply.via_tool_call is False
    assert reply.payload == {"claims": [], "insufficient_evidence": True}


async def test_probe_makes_a_real_loopback_request() -> None:
    verdict = await probe_local_client()
    assert verdict.made_request is True
    assert verdict.tool_choice_sent is True
    assert verdict.tool_call_returned is True
    assert verdict.honoured is True
    assert verdict.endpoint == "local loopback stub"
    assert "unverified" in verdict.detail

"""Citation integrity and the /ask contract, proven with forged model responses.

No network and no database: every model reply here is fabricated by the test.
The server must drop a claim before it can reach a caller unless its citation id
exists in the retrieved context and its quote is a normalised substring of every
cited chunk.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Literal

import pytest
from fastapi.testclient import TestClient

from docqa.answer import (
    Calibration,
    ExtractiveModel,
    ModelReply,
    ModelRequest,
    StructuredModel,
    answer_question,
    normalise_text,
    parse_json_lenient,
)
from docqa.api import create_app
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
        breadcrumb="3 Model Architecture > 3.1 Encoder",
        text=text,
        vector_rank=1,
        lexical_rank=2,
        rrf_score=0.031,
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


class FakeModel:
    """Returns queued payloads and records every request it receives."""

    name = "fake:forged-responses"
    structured_output: Literal["tool", "json", "extractive"] = "tool"

    def __init__(self, payloads: Sequence[object]) -> None:
        self._payloads = list(payloads)
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelReply:
        self.requests.append(request)
        assert self._payloads, "model called more times than the test expected"
        payload = self._payloads.pop(0)
        return ModelReply(
            payload=payload,
            raw_text=json.dumps(payload),
            via_tool_call=True,
        )


def _claim(
    text: str,
    citation_ids: Sequence[object],
    quote: str,
) -> dict[str, object]:
    return {"text": text, "citation_ids": list(citation_ids), "quote": quote}


async def test_valid_answer_keeps_the_quote_and_full_provenance() -> None:
    model = FakeModel(
        [
            {
                "claims": [_claim("The encoder is continuous.", ["7"], QUOTE)],
                "insufficient_evidence": False,
            }
        ]
    )
    response = await answer_question(
        "How are tokens represented?", [_chunk(7)], model=model, calibration=_calibration()
    )

    assert response.insufficient_evidence is False
    assert response.reason is None
    assert len(response.claims) == 1
    citation = response.claims[0].citations[0]
    assert citation.chunk_id == 7
    assert citation.doc_id == "attention-is-all-you-need"
    assert citation.page_no == 3
    assert citation.page_text_sha256 == "a" * 64
    assert citation.quote == QUOTE
    assert response.validation.claims_kept == 1
    assert response.model == "fake:forged-responses"


async def test_uncited_claim_is_dropped_and_the_drop_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = FakeModel(
        [
            {
                "claims": [
                    _claim("Supported claim.", ["7"], QUOTE),
                    _claim("Unsupported claim.", [], QUOTE),
                ],
                "insufficient_evidence": False,
            }
        ]
    )
    with caplog.at_level(logging.WARNING, logger="docqa.answer"):
        response = await answer_question(
            "How are tokens represented?",
            [_chunk(7)],
            model=model,
            calibration=_calibration(),
        )

    assert [claim.text for claim in response.claims] == ["Supported claim."]
    assert response.validation.dropped_uncited == 1
    assert response.validation.claims_dropped == 1
    messages = [record.getMessage() for record in caplog.records]
    assert any("reason=uncited" in message for message in messages)


async def test_citation_id_absent_from_context_drops_the_claim() -> None:
    model = FakeModel(
        [
            {
                "claims": [
                    _claim("Cites a chunk that was never retrieved.", ["999"], QUOTE)
                ],
                "insufficient_evidence": False,
            }
        ]
    )
    response = await answer_question(
        "How are tokens represented?",
        [_chunk(7)],
        model=model,
        calibration=_calibration(),
    )

    assert response.claims == ()
    assert response.validation.dropped_unknown_citation == 1
    assert response.insufficient_evidence is True
    assert response.reason == "validation_failed"
    assert len(response.passages) == 1


async def test_forged_quote_not_in_the_cited_chunk_is_dropped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    forged = "Transformers use convolutional layers exclusively."
    model = FakeModel(
        [
            {
                "claims": [_claim("A plausible but unquoted claim.", ["7"], forged)],
                "insufficient_evidence": False,
            }
        ]
    )
    with caplog.at_level(logging.WARNING, logger="docqa.answer"):
        response = await answer_question(
            "How are tokens represented?",
            [_chunk(7)],
            model=model,
            calibration=_calibration(),
        )

    assert response.claims == ()
    assert response.validation.dropped_quote_mismatch == 1
    assert response.insufficient_evidence is True
    assert any(
        "reason=quote_mismatch" in record.getMessage() for record in caplog.records
    )


def test_quote_matching_normalises_space_and_typography_but_not_words() -> None:
    chunk_text = "The encoder   maps\ntokens to continuous representations."
    normalised_chunk = normalise_text(chunk_text)

    assert normalise_text("  the ENCODER maps tokens ") in normalised_chunk
    assert normalise_text("the encoder maps tokens") in normalised_chunk
    assert normalise_text("the encoder maps words") not in normalised_chunk


async def test_quote_that_is_not_verbatim_in_every_cited_chunk_is_dropped() -> None:
    model = FakeModel(
        [
            {
                "claims": [_claim("Cites two chunks.", ["7", "8"], QUOTE)],
                "insufficient_evidence": False,
            }
        ]
    )
    response = await answer_question(
        "How are tokens represented?",
        [_chunk(7), _chunk(8, text=SECOND_TEXT, final_rank=2)],
        model=model,
        calibration=_calibration(),
    )

    assert response.claims == ()
    assert response.validation.dropped_quote_mismatch == 1


async def test_overlong_quote_is_dropped() -> None:
    long_quote = QUOTE + " " + "x" * 240
    model = FakeModel(
        [
            {
                "claims": [_claim("Too long.", ["7"], long_quote)],
                "insufficient_evidence": False,
            }
        ]
    )
    response = await answer_question(
        "How are tokens represented?",
        [_chunk(7)],
        model=model,
        calibration=_calibration(),
    )
    assert response.claims == ()
    assert response.validation.dropped_bad_quote == 1


async def test_schema_failure_gets_exactly_one_repair_turn() -> None:
    model = FakeModel(
        [
            {"claims": "not-an-array", "insufficient_evidence": False},
            {
                "claims": [_claim("Recovered claim.", ["7"], QUOTE)],
                "insufficient_evidence": False,
            },
        ]
    )
    response = await answer_question(
        "How are tokens represented?",
        [_chunk(7)],
        model=model,
        calibration=_calibration(),
    )

    assert len(model.requests) == 2
    assert model.requests[1].repair_errors
    assert any("claims must be an array" in error for error in model.requests[1].repair_errors)
    assert response.validation.repair_turns == 1
    assert response.validation.schema_failures == 1
    assert len(response.claims) == 1


async def test_second_schema_failure_refuses_with_closest_two_passages() -> None:
    bad_payload: dict[str, object] = {"claims": [], "insufficient_evidence": "no"}
    model = FakeModel([bad_payload, dict(bad_payload)])
    response = await answer_question(
        "How are tokens represented?",
        [_chunk(7), _chunk(8, text=SECOND_TEXT, final_rank=2)],
        model=model,
        calibration=_calibration(),
    )

    assert len(model.requests) == 2
    assert response.insufficient_evidence is True
    assert response.reason == "validation_failed"
    assert response.validation.schema_failures == 2
    assert response.validation.repair_turns == 1
    assert [passage.chunk_id for passage in response.passages] == [7, 8]


async def test_claim_dropped_to_zero_refuses_instead_of_showing_an_answer() -> None:
    model = FakeModel(
        [
            {
                "claims": [
                    _claim("Forged one.", ["7"], "not present anywhere"),
                    _claim("Forged two.", ["999"], QUOTE),
                ],
                "insufficient_evidence": False,
            }
        ]
    )
    response = await answer_question(
        "How are tokens represented?",
        [_chunk(7), _chunk(8, text=SECOND_TEXT, final_rank=2)],
        model=model,
        calibration=_calibration(),
    )

    assert response.insufficient_evidence is True
    assert response.reason == "validation_failed"
    assert response.validation.claims_returned == 2
    assert response.validation.claims_kept == 0
    assert len(response.passages) == 2


async def test_extractive_fallback_still_passes_citation_validation() -> None:
    response = await answer_question(
        "How does the encoder map tokens?",
        [_chunk(7)],
        model=ExtractiveModel(),
        calibration=_calibration(),
    )

    assert response.model == "extractive fallback (no model key)"
    assert response.structured_output == "extractive"
    assert response.insufficient_evidence is False
    assert response.claims
    for claim in response.claims:
        assert claim.citations
        for citation in claim.citations:
            assert citation.quote in CHUNK_TEXT
            assert "encoder" in citation.quote.lower()


def test_lenient_json_parser_repairs_fenced_output() -> None:
    fenced = '```json\n{"claims": [], "insufficient_evidence": true,}\n```'
    parsed = parse_json_lenient(fenced)
    assert isinstance(parsed, dict)
    assert parsed["insufficient_evidence"] is True


def test_post_ask_returns_a_validated_answer() -> None:
    model = FakeModel(
        [
            {
                "claims": [_claim("The encoder is continuous.", ["7"], QUOTE)],
                "insufficient_evidence": False,
            }
        ]
    )

    async def retriever(question: str) -> list[RetrievedChunk]:
        assert question == "How are tokens represented?"
        return [_chunk(7)]

    app = create_app(
        calibration=_calibration(), model=model, retriever=retriever
    )
    with TestClient(app) as client:
        response = client.post(
            "/ask", json={"question": "How are tokens represented?"}
        )

    assert response.status_code == 200
    body = response.json()
    assert body["insufficient_evidence"] is False
    assert body["claims"][0]["text"] == "The encoder is continuous."
    citation = body["claims"][0]["citations"][0]
    assert citation["chunk_id"] == 7
    assert citation["quote"] == QUOTE
    assert citation["page_text_sha256"] == "a" * 64
    assert body["validation"]["claims_kept"] == 1


def test_post_ask_hides_an_uncited_claim_from_the_caller() -> None:
    model = FakeModel(
        [
            {
                "claims": [
                    _claim("Grounded.", ["7"], QUOTE),
                    _claim("Not grounded.", ["7"], "a sentence the paper never wrote"),
                ],
                "insufficient_evidence": False,
            }
        ]
    )

    async def retriever(question: str) -> list[RetrievedChunk]:
        return [_chunk(7)]

    app = create_app(
        calibration=_calibration(), model=model, retriever=retriever
    )
    with TestClient(app) as client:
        response = client.post("/ask", json={"question": "question"})

    assert response.status_code == 200
    body = response.json()
    assert [claim["text"] for claim in body["claims"]] == ["Grounded."]
    assert body["validation"]["dropped_quote_mismatch"] == 1


def test_structured_model_protocol_is_satisfied_by_the_fake() -> None:
    model: StructuredModel = FakeModel([{"claims": [], "insufficient_evidence": True}])
    assert model.structured_output == "tool"

"""FastAPI surface for validated answers.

``POST /ask`` accepts a question, runs hybrid retrieval, and returns the
:class:`docqa.answer.AnswerResponse` as JSON. Generation is non-streaming: the
handler only emits claims the server has already checked against the context.

The calibration file is loaded when the app is created. Without it,
``create_app`` raises unless ``allow_uncalibrated=True`` is passed; the resulting
responses carry ``"uncalibrated": true`` in their metadata.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import cast

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from docqa.answer import (
    AnswerResponse,
    Calibration,
    ModelCallError,
    StructuredModel,
    answer_question,
    build_model,
    load_calibration,
)
from docqa.config import PROJECT_ROOT, Settings, get_settings
from docqa.db import make_engine, make_session_factory, session_scope
from docqa.ingest.embedder import build_embedder
from docqa.rerank import build_reranker
from docqa.retrieval import (
    PgCandidateSource,
    RerankTraceHit,
    RetrievalResult,
    RetrievalTrace,
    RetrievedChunk,
    retrieve_with_trace,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_K = 8

Retriever = Callable[[str], Awaitable[list[RetrievedChunk]]]
TraceRetriever = Callable[[str], Awaitable[RetrievalResult]]


class AskPayload(BaseModel):
    """Request body for ``POST /ask``."""

    question: str = Field(min_length=1, max_length=2000)


class PgRetriever:
    """Production retriever: hybrid retrieval over the configured database."""

    def __init__(self, settings: Settings) -> None:
        """Build the engine, embedder, and reranker; does not connect yet."""
        self._engine = make_engine(settings)
        self._factory = make_session_factory(self._engine)
        embedder_choice = build_embedder(settings)
        self._embedder = embedder_choice.embedder
        LOGGER.info("%s", embedder_choice.banner)
        rerank_mode = os.environ.get("DOCQA_RERANKER", "auto").strip().lower()
        rerank_choice = build_reranker(mode=rerank_mode)
        self._reranker = rerank_choice.reranker
        LOGGER.info("%s", rerank_choice.banner)

    async def retrieve_result(self, question: str) -> RetrievalResult:
        """Run the R2 pipeline and return the context plus its trace."""
        async with session_scope(self._factory) as session:
            return await retrieve_with_trace(
                question,
                DEFAULT_K,
                embedder=self._embedder,
                source=PgCandidateSource(session),
                reranker=self._reranker,
            )

    async def __call__(self, question: str) -> list[RetrievedChunk]:
        """Run the R2 pipeline and return the final context."""
        result = await self.retrieve_result(question)
        return list(result.chunks)


@dataclasses.dataclass
class AnswerService:
    """The collaborators ``/ask`` needs, bundled for injection in tests."""

    retriever: Retriever
    model: StructuredModel
    calibration: Calibration
    trace_retriever: TraceRetriever | None = None

    async def answer(self, question: str) -> AnswerResponse:
        """Retrieve, generate, validate, and return the response."""
        response, _ = await self.answer_with_trace(question)
        return response

    async def answer_with_trace(
        self, question: str
    ) -> tuple[AnswerResponse, RetrievalTrace | None]:
        """Answer and return the retrieval trace that produced the context.

        A retriever injected by a test has no trace; the response then carries
        ``trace: null`` rather than inventing stage numbers.
        """
        if self.trace_retriever is not None:
            result = await self.trace_retriever(question)
            chunks = list(result.chunks)
            trace: RetrievalTrace | None = result.trace
        else:
            chunks = await self.retriever(question)
            trace = None
        response = await answer_question(
            question, chunks, model=self.model, calibration=self.calibration
        )
        return response, trace


def _trace_payload(
    trace: RetrievalTrace, response: AnswerResponse, calibration: Calibration
) -> dict[str, object]:
    """Serialise the trace and attach the answer-level refusal numbers."""
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
            "decision": (
                "refused" if response.insufficient_evidence else "answered"
            ),
            "reason": response.reason,
            "uncalibrated": calibration.uncalibrated,
        },
    }


def _calibrated(calibration: Calibration, hit: RerankTraceHit) -> float | None:
    """Apply the Platt scaling to one reranked candidate."""
    return calibration.probability(hit.rerank_score)


def create_app(
    *,
    settings: Settings | None = None,
    calibration: Calibration | None = None,
    calibration_path: Path | None = None,
    allow_uncalibrated: bool = False,
    model: StructuredModel | None = None,
    retriever: Retriever | None = None,
    trace_retriever: TraceRetriever | None = None,
) -> FastAPI:
    """Create the application, refusing to boot without a usable calibration.

    Tests inject ``calibration``, ``model``, ``retriever``, and optionally a
    ``trace_retriever``; production builds the model immediately and the
    database retriever in the lifespan hook. A plain injected retriever yields
    ``trace: null`` so the UI never shows invented stage numbers.
    """
    resolved = settings if settings is not None else get_settings()
    chosen_calibration = (
        calibration
        if calibration is not None
        else load_calibration(
            calibration_path or resolved.calibration_path,
            allow_uncalibrated=allow_uncalibrated,
        )
    )
    chosen_model = model if model is not None else build_model(resolved)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        if application.state.retriever is None:
            production = PgRetriever(resolved)
            application.state.retriever = production
            application.state.trace_retriever = production.retrieve_result
        yield

    application = FastAPI(
        title="docqa-assistant",
        version="0.1.0",
        description="Cited answers over a pinned PDF corpus.",
        lifespan=lifespan,
    )
    application.state.settings = resolved
    application.state.calibration = chosen_calibration
    application.state.model = chosen_model
    application.state.retriever = retriever
    application.state.trace_retriever = trace_retriever
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @application.get("/health")
    async def health(request: Request) -> dict[str, object]:
        """Report the model, structured-output mode, and calibration provenance."""
        app_model = cast(StructuredModel, request.app.state.model)
        app_calibration = cast(Calibration, request.app.state.calibration)
        return {
            "status": "ok",
            "model": app_model.name,
            "model_is_fallback": app_model.structured_output == "extractive",
            "structured_output": app_model.structured_output,
            "uncalibrated": app_calibration.uncalibrated,
            "calibration": dataclasses.asdict(app_calibration.info()),
        }

    @application.post("/ask")
    async def ask(payload: AskPayload, request: Request) -> dict[str, object]:
        """Answer one question; every returned claim is server-validated."""
        app_retriever = cast(Retriever | None, request.app.state.retriever)
        if app_retriever is None:
            raise HTTPException(status_code=503, detail="retriever is not ready")
        app_model = cast(StructuredModel, request.app.state.model)
        app_calibration = cast(Calibration, request.app.state.calibration)
        app_trace_retriever = cast(
            TraceRetriever | None, request.app.state.trace_retriever
        )
        try:
            response, trace = await AnswerService(
                retriever=app_retriever,
                model=app_model,
                calibration=app_calibration,
                trace_retriever=app_trace_retriever,
            ).answer_with_trace(payload.question)
        except ModelCallError as exc:
            LOGGER.error("model call failed: %s", exc)
            raise HTTPException(
                status_code=502, detail=f"model endpoint failed: {exc}"
            ) from exc
        body = dataclasses.asdict(response)
        body["trace"] = (
            None
            if trace is None
            else _trace_payload(trace, response, app_calibration)
        )
        # The server cannot see the browser's text layer. The slot is part of
        # the contract so the UI can show the client-measured count in the same
        # place for every answer; the client value is authoritative.
        body["citation_render_miss"] = 0
        return body

    @application.get("/eval/report")
    async def eval_report() -> JSONResponse:
        """Serve the committed ``eval-report.json`` or an explicit 404."""
        path = PROJECT_ROOT / "eval-report.json"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="no eval run yet")
        try:
            report: object = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise HTTPException(
                status_code=500, detail=f"eval report is unreadable: {exc}"
            ) from exc
        return JSONResponse(report)

    @application.get("/eval-report.json")
    async def eval_report_download() -> FileResponse:
        """Download the committed eval artifact, or 404 when it does not exist."""
        path = PROJECT_ROOT / "eval-report.json"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="no eval run yet")
        return FileResponse(path, media_type="application/json")

    return application

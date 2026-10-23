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
import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import cast

from fastapi import FastAPI, HTTPException, Request
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
from docqa.config import Settings, get_settings
from docqa.db import make_engine, make_session_factory, session_scope
from docqa.ingest.embedder import build_embedder
from docqa.rerank import build_reranker
from docqa.retrieval import PgCandidateSource, RetrievedChunk, retrieve

LOGGER = logging.getLogger(__name__)

DEFAULT_K = 8

Retriever = Callable[[str], Awaitable[list[RetrievedChunk]]]


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

    async def __call__(self, question: str) -> list[RetrievedChunk]:
        """Run the R2 pipeline and return the final context."""
        async with session_scope(self._factory) as session:
            return await retrieve(
                question,
                DEFAULT_K,
                embedder=self._embedder,
                source=PgCandidateSource(session),
                reranker=self._reranker,
            )


@dataclasses.dataclass
class AnswerService:
    """The three collaborators ``/ask`` needs, bundled for injection in tests."""

    retriever: Retriever
    model: StructuredModel
    calibration: Calibration

    async def answer(self, question: str) -> AnswerResponse:
        """Retrieve, generate, validate, and return the response."""
        chunks = await self.retriever(question)
        return await answer_question(
            question, chunks, model=self.model, calibration=self.calibration
        )


def create_app(
    *,
    settings: Settings | None = None,
    calibration: Calibration | None = None,
    calibration_path: Path | None = None,
    allow_uncalibrated: bool = False,
    model: StructuredModel | None = None,
    retriever: Retriever | None = None,
) -> FastAPI:
    """Create the application, refusing to boot without a usable calibration.

    Tests inject ``calibration``, ``model``, and ``retriever``; production builds
    the model immediately and the database retriever in the lifespan hook.
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
            application.state.retriever = PgRetriever(resolved)
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
        try:
            response = await AnswerService(
                retriever=app_retriever,
                model=app_model,
                calibration=app_calibration,
            ).answer(payload.question)
        except ModelCallError as exc:
            LOGGER.error("model call failed: %s", exc)
            raise HTTPException(
                status_code=502, detail=f"model endpoint failed: {exc}"
            ) from exc
        return dataclasses.asdict(response)

    return application

"""FastAPI surface for validated answers.

``POST /ask`` accepts a question, runs hybrid retrieval, and returns the
:class:`docqa.answer.AnswerResponse` as JSON. Generation is non-streaming: the
handler only emits claims the server has already checked against the context.

The calibration file is loaded when the app is created. Without it,
``create_app`` raises unless ``allow_uncalibrated=True`` is passed; the resulting
responses carry ``"uncalibrated": true`` in their metadata.
"""

from __future__ import annotations

import asyncio
import copy
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
from sqlalchemy import text

from docqa.answer import (
    AnswerResponse,
    Calibration,
    ModelCallError,
    RateLimitedError,
    StructuredModel,
    answer_question,
    build_model,
    load_calibration,
)
from docqa.cache import AnswerCache, trace_payload
from docqa.config import PROJECT_ROOT, Settings, get_settings
from docqa.db import (
    alembic_config,
    make_engine,
    make_session_factory,
    session_scope,
)
from docqa.ingest.embedder import build_embedder
from docqa.rerank import build_reranker
from docqa.retrieval import (
    PgCandidateSource,
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


def corpus_document_count(lock_path: Path) -> int | None:
    """Count the PDFs pinned in ``corpus.lock.json``; ``None`` if unreadable."""
    try:
        raw: object = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if isinstance(raw, dict):
        files = cast(dict[str, object], raw).get("files")
        if isinstance(files, list):
            return len(cast(list[object], files))
    return None


def migration_head() -> str | None:
    """Return the revision at the tip of the Alembic history, if available."""
    try:
        from alembic.script import ScriptDirectory

        config = alembic_config("postgresql+asyncpg://unused/unused")
        return ScriptDirectory.from_config(config).get_current_head()
    except Exception:
        return None


async def readiness_payload(
    settings: Settings, expected_documents: int | None, head: str | None
) -> tuple[dict[str, object], int]:
    """Read the live database's corpus counts and applied migration.

    The free Render plan runs no pre-deploy command, so a service can boot
    against a database that has the schema but not the corpus. This route makes
    that state a failed readiness probe instead of an empty answer.
    """
    engine = make_engine(settings)
    try:
        async with asyncio.timeout(15):
            async with engine.connect() as connection:
                documents = int(
                    (
                        await connection.execute(text("select count(*) from documents"))
                    ).scalar_one()
                )
                chunks = int(
                    (
                        await connection.execute(text("select count(*) from chunks"))
                    ).scalar_one()
                )
                revision_row = (
                    await connection.execute(text("select version_num from alembic_version"))
                ).scalar_one_or_none()
                revision = None if revision_row is None else str(revision_row)
    except Exception as exc:
        return {
            "status": "unavailable",
            "database": {"status": "error", "error": type(exc).__name__},
            "migration": {"current": None, "head": head, "ok": False},
            "corpus": {
                "documents": None,
                "chunks": None,
                "expected_documents": expected_documents,
                "ok": False,
            },
        }, 503
    finally:
        await engine.dispose()

    migration_ok = head is not None and revision == head
    corpus_ok = (
        expected_documents is not None
        and documents == expected_documents
        and chunks > 0
    )
    ready = migration_ok and corpus_ok
    return {
        "status": "ok" if ready else "degraded",
        "database": {"status": "ok", "documents": documents, "chunks": chunks},
        "migration": {"current": revision, "head": head, "ok": migration_ok},
        "corpus": {
            "documents": documents,
            "chunks": chunks,
            "expected_documents": expected_documents,
            "ok": corpus_ok,
        },
    }, (200 if ready else 503)


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
        rerank_choice = build_reranker(
            mode=rerank_mode,
            cache_dir=str(settings.hf_cache_dir) if settings.hf_cache_dir else None,
            threads=settings.onnx_threads,
        )
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


def create_app(
    *,
    settings: Settings | None = None,
    calibration: Calibration | None = None,
    calibration_path: Path | None = None,
    allow_uncalibrated: bool = False,
    model: StructuredModel | None = None,
    retriever: Retriever | None = None,
    trace_retriever: TraceRetriever | None = None,
    answer_cache: AnswerCache | None = None,
    cache_path: Path | None = None,
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
    chosen_cache = (
        answer_cache
        if answer_cache is not None
        else AnswerCache.load(cache_path or resolved.demo_answers_path)
    )
    expected_documents = corpus_document_count(resolved.corpus_lock)
    head_revision = migration_head()

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
    application.state.answer_cache = chosen_cache
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved.cors_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @application.get("/health")
    async def health(request: Request) -> dict[str, object]:
        """Report the model, structured-output mode, and calibration provenance."""
        app_model = cast(StructuredModel, request.app.state.model)
        app_calibration = cast(Calibration, request.app.state.calibration)
        app_cache = cast(AnswerCache, request.app.state.answer_cache)
        return {
            "status": "ok",
            "model": app_model.name,
            "model_is_fallback": app_model.structured_output == "extractive",
            "structured_output": app_model.structured_output,
            "structured_output_source": resolved.structured_output_source,
            "uncalibrated": app_calibration.uncalibrated,
            "calibration": dataclasses.asdict(app_calibration.info()),
            "demo_cache": {
                "path": str(app_cache.path),
                "entries": len(app_cache),
                "model": app_cache.model,
                "generated_at": app_cache.generated_at,
            },
            "ready": "/ready",
        }

    @application.get("/ready")
    async def ready() -> JSONResponse:
        """Report database reachability, corpus counts, and the applied migration."""
        payload, status_code = await readiness_payload(
            resolved, expected_documents, head_revision
        )
        return JSONResponse(payload, status_code=status_code)

    @application.post("/ask", response_model=None)
    async def ask(
        payload: AskPayload, request: Request
    ) -> dict[str, object] | JSONResponse:
        """Serve a recorded demo answer, or call the live model on a miss."""
        app_cache = cast(AnswerCache, request.app.state.answer_cache)
        cached = app_cache.get(payload.question)
        if cached is not None:
            body = copy.deepcopy(cached.response)
            body["cached"] = True
            body["cache"] = {
                "model": cached.model,
                "generated_at": cached.generated_at,
            }
            body["trace"] = copy.deepcopy(cached.trace)
            body["citation_render_miss"] = 0
            return body

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
        except RateLimitedError as exc:
            LOGGER.warning("live endpoint rate-limited: %s", exc)
            headers = (
                {"Retry-After": f"{exc.retry_after_s:g}"}
                if exc.retry_after_s is not None
                else None
            )
            return JSONResponse(
                status_code=429,
                headers=headers,
                content={
                    "state": "rate_limited",
                    "cached": False,
                    "detail": str(exc),
                    "retry_after_s": exc.retry_after_s,
                    "hint": (
                        "the free endpoint answers once per 30 s; this question "
                        "is not in data/demo-answers.json"
                    ),
                },
            )
        except ModelCallError as exc:
            LOGGER.error("model call failed: %s", exc)
            return JSONResponse(
                status_code=502,
                content={
                    "state": "model_unavailable",
                    "cached": False,
                    "detail": f"model endpoint failed: {exc}",
                },
            )
        body = dataclasses.asdict(response)
        body["cached"] = False
        body["trace"] = (
            None
            if trace is None
            else trace_payload(trace, response, app_calibration)
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

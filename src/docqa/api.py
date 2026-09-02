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
import secrets
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal, cast

import anyio
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

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
from docqa.config import DEMO_COLLECTION, PROJECT_ROOT, Settings, get_settings
from docqa.db import (
    alembic_config,
    make_engine,
    make_session_factory,
    session_scope,
)
from docqa.ingest.embedder import Embedder, build_embedder
from docqa.rerank import Reranker, RerankerChoice, build_reranker
from docqa.retrieval import (
    PgCandidateSource,
    RetrievalResult,
    RetrievalTrace,
    RetrievedChunk,
    retrieve_with_trace,
)
from docqa.uploads.limits import UploadLimitError
from docqa.uploads.service import (
    UploadBusyError,
    UploadQuotaError,
    UploadService,
    UploadView,
)
from docqa.uploads.service import (
    UploadFile as UploadDownload,
)
from docqa.uploads.session import CLIENT_COOKIE, COOKIE_MAX_AGE_S, SessionSigner

LOGGER = logging.getLogger(__name__)

DEFAULT_K = 8

# Fixed for every response, including CORS preflights, 404s, and validation
# errors. The CSP is the JSON-API baseline: nothing may be loaded or framed.
SECURITY_HEADERS: tuple[tuple[str, str], ...] = (
    ("Strict-Transport-Security", "max-age=31536000; includeSubDomains"),
    ("X-Content-Type-Options", "nosniff"),
    ("X-Frame-Options", "DENY"),
    ("Referrer-Policy", "no-referrer"),
    (
        "Permissions-Policy",
        "accelerometer=(), camera=(), geolocation=(), gyroscope=(), "
        "magnetometer=(), microphone=(), payment=(), usb=()",
    ),
    ("Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'"),
)


class SecurityHeadersMiddleware:
    """Attach the fixed security headers to every HTTP response."""

    def __init__(self, app: ASGIApp) -> None:
        """Wrap the downstream ASGI application."""
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Add the headers at ``http.response.start`` and pass everything else through."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for name, value in SECURITY_HEADERS:
                    headers[name] = value
            await send(message)

        await self.app(scope, receive, send_with_headers)


class HeadAsGetMiddleware:
    """Answer ``HEAD`` like ``GET`` with an empty body.

    Rewriting the method lets the router decide whether the route supports
    ``GET``: a route that only supports another method still answers ``405``
    with its ``Allow`` header. The downstream body is consumed but never
    forwarded, and ``Content-Length`` keeps the value ``GET`` would report.
    """

    def __init__(self, app: ASGIApp) -> None:
        """Wrap the downstream ASGI application."""
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Dispatch a HEAD request as GET and suppress the body."""
        if scope["type"] != "http" or scope["method"] != "HEAD":
            await self.app(scope, receive, send)
            return
        rewritten = dict(scope)
        rewritten["method"] = "GET"

        async def send_without_body(message: Message) -> None:
            if message["type"] == "http.response.body":
                if not message.get("more_body", False):
                    await send(
                        {
                            "type": "http.response.body",
                            "body": b"",
                            "more_body": False,
                        }
                    )
                return
            await send(message)

        await self.app(rewritten, receive, send_without_body)


def validation_error_payload(exc: RequestValidationError) -> dict[str, object]:
    """Return field paths and reasons without echoing the submitted bytes.

    FastAPI's default 422 body includes each error's ``input`` (and sometimes
    ``ctx``), which reflects the whole request value back to the caller. The
    response here keeps only the location, the error type, and Pydantic's
    value-free message.
    """
    errors: list[dict[str, object]] = []
    for error in exc.errors():
        location = [str(part) for part in error.get("loc", ())]
        errors.append(
            {
                "path": ".".join(location),
                "type": str(error.get("type", "value_error")),
                "reason": str(error.get("msg", "invalid value")),
            }
        )
    return {"state": "invalid_request", "detail": errors}

Retriever = Callable[[str], Awaitable[list[RetrievedChunk]]]
TraceRetriever = Callable[[str], Awaitable[RetrievalResult]]
CollectionTraceRetriever = Callable[[str, str], Awaitable[RetrievalResult]]


class AskPayload(BaseModel):
    """Request body for ``POST /ask``."""

    question: str = Field(min_length=1, max_length=2000)
    collection: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$"
    )


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
        """Build the engine and embedder; load the reranker on first use.

        Both ONNX models stay resident for the life of the process. The
        embedder is needed for every live question; the cross-encoder adds
        ~62 MB and is not needed for a recorded demo answer, so it is loaded on
        the first retrieval instead of at boot, shortening the cold start.
        """
        self._engine = make_engine(settings)
        self._factory = make_session_factory(self._engine)
        self._settings = settings
        self._rerank_mode = os.environ.get("DOCQA_RERANKER", "auto").strip().lower()
        embedder_choice = build_embedder(settings)
        self._embedder = embedder_choice.embedder
        LOGGER.info("%s", embedder_choice.banner)
        self._reranker: Reranker | None = None
        self._reranker_lock = asyncio.Lock()

    def _load_reranker(self) -> RerankerChoice:
        """Build the configured reranker; runs in a worker thread."""
        return build_reranker(
            mode=self._rerank_mode,
            cache_dir=(
                str(self._settings.hf_cache_dir)
                if self._settings.hf_cache_dir
                else None
            ),
            threads=self._settings.onnx_threads,
            batch_size=self._settings.rerank_batch_size,
            memory_arena=self._settings.onnx_memory_arena,
        )

    async def _ensure_reranker(self) -> Reranker:
        """Load the reranker once, even under concurrent first requests."""
        if self._reranker is not None:
            return self._reranker
        async with self._reranker_lock:
            if self._reranker is None:
                choice = await anyio.to_thread.run_sync(self._load_reranker)
                LOGGER.info("%s", choice.banner)
                self._reranker = choice.reranker
        return self._reranker

    @property
    def embedder(self) -> Embedder:
        """The resident embedder, shared with the upload worker."""
        return self._embedder

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        """The process's session factory, shared with the upload service."""
        return self._factory

    async def retrieve_result(
        self, question: str, collection: str = DEMO_COLLECTION
    ) -> RetrievalResult:
        """Run the R2 pipeline over one collection and return context plus trace."""
        reranker = await self._ensure_reranker()
        async with session_scope(self._factory) as session:
            return await retrieve_with_trace(
                question,
                DEFAULT_K,
                embedder=self._embedder,
                source=PgCandidateSource(session, collection),
                reranker=reranker,
            )

    async def __call__(self, question: str) -> list[RetrievedChunk]:
        """Run the R2 pipeline over the demo corpus and return the context."""
        result = await self.retrieve_result(question)
        return list(result.chunks)


@dataclasses.dataclass
class AnswerService:
    """The collaborators ``/ask`` needs, bundled for injection in tests."""

    retriever: Retriever
    model: StructuredModel
    calibration: Calibration
    trace_retriever: TraceRetriever | None = None
    collection_trace_retriever: CollectionTraceRetriever | None = None

    async def answer(self, question: str) -> AnswerResponse:
        """Retrieve, generate, validate, and return the response."""
        response, _ = await self.answer_with_trace(question)
        return response

    async def answer_with_trace(
        self, question: str, collection: str | None = None
    ) -> tuple[AnswerResponse, RetrievalTrace | None]:
        """Answer and return the retrieval trace that produced the context.

        A retriever injected by a test has no trace; the response then carries
        ``trace: null`` rather than inventing stage numbers. A named
        ``collection`` must use the collection-aware retriever; falling back to
        the demo one would mix another corpus into the answer.
        """
        trace: RetrievalTrace | None
        if collection is not None:
            if self.collection_trace_retriever is None:
                raise RuntimeError("no collection-aware retriever is configured")
            result = await self.collection_trace_retriever(question, collection)
            chunks = list(result.chunks)
            trace = result.trace
        elif self.trace_retriever is not None:
            result = await self.trace_retriever(question)
            chunks = list(result.chunks)
            trace = result.trace
        else:
            chunks = await self.retriever(question)
            trace = None
        response = await answer_question(
            question, chunks, model=self.model, calibration=self.calibration
        )
        return response, trace



# -- upload helpers -------------------------------------------------------


def _upload_service(request: Request) -> UploadService:
    """Return the process's upload service or fail with 503."""
    service = cast(UploadService | None, request.app.state.upload_service)
    if service is None:
        raise HTTPException(status_code=503, detail="uploads are not available")
    return service


def _require_origin(request: Request) -> None:
    """Reject a cookie-bearing request from an origin the browser would not trust."""
    settings = cast(Settings, request.app.state.settings)
    origin = (request.headers.get("origin") or "").rstrip("/")
    if origin and origin not in settings.cors_origins:
        raise HTTPException(status_code=403, detail="origin not allowed")


def _session_signer(request: Request) -> SessionSigner:
    """Return the cookie signer bound to this app."""
    return cast(SessionSigner, request.app.state.upload_signer)


def _client_or_none(request: Request) -> str | None:
    """Return the signed client id, or ``None`` for a missing or forged cookie."""
    return _session_signer(request).verify(request.cookies.get(CLIENT_COOKIE))


def _require_client(request: Request) -> str:
    """Return the signed client id; absent or forged cookies are refused."""
    client_id = _client_or_none(request)
    if client_id is None:
        raise HTTPException(status_code=401, detail="start a session first")
    return client_id


def _parse_upload_id(raw: str) -> uuid.UUID:
    """Parse a path id, treating anything malformed as unknown."""
    try:
        return uuid.UUID(raw)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="unknown document") from exc


def _set_client_cookie(
    response: Response, request: Request, signer: SessionSigner, client_id: str
) -> None:
    """Set the signed cookie; a cross-origin HTTPS UI needs Secure + SameSite=None."""
    settings = cast(Settings, request.app.state.settings)
    secure = settings.cookie_secure or request.url.scheme == "https"
    same_site: Literal["none", "lax"] = "none" if secure else "lax"
    response.set_cookie(
        CLIENT_COOKIE,
        signer.token_for(client_id),
        max_age=COOKIE_MAX_AGE_S,
        httponly=True,
        secure=secure,
        samesite=same_site,
        path="/",
    )


def _limit_response(exc: UploadLimitError) -> JSONResponse:
    """Map a refused upload onto its documented status and code."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "state": "rejected",
            "code": exc.code,
            "detail": exc.detail,
            "limit": exc.limit,
        },
    )


def _upload_payload(view: UploadView) -> dict[str, object]:
    """Return the JSON shape shared by ``POST /documents`` and its status route."""
    return {
        "id": view.id,
        "status": view.status,
        "collection": view.collection,
        "filename": view.filename,
        "byte_size": view.byte_size,
        "pages_done": view.pages_done,
        "pages_total": view.pages_total,
        "error_code": view.error_code,
        "error_detail": view.error_detail,
        "created_at": view.created_at.isoformat(),
        "expires_at": view.expires_at.isoformat(),
    }


def create_app(
    *,
    settings: Settings | None = None,
    calibration: Calibration | None = None,
    calibration_path: Path | None = None,
    allow_uncalibrated: bool = False,
    model: StructuredModel | None = None,
    retriever: Retriever | None = None,
    trace_retriever: TraceRetriever | None = None,
    collection_retriever: CollectionTraceRetriever | None = None,
    answer_cache: AnswerCache | None = None,
    cache_path: Path | None = None,
    upload_service: UploadService | None = None,
    upload_signer: SessionSigner | None = None,
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
        started: UploadService | None = None
        if application.state.retriever is None:
            production = PgRetriever(resolved)
            application.state.retriever = production
            application.state.trace_retriever = production.retrieve_result
            application.state.collection_retriever = production.retrieve_result
            if application.state.upload_service is None:
                service = UploadService(
                    resolved, production.session_factory, production.embedder
                )
                application.state.upload_service = service
                await service.start()
                started = service
        try:
            yield
        finally:
            if started is not None:
                await started.stop()

    application = FastAPI(
        title="docqa-assistant",
        version="0.1.0",
        description="Cited answers over a pinned PDF corpus.",
        lifespan=lifespan,
        # The public API surface is the documented routes only; the interactive
        # schema pages are not part of it and are not advertised.
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    application.state.settings = resolved
    application.state.calibration = chosen_calibration
    application.state.model = chosen_model
    application.state.retriever = retriever
    application.state.trace_retriever = trace_retriever
    application.state.collection_retriever = collection_retriever
    application.state.answer_cache = chosen_cache
    application.state.upload_service = upload_service
    secret = resolved.cookie_secret or secrets.token_hex(32)
    if resolved.cookie_secret is None:
        LOGGER.warning(
            "DOCQA_COOKIE_SECRET is unset; using a per-process secret, so "
            "sessions do not survive a restart"
        )
    application.state.upload_signer = (
        upload_signer if upload_signer is not None else SessionSigner(secret)
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved.cors_origins),
        allow_methods=["GET", "POST", "HEAD"],
        allow_headers=["*"],
        allow_credentials=True,
    )
    application.add_middleware(SecurityHeadersMiddleware)
    application.add_middleware(HeadAsGetMiddleware)

    @application.exception_handler(RequestValidationError)
    async def invalid_request(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Answer 422 with field paths and reasons, never the submitted bytes."""
        return JSONResponse(status_code=422, content=validation_error_payload(exc))

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

    @application.get("/session")
    async def create_session(request: Request) -> JSONResponse:
        """Mint or keep an opaque client id, signed into an HttpOnly cookie."""
        _require_origin(request)
        signer = _session_signer(request)
        client_id, is_new = signer.session_for(request.cookies.get(CLIENT_COOKIE))
        response = JSONResponse({"client_id": client_id, "new": is_new})
        _set_client_cookie(response, request, signer, client_id)
        return response

    @application.post("/documents", status_code=202)
    async def create_document(
        request: Request, file: Annotated[UploadFile, File()]
    ) -> JSONResponse:
        """Store an upload's bytes and return before parsing starts."""
        _require_origin(request)
        service = _upload_service(request)
        client_id = _require_client(request)
        raw = await file.read(resolved.upload_max_bytes + 1)
        if len(raw) > resolved.upload_max_bytes:
            return _limit_response(
                UploadLimitError(
                    413,
                    "too_large",
                    f"The PDF is larger than {resolved.upload_max_bytes} bytes.",
                    limit=resolved.upload_max_bytes,
                )
            )
        try:
            view = await service.submit(
                client_id,
                file.filename or "upload.pdf",
                file.content_type or "application/pdf",
                raw,
            )
        except UploadLimitError as exc:
            return _limit_response(exc)
        except UploadBusyError as exc:
            return JSONResponse(
                status_code=409,
                content={
                    "state": "busy",
                    "id": exc.upload_id,
                    "status": exc.status,
                    "detail": "This browser already has an upload in progress.",
                },
            )
        except UploadQuotaError as exc:
            return JSONResponse(
                status_code=exc.status_code,
                headers={"Retry-After": str(exc.retry_after_s)},
                content={
                    "state": "rate_limited",
                    "code": exc.code,
                    "detail": exc.detail,
                    "retry_after_s": exc.retry_after_s,
                    "reset_at": exc.reset_at.isoformat(),
                },
            )
        return JSONResponse(status_code=202, content=_upload_payload(view))

    @application.get("/documents/{upload_id}")
    async def document_status(upload_id: str, request: Request) -> JSONResponse:
        """Report one upload's stage and page progress to its owner."""
        _require_origin(request)
        service = _upload_service(request)
        client_id = _require_client(request)
        view = await service.status(client_id, _parse_upload_id(upload_id))
        if view is None:
            raise HTTPException(status_code=404, detail="unknown document")
        return JSONResponse(_upload_payload(view))

    @application.get("/documents/{upload_id}/pages/{page_no}")
    async def document_page(
        upload_id: str, page_no: int, request: Request
    ) -> JSONResponse:
        """Serve one stored page's text to the owning client only."""
        _require_origin(request)
        service = _upload_service(request)
        client_id = _require_client(request)
        if page_no < 1:
            raise HTTPException(status_code=404, detail="unknown page")
        text = await service.page(client_id, _parse_upload_id(upload_id), page_no)
        if text is None:
            raise HTTPException(status_code=404, detail="unknown page")
        return JSONResponse({"page_no": page_no, "text": text})

    @application.get("/documents/{upload_id}/file")
    async def document_file(upload_id: str, request: Request) -> Response:
        """Serve the original PDF to the owning client only."""
        _require_origin(request)
        service = _upload_service(request)
        client_id = _require_client(request)
        download: UploadDownload | None = await service.file(
            client_id, _parse_upload_id(upload_id)
        )
        if download is None:
            raise HTTPException(status_code=404, detail="unknown document")
        safe = download.filename.replace('"', "").replace("\\", "")
        return Response(
            content=download.data,
            media_type="application/pdf",
            headers={
                "Cache-Control": "no-store",
                "Content-Disposition": f'inline; filename="{safe}"',
            },
        )

    @application.post("/ask", response_model=None)
    async def ask(
        payload: AskPayload, request: Request
    ) -> dict[str, object] | JSONResponse:
        """Answer the demo corpus, or one visitor collection its owner may read."""
        collection = (payload.collection or "").strip() or DEMO_COLLECTION
        app_cache = cast(AnswerCache, request.app.state.answer_cache)
        if collection == DEMO_COLLECTION:
            cached = app_cache.get(payload.question)
            if cached is not None:
                body = copy.deepcopy(cached.response)
                body["cached"] = True
                body["collection"] = DEMO_COLLECTION
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
        app_collection_retriever = cast(
            CollectionTraceRetriever | None,
            request.app.state.collection_retriever,
        )
        if collection != DEMO_COLLECTION:
            _require_origin(request)
            service = _upload_service(request)
            client_id = _require_client(request)
            if await service.collection_for(client_id, collection) is None:
                raise HTTPException(status_code=404, detail="unknown collection")
            if app_collection_retriever is None:
                raise HTTPException(
                    status_code=503, detail="collection retrieval is not ready"
                )
        try:
            response, trace = await AnswerService(
                retriever=app_retriever,
                model=app_model,
                calibration=app_calibration,
                trace_retriever=app_trace_retriever,
                collection_trace_retriever=app_collection_retriever,
            ).answer_with_trace(
                payload.question,
                None if collection == DEMO_COLLECTION else collection,
            )
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
                    "code": "model_unavailable",
                    "cached": False,
                    "detail": "The model endpoint did not return a usable answer.",
                },
            )
        body = dataclasses.asdict(response)
        body["cached"] = False
        body["collection"] = collection
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

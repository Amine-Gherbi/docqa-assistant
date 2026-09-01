"""The upload HTTP surface: session cookie, limits, ownership, collection ask.

Most tests drive the endpoints through a fake service so the status mapping is
checked exactly. Two tests use the real service against the local PostgreSQL to
prove that one client cannot read another client's document or page text.
"""

from __future__ import annotations

import asyncio
import dataclasses
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Literal

import httpx
import pymupdf
import pytest
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from docqa.answer import Calibration, ModelReply, ModelRequest
from docqa.api import create_app
from docqa.config import get_settings
from docqa.db import (
    Chunk as ChunkRow,
)
from docqa.db import (
    Document,
    Upload,
    make_engine,
    make_session_factory,
    upgrade_to_head,
)
from docqa.ingest.embedder import HashEmbedder
from docqa.retrieval import (
    PgCandidateSource,
    RetrievalResult,
    RetrievalTrace,
    RetrievedChunk,
)
from docqa.uploads.limits import UploadLimitError
from docqa.uploads.service import (
    UploadBusyError,
    UploadFile,
    UploadQuotaError,
    UploadService,
    UploadView,
)


def _calibration() -> Calibration:
    return Calibration(
        a=1.0,
        b=0.0,
        threshold=0.5,
        fit_set_sha256="b" * 64,
        uncalibrated=False,
        provenance="test fixture",
    )


class NeverCalledModel:
    """Fails the test if a retrieval reaches generation."""

    name = "fake:never-called"
    requested_model = "fake:never-called"
    structured_output: Literal["tool", "json", "extractive"] = "tool"

    async def complete(self, request: ModelRequest) -> ModelReply:
        """Never reached: the fake retrieval returns no chunks."""
        raise AssertionError("the model must not be called")


async def _empty(question: str) -> list[RetrievedChunk]:
    return []


def _empty_trace() -> RetrievalTrace:
    return RetrievalTrace(
        mode="hybrid_rerank",
        embedder="test",
        reranker="test",
        vector_hits=(),
        lexical_hits=(),
        fusion_hits=(),
        rerank_hits=(),
        context_chunk_ids=(),
        context_tokens=0,
    )


class RecordingRetriever:
    """A collection retriever that records the collection it was asked for."""

    def __init__(self) -> None:
        self.collections: list[str] = []

    async def __call__(self, question: str, collection: str) -> RetrievalResult:
        self.collections.append(collection)
        return RetrievalResult(chunks=(), trace=_empty_trace())


def _view(owner: str, **overrides: object) -> UploadView:
    now = datetime.now(UTC)
    values: dict[str, object] = {
        "id": str(uuid.uuid4()),
        "status": "ready",
        "collection": "u_fake",
        "filename": "visitor.pdf",
        "byte_size": 12,
        "pages_done": 1,
        "pages_total": 1,
        "error_code": None,
        "error_detail": None,
        "created_at": now,
        "expires_at": now + timedelta(hours=24),
    }
    values.update(overrides)
    return UploadView(**values)  # type: ignore[arg-type]


class FakeService:
    """A scriptable stand-in for the real upload service."""

    def __init__(self) -> None:
        self.owner: str | None = None
        self.upload = _view("nobody")
        self.error: Exception | None = None
        self.pages = {1: "the visitor page"}
        self.data = b"%PDF-1.7 fake"
        self.submissions: list[tuple[str, bytes]] = []

    async def submit(
        self, client_id: str, filename: str, content_type: str, raw: bytes
    ) -> UploadView:
        if self.error is not None:
            raise self.error
        self.owner = client_id
        self.submissions.append((filename, raw))
        return self.upload

    async def status(
        self, client_id: str, upload_id: uuid.UUID
    ) -> UploadView | None:
        if self.owner != client_id or str(upload_id) != self.upload.id:
            return None
        return self.upload

    async def file(self, client_id: str, upload_id: uuid.UUID) -> UploadFile | None:
        if self.owner != client_id or str(upload_id) != self.upload.id:
            return None
        return UploadFile(
            filename="visitor.pdf",
            content_type="application/pdf",
            sha256="c" * 64,
            data=self.data,
        )

    async def page(
        self, client_id: str, upload_id: uuid.UUID, page_no: int
    ) -> str | None:
        if self.owner != client_id or str(upload_id) != self.upload.id:
            return None
        return self.pages.get(page_no)

    async def collection_for(
        self, client_id: str, collection_id: str
    ) -> UploadView | None:
        if self.owner != client_id or collection_id != self.upload.collection:
            return None
        return self.upload


@asynccontextmanager
async def _client(
    service: FakeService,
    *,
    retriever: RecordingRetriever | None = None,
    collection_retriever: RecordingRetriever | None = None,
    **settings_overrides: object,
) -> AsyncIterator[httpx.AsyncClient]:
    settings = dataclasses.replace(
        get_settings(), cookie_secret="test-cookie-secret", **settings_overrides
    )
    app = create_app(
        settings=settings,
        calibration=_calibration(),
        model=NeverCalledModel(),
        retriever=_empty,
        collection_retriever=(
            collection_retriever
            if collection_retriever is not None
            else RecordingRetriever()
        ),
        upload_service=service,  # type: ignore[arg-type]
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


async def test_session_mints_an_http_only_signed_cookie() -> None:
    async with _client(FakeService()) as client:
        first = await client.get("/session")
        second = await client.get("/session")

    assert first.status_code == 200
    client_id = first.json()["client_id"]
    assert len(client_id) == 32
    assert second.json()["client_id"] == client_id
    set_cookie = first.headers["set-cookie"]
    assert "docqa_client=" in set_cookie
    assert "HttpOnly" in set_cookie
    assert "SameSite=lax" in set_cookie


async def test_a_hosted_cookie_is_secure_and_none_same_site() -> None:
    async with _client(FakeService(), cookie_secure=True) as client:
        response = await client.get("/session")
    cookie = response.headers["set-cookie"]
    assert "Secure" in cookie
    assert "SameSite=none" in cookie


async def test_upload_without_a_cookie_is_refused() -> None:
    async with _client(FakeService()) as client:
        response = await client.post(
            "/documents", files={"file": ("a.pdf", b"%PDF-1.7", "application/pdf")}
        )
    assert response.status_code == 401


async def test_upload_with_a_forged_cookie_is_refused() -> None:
    async with _client(FakeService()) as client:
        response = await client.post(
            "/documents",
            files={"file": ("a.pdf", b"%PDF-1.7", "application/pdf")},
            headers={"Cookie": f"docqa_client={'a' * 32}.{'0' * 64}"},
        )
    assert response.status_code == 401


@pytest.mark.parametrize(
    ("status_code", "code"),
    [(413, "too_large"), (415, "not_pdf"), (422, "page_limit")],
)
async def test_limit_errors_keep_their_status_and_code(
    status_code: int, code: str
) -> None:
    service = FakeService()
    service.error = UploadLimitError(status_code, code, "refused", limit=80)
    async with _client(service) as client:
        await client.get("/session")
        response = await client.post(
            "/documents", files={"file": ("a.pdf", b"%PDF-1.7", "application/pdf")}
        )
    assert response.status_code == status_code
    body = response.json()
    assert body["code"] == code
    assert body["state"] == "rejected"


async def test_a_second_upload_is_a_conflict_naming_the_running_one() -> None:
    service = FakeService()
    running = str(uuid.uuid4())
    service.error = UploadBusyError(running, "parsing")
    async with _client(service) as client:
        await client.get("/session")
        response = await client.post(
            "/documents", files={"file": ("a.pdf", b"%PDF-1.7", "application/pdf")}
        )
    assert response.status_code == 409
    assert response.json()["id"] == running
    assert response.json()["state"] == "busy"


async def test_a_quota_error_is_a_429_with_retry_after() -> None:
    service = FakeService()
    service.error = UploadQuotaError(
        "client_daily_limit",
        "slow down",
        retry_after_s=3600,
        reset_at=datetime.now(UTC) + timedelta(hours=1),
    )
    async with _client(service) as client:
        await client.get("/session")
        response = await client.post(
            "/documents", files={"file": ("a.pdf", b"%PDF-1.7", "application/pdf")}
        )
    assert response.status_code == 429
    assert response.headers["retry-after"] == "3600"
    assert response.json()["code"] == "client_daily_limit"


async def test_an_accepted_upload_returns_202_id_and_status() -> None:
    service = FakeService()
    service.upload = _view("x", status="queued")
    async with _client(service) as client:
        await client.get("/session")
        response = await client.post(
            "/documents",
            files={"file": ("visitor.pdf", b"%PDF-1.7", "application/pdf")},
        )
    assert response.status_code == 202
    body = response.json()
    assert body["id"] == service.upload.id
    assert body["status"] == "queued"


async def test_a_foreign_client_gets_404_for_status_page_and_file() -> None:
    service = FakeService()
    async with _client(service) as alice, _client(service) as bob:
        await alice.get("/session")
        await bob.get("/session")
        uploaded = await alice.post(
            "/documents", files={"file": ("a.pdf", b"%PDF-1.7", "application/pdf")}
        )
        upload_id = uploaded.json()["id"]
        mine = await alice.get(f"/documents/{upload_id}")
        foreign_status = await bob.get(f"/documents/{upload_id}")
        foreign_page = await bob.get(f"/documents/{upload_id}/pages/1")
        foreign_file = await bob.get(f"/documents/{upload_id}/file")
    assert mine.status_code == 200
    assert mine.json()["pages_done"] == 1
    assert foreign_status.status_code == 404
    assert foreign_page.status_code == 404
    assert foreign_file.status_code == 404


async def test_the_file_and_a_page_are_served_only_to_their_owner() -> None:
    service = FakeService()
    async with _client(service) as client:
        await client.get("/session")
        uploaded = await client.post(
            "/documents", files={"file": ("a.pdf", b"%PDF-1.7", "application/pdf")}
        )
        upload_id = uploaded.json()["id"]
        page = await client.get(f"/documents/{upload_id}/pages/1")
        missing = await client.get(f"/documents/{upload_id}/pages/9")
        file = await client.get(f"/documents/{upload_id}/file")
    assert page.status_code == 200
    assert page.json() == {"page_no": 1, "text": "the visitor page"}
    assert missing.status_code == 404
    assert file.status_code == 200
    assert file.headers["content-type"] == "application/pdf"
    assert file.headers["cache-control"] == "no-store"
    assert file.content == service.data


async def test_a_disallowed_origin_is_refused() -> None:
    async with _client(FakeService()) as client:
        await client.get("/session")
        response = await client.post(
            "/documents",
            files={"file": ("a.pdf", b"%PDF-1.7", "application/pdf")},
            headers={"Origin": "https://evil.example"},
        )
    assert response.status_code == 403


async def test_ask_on_an_upload_collection_uses_that_collection() -> None:
    service = FakeService()
    retriever = RecordingRetriever()
    async with _client(service, collection_retriever=retriever) as client:
        await client.get("/session")
        await client.post(
            "/documents", files={"file": ("a.pdf", b"%PDF-1.7", "application/pdf")}
        )
        response = await client.post(
            "/ask",
            json={"question": "What does it say?", "collection": service.upload.collection},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["insufficient_evidence"] is True
    assert body["cached"] is False
    assert retriever.collections == [service.upload.collection]


async def test_ask_on_a_foreign_collection_is_refused() -> None:
    service = FakeService()
    retriever = RecordingRetriever()
    async with _client(service, collection_retriever=retriever) as alice:
        await alice.get("/session")
        await alice.post(
            "/documents", files={"file": ("a.pdf", b"%PDF-1.7", "application/pdf")}
        )
        response = await alice.post(
            "/ask", json={"question": "What?", "collection": "u_someone_else"}
        )
    assert response.status_code == 404
    assert retriever.collections == []


# -- real service: isolation and collection-filtered SQL -------------------


def _pdf() -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "zebrafruit appears on the visitor page")
    raw = document.tobytes()
    document.close()
    return raw


async def _database() -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    settings = get_settings()
    try:
        await asyncio.to_thread(upgrade_to_head, settings.database_url)
    except Exception as exc:  # pragma: no cover - depends on the machine
        pytest.skip(f"PostgreSQL is unavailable: {exc}")
    engine = make_engine(settings)
    return engine, make_session_factory(engine)


async def test_a_second_client_cannot_read_the_uploads_status_or_bytes() -> None:
    engine, factory = await _database()
    service = UploadService(get_settings(), factory, HashEmbedder())
    settings = dataclasses.replace(
        get_settings(), cookie_secret="test-cookie-secret"
    )
    app = create_app(
        settings=settings,
        calibration=_calibration(),
        model=NeverCalledModel(),
        retriever=_empty,
        collection_retriever=RecordingRetriever(),
        upload_service=service,
    )
    transport = httpx.ASGITransport(app=app)
    upload_id: str | None = None
    try:
        async with (
            httpx.AsyncClient(transport=transport, base_url="http://test") as alice,
            httpx.AsyncClient(transport=transport, base_url="http://test") as bob,
        ):
            await alice.get("/session")
            await bob.get("/session")
            uploaded = await alice.post(
                "/documents", files={"file": ("mine.pdf", _pdf(), "application/pdf")}
            )
            assert uploaded.status_code == 202
            upload_id = uploaded.json()["id"]
            await service.drain()

            mine = await alice.get(f"/documents/{upload_id}")
            assert mine.status_code == 200
            assert mine.json()["status"] == "ready"
            page = await alice.get(f"/documents/{upload_id}/pages/1")
            assert page.status_code == 200
            assert "zebrafruit" in page.json()["text"]

            assert (await bob.get(f"/documents/{upload_id}")).status_code == 404
            assert (
                await bob.get(f"/documents/{upload_id}/pages/1")
            ).status_code == 404
            assert (await bob.get(f"/documents/{upload_id}/file")).status_code == 404
            collection = mine.json()["collection"]
            refused = await bob.post(
                "/ask", json={"question": "What?", "collection": collection}
            )
            assert refused.status_code == 404
    finally:
        async with factory() as session, session.begin():
            if upload_id is not None:
                row = await session.get(Upload, uuid.UUID(upload_id))
                if row is not None:
                    await session.execute(
                        delete(ChunkRow).where(
                            ChunkRow.collection_id == row.collection_id
                        )
                    )
                    await session.execute(
                        delete(Document).where(
                            Document.collection_id == row.collection_id
                        )
                    )
                    await session.delete(row)
        await engine.dispose()


async def test_candidate_sources_read_only_their_collection() -> None:
    engine, factory = await _database()
    demo_doc_id = f"tmp-demo-{uuid.uuid4().hex}"
    upload_doc_id = f"tmp-upload-{uuid.uuid4().hex}"
    collection = f"u_{uuid.uuid4().hex}"
    try:
        async with factory() as session, session.begin():
            demo_document = Document(
                doc_id=demo_doc_id,
                title="Demo",
                filename="demo.pdf",
                source="test",
                source_url="",
                licence="test",
                licence_url="",
                retrieved_at=datetime.now(UTC).date(),
                sha256=uuid.uuid4().hex * 2,
                byte_size=1,
                page_count=1,
                embedder="test",
                collection_id="demo",
            )
            visitor_document = Document(
                doc_id=upload_doc_id,
                title="Visitor",
                filename="visitor.pdf",
                source="test",
                source_url="",
                licence="test",
                licence_url="",
                retrieved_at=datetime.now(UTC).date(),
                sha256=uuid.uuid4().hex * 2,
                byte_size=1,
                page_count=1,
                embedder="test",
                collection_id=collection,
            )
            session.add_all([demo_document, visitor_document])
            await session.flush()
            session.add_all(
                [
                    ChunkRow(
                        document_id=demo_document.id,
                        chunk_index=0,
                        page_no=1,
                        char_start=0,
                        char_end=10,
                        page_text_sha256="0" * 64,
                        breadcrumb="Demo | p.1",
                        text="the demo sentence",
                        token_count=3,
                        collection_id="demo",
                        embedding=[1.0] + [0.0] * 383,
                    ),
                    ChunkRow(
                        document_id=visitor_document.id,
                        chunk_index=0,
                        page_no=1,
                        char_start=0,
                        char_end=10,
                        page_text_sha256="1" * 64,
                        breadcrumb="Visitor | p.1",
                        text="the zebrafruit sentence",
                        token_count=3,
                        collection_id=collection,
                        embedding=[1.0] + [0.0] * 383,
                    ),
                ]
            )

        async with factory() as session:
            demo = await PgCandidateSource(session).vector_candidates(
                [1.0] + [0.0] * 383, 10
            )
            demo_ids = {hit.doc_id for hit in demo}
            assert demo_doc_id in demo_ids
            assert upload_doc_id not in demo_ids

            lexical = await PgCandidateSource(session).lexical_candidates(
                "zebrafruit", 10
            )
            assert upload_doc_id not in {hit.doc_id for hit in lexical}
            assert not lexical  # the word exists only in the other collection

            visitor = await PgCandidateSource(
                session, collection=collection
            ).vector_candidates([1.0] + [0.0] * 383, 10)
            assert {hit.doc_id for hit in visitor} == {upload_doc_id}
    finally:
        async with factory() as session, session.begin():
            await session.execute(
                delete(ChunkRow).where(
                    ChunkRow.document_id.in_(
                        select(Document.id).where(
                            Document.doc_id.in_([demo_doc_id, upload_doc_id])
                        )
                    )
                )
            )
            await session.execute(
                delete(Document).where(
                    Document.doc_id.in_([demo_doc_id, upload_doc_id])
                )
            )
        await engine.dispose()

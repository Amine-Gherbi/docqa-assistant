"""The upload worker and service: quotas, one-at-a-time, status, cleanup.

These tests use the local PostgreSQL and skip when it is unreachable, like
``test_upload_store.py``. The embedder is the deterministic hash fallback so a
test never downloads model weights; the parse path is the real PyMuPDF one.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import uuid
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pymupdf
import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from docqa.config import get_settings
from docqa.db import (
    Chunk as ChunkRow,
)
from docqa.db import (
    Document,
    Upload,
    UploadPage,
    make_engine,
    make_session_factory,
    upgrade_to_head,
)
from docqa.ingest.chunking import build_chunks
from docqa.ingest.embedder import HashEmbedder
from docqa.ingest.extract import EncryptedPdfError, ExtractedPdf, Page, extract_pdf_bytes
from docqa.ingest.lock import LockEntry
from docqa.ingest.pipeline import _upsert_document
from docqa.uploads.limits import UploadLimitError
from docqa.uploads.service import UploadBusyError, UploadQuotaError, UploadService


def _pdf(pages: int) -> bytes:
    document = pymupdf.open()
    for index in range(pages):
        page = document.new_page()
        page.insert_text(
            (72, 72),
            f"visitor page {index + 1} " + "the quick brown fox jumps " * 8,
        )
    raw = document.tobytes()
    document.close()
    return raw


def _fake_extraction(texts: list[str]) -> ExtractedPdf:
    pages = tuple(
        Page(page_no=index, text=text, sha256=hashlib.sha256(text.encode()).hexdigest())
        for index, text in enumerate(texts, start=1)
    )
    return ExtractedPdf(
        path=Path("visitor.pdf"),
        sha256="a" * 64,
        byte_size=10,
        page_count=len(pages),
        pages=pages,
        pymupdf_version="2.0",
    )


async def _database() -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    settings = get_settings()
    try:
        await asyncio.to_thread(upgrade_to_head, settings.database_url)
    except Exception as exc:  # pragma: no cover - depends on the machine
        pytest.skip(f"PostgreSQL is unavailable: {exc}")
    engine = make_engine(settings)
    return engine, make_session_factory(engine)


def _service(
    factory: async_sessionmaker[AsyncSession],
    *,
    extractor: object = None,
    **overrides: object,
) -> UploadService:
    settings = dataclasses.replace(get_settings(), **overrides)
    return UploadService(settings, factory, HashEmbedder(), extractor=extractor)


async def _cleanup(
    factory: async_sessionmaker[AsyncSession], client_ids: list[str]
) -> None:
    async with factory() as session, session.begin():
        collections = (
            (
                await session.execute(
                    select(Upload.collection_id).where(
                        Upload.client_id.in_(client_ids)
                    )
                )
            )
            .scalars()
            .all()
        )
        if collections:
            await session.execute(
                delete(ChunkRow).where(ChunkRow.collection_id.in_(collections))
            )
            await session.execute(
                delete(Document).where(Document.collection_id.in_(collections))
            )
        await session.execute(
            delete(Upload).where(Upload.client_id.in_(client_ids))
        )


async def _insert_upload(
    session: AsyncSession,
    client_id: str,
    *,
    created_at: datetime | None = None,
    expires_at: datetime | None = None,
) -> Upload:
    now = datetime.now(UTC)
    upload = Upload(
        id=uuid.uuid4(),
        client_id=client_id,
        filename="prior.pdf",
        content_type="application/pdf",
        byte_size=9,
        sha256="d" * 64,
        status="ready",
        collection_id=f"u_{uuid.uuid4().hex}",
        pages_total=1,
        pages_done=1,
        data=b"%PDF-1.7",
        created_at=created_at or now,
        expires_at=expires_at or now + timedelta(hours=24),
    )
    session.add(upload)
    await session.flush()
    return upload


async def _wait_for_status(
    service: UploadService, client_id: str, upload_id: uuid.UUID, wanted: str
) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 5.0
    while loop.time() < deadline:
        view = await service.status(client_id, upload_id)
        if view is not None and view.status == wanted:
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"upload never reached {wanted!r}")


async def _count(factory: async_sessionmaker[AsyncSession], collection: str) -> int:
    async with factory() as session:
        return int(
            await session.scalar(
                select(func.count())
                .select_from(ChunkRow)
                .where(ChunkRow.collection_id == collection)
            )
            or 0
        )


async def test_an_upload_becomes_ready_with_chunks_in_its_own_collection() -> None:
    engine, factory = await _database()
    client = uuid.uuid4().hex
    try:
        service = _service(factory)
        submitted = await service.submit(
            client, "visitor.pdf", "application/pdf", _pdf(2)
        )
        assert submitted.status == "queued"
        await service.drain()

        view = await service.status(client, uuid.UUID(submitted.id))
        assert view is not None
        assert view.status == "ready"
        assert view.pages_total == 2
        assert view.pages_done == 2
        assert view.error_code is None
        assert await _count(factory, view.collection) >= 2
        assert await service.collection_for(client, view.collection) is not None
        assert await service.collection_for(uuid.uuid4().hex, view.collection) is None
    finally:
        await _cleanup(factory, [client])
        await engine.dispose()


async def test_a_second_upload_while_one_runs_is_refused_with_its_id() -> None:
    engine, factory = await _database()
    client = uuid.uuid4().hex
    gate = asyncio.Event()

    async def gated(raw: bytes) -> ExtractedPdf:
        await gate.wait()
        return extract_pdf_bytes(raw)

    try:
        service = _service(factory, extractor=gated)
        first = await service.submit(
            client, "first.pdf", "application/pdf", _pdf(1)
        )
        await _wait_for_status(service, client, uuid.UUID(first.id), "parsing")

        with pytest.raises(UploadBusyError) as caught:
            await service.submit(client, "second.pdf", "application/pdf", _pdf(1))
        assert caught.value.upload_id == first.id

        gate.set()
        await service.drain()
        view = await service.status(client, uuid.UUID(first.id))
        assert view is not None and view.status == "ready"
    finally:
        gate.set()
        await _cleanup(factory, [client])
        await engine.dispose()


async def test_the_fourth_upload_in_a_day_is_refused_with_a_reset() -> None:
    engine, factory = await _database()
    client = uuid.uuid4().hex
    try:
        async with factory() as session, session.begin():
            for minutes in (60, 120, 180):
                await _insert_upload(
                    session,
                    client,
                    created_at=datetime.now(UTC) - timedelta(minutes=minutes),
                )
        service = _service(factory)
        with pytest.raises(UploadQuotaError) as caught:
            await service.submit(client, "fourth.pdf", "application/pdf", _pdf(1))
        error = caught.value
        assert error.status_code == 429
        assert error.code == "client_daily_limit"
        assert 0 < error.retry_after_s <= 24 * 60 * 60
        assert await service.status(client, uuid.uuid4()) is None
    finally:
        await _cleanup(factory, [client])
        await engine.dispose()


async def test_the_twenty_first_upload_in_an_hour_is_refused() -> None:
    engine, factory = await _database()
    clients = [uuid.uuid4().hex for _ in range(20)]
    try:
        async with factory() as session, session.begin():
            for index, other in enumerate(clients):
                await _insert_upload(
                    session,
                    other,
                    created_at=datetime.now(UTC)
                    - timedelta(minutes=index + 1),
                )
        service = _service(factory)
        stranger = uuid.uuid4().hex
        with pytest.raises(UploadQuotaError) as caught:
            await service.submit(stranger, "one.pdf", "application/pdf", _pdf(1))
        assert caught.value.status_code == 429
        assert caught.value.code == "host_hourly_limit"
        assert 0 < caught.value.retry_after_s <= 60 * 60
    finally:
        await _cleanup(factory, [*clients, stranger])
        await engine.dispose()


async def test_a_parse_timeout_marks_failed_and_removes_partial_rows() -> None:
    engine, factory = await _database()
    client = uuid.uuid4().hex

    async def hanging(raw: bytes) -> ExtractedPdf:
        await asyncio.sleep(30)
        return extract_pdf_bytes(raw)

    try:
        service = _service(factory, upload_timeout_s=0.05, extractor=hanging)
        submitted = await service.submit(
            client, "hang.pdf", "application/pdf", _pdf(1)
        )
        await service.drain()
        view = await service.status(client, uuid.UUID(submitted.id))
        assert view is not None
        assert view.status == "failed"
        assert view.error_code == "timeout"
        assert await _count(factory, view.collection) == 0
        async with factory() as session:
            assert await session.scalar(
                select(func.count())
                .select_from(UploadPage)
                .where(UploadPage.upload_id == uuid.UUID(view.id))
            ) == 0
    finally:
        await _cleanup(factory, [client])
        await engine.dispose()


async def test_an_exception_marks_failed_with_its_code_and_cleans_up() -> None:
    engine, factory = await _database()
    client = uuid.uuid4().hex

    async def broken(raw: bytes) -> ExtractedPdf:
        raise ValueError("PyMuPDF gave up")

    try:
        service = _service(factory, extractor=broken)
        submitted = await service.submit(
            client, "broken.pdf", "application/pdf", _pdf(1)
        )
        await service.drain()
        view = await service.status(client, uuid.UUID(submitted.id))
        assert view is not None
        assert view.status == "failed"
        assert view.error_code == "failed"
        assert "PyMuPDF gave up" in (view.error_detail or "")
        assert await _count(factory, view.collection) == 0
    finally:
        await _cleanup(factory, [client])
        await engine.dispose()


async def test_the_worker_enforces_the_page_limit() -> None:
    engine, factory = await _database()
    client = uuid.uuid4().hex

    async def three_pages(raw: bytes) -> ExtractedPdf:
        return _fake_extraction(["one", "two", "three"])

    try:
        service = _service(factory, upload_max_pages=2, extractor=three_pages)
        submitted = await service.submit(
            client, "long.pdf", "application/pdf", _pdf(1)
        )
        await service.drain()
        view = await service.status(client, uuid.UUID(submitted.id))
        assert view is not None
        assert view.status == "failed"
        assert view.error_code == "page_limit"
    finally:
        await _cleanup(factory, [client])
        await engine.dispose()


async def test_an_encrypted_upload_fails_with_the_encrypted_code() -> None:
    engine, factory = await _database()
    client = uuid.uuid4().hex

    async def encrypted(raw: bytes) -> ExtractedPdf:
        raise EncryptedPdfError("password required")

    try:
        service = _service(factory, extractor=encrypted)
        submitted = await service.submit(
            client, "secret.pdf", "application/pdf", _pdf(1)
        )
        await service.drain()
        view = await service.status(client, uuid.UUID(submitted.id))
        assert view is not None
        assert view.status == "failed"
        assert view.error_code == "encrypted"
    finally:
        await _cleanup(factory, [client])
        await engine.dispose()


async def test_a_scanned_pdf_without_text_fails_instead_of_being_ready() -> None:
    engine, factory = await _database()
    client = uuid.uuid4().hex

    async def empty(raw: bytes) -> ExtractedPdf:
        return _fake_extraction(["", "   "])

    try:
        service = _service(factory, extractor=empty)
        submitted = await service.submit(
            client, "scan.pdf", "application/pdf", _pdf(1)
        )
        await service.drain()
        view = await service.status(client, uuid.UUID(submitted.id))
        assert view is not None
        assert view.status == "failed"
        assert view.error_code == "no_text"
    finally:
        await _cleanup(factory, [client])
        await engine.dispose()


async def test_two_clients_can_upload_the_same_bytes() -> None:
    engine, factory = await _database()
    first_client = uuid.uuid4().hex
    second_client = uuid.uuid4().hex
    raw = _pdf(1)
    try:
        service = _service(factory)
        first = await service.submit(
            first_client, "same.pdf", "application/pdf", raw
        )
        await service.drain()
        second = await service.submit(
            second_client, "same.pdf", "application/pdf", raw
        )
        await service.drain()

        first_view = await service.status(first_client, uuid.UUID(first.id))
        second_view = await service.status(second_client, uuid.UUID(second.id))
        assert first_view is not None and first_view.status == "ready"
        assert second_view is not None and second_view.status == "ready"
        assert first_view.collection != second_view.collection
    finally:
        await _cleanup(factory, [first_client, second_client])
        await engine.dispose()


async def test_the_service_sweep_deletes_an_expired_upload() -> None:
    engine, factory = await _database()
    client = uuid.uuid4().hex
    try:
        async with factory() as session, session.begin():
            expired = await _insert_upload(
                session,
                client,
                expires_at=datetime.now(UTC) - timedelta(minutes=1),
            )
        service = _service(factory)
        report = await service.sweep()
        assert report.uploads >= 1
        async with factory() as session:
            assert await session.get(Upload, expired.id) is None
    finally:
        await _cleanup(factory, [client])
        await engine.dispose()


async def test_the_service_refuses_a_file_over_the_byte_limit() -> None:
    engine, factory = await _database()
    client = uuid.uuid4().hex
    try:
        service = _service(factory, upload_max_bytes=10)
        with pytest.raises(UploadLimitError) as caught:
            await service.submit(client, "big.pdf", "application/pdf", _pdf(1))
        assert caught.value.status_code == 413
        assert caught.value.code == "too_large"
    finally:
        await _cleanup(factory, [client])
        await engine.dispose()


async def test_demo_ingest_ignores_a_visitor_document_with_the_same_bytes() -> None:
    engine, factory = await _database()
    client = uuid.uuid4().hex
    raw = _pdf(1)
    try:
        service = _service(factory)
        submitted = await service.submit(
            client, "same.pdf", "application/pdf", raw
        )
        await service.drain()
        view = await service.status(client, uuid.UUID(submitted.id))
        assert view is not None and view.status == "ready"

        extracted = extract_pdf_bytes(raw, "corpus.pdf")
        entry = LockEntry(
            doc_id="demo-same-bytes",
            filename="corpus.pdf",
            title="Pinned Paper",
            source="corpus",
            url="file://corpus.pdf",
            sha256=extracted.sha256,
            bytes=extracted.byte_size,
            pages=1,
            licence="self-generated",
            licence_url="https://example.invalid/licence",
            retrieved_at=date(2026, 10, 29),
        )
        chunks = build_chunks(extracted, HashEmbedder().counter, entry.title)
        vectors = [[0.0] * 384 for _ in chunks]
        async with factory() as session, session.begin():
            updated = await _upsert_document(
                session, entry, extracted, chunks, vectors, "test"
            )
        assert updated is False

        async with factory() as session:
            demo = await session.scalar(
                select(Document).where(
                    Document.collection_id == "demo",
                    Document.doc_id == entry.doc_id,
                )
            )
            assert demo is not None
            demo_chunks = await session.scalar(
                select(func.count())
                .select_from(ChunkRow)
                .where(ChunkRow.document_id == demo.id)
            )
            assert demo_chunks == len(chunks)
            assert (
                await session.scalar(
                    select(func.count())
                    .select_from(ChunkRow)
                    .where(
                        ChunkRow.document_id == demo.id,
                        ChunkRow.collection_id == "demo",
                    )
                )
                == len(chunks)
            )
            visitor = await session.scalar(
                select(Document).where(Document.collection_id == view.collection)
            )
            assert visitor is not None
            assert visitor.doc_id == f"upload:{view.collection}"
    finally:
        async with factory() as session, session.begin():
            await session.execute(
                delete(Document).where(Document.doc_id == "demo-same-bytes")
            )
        await _cleanup(factory, [client])
        await engine.dispose()

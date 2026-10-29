"""Upload rows, quotas, ownership, and the idempotent TTL sweep.

These tests use the local PostgreSQL. They skip when it is unreachable, like
``test_idempotency.py``; the hosted proof is recorded in ``docs/PROOF.md``.
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from datetime import UTC, date, datetime, timedelta

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
from docqa.uploads.store import (
    active_upload_for,
    count_client_uploads_since,
    count_host_uploads_since,
    owned_upload,
    page_for,
    ready_upload_for_collection,
    recover_interrupted,
    sweep_expired,
)

CLIENT_A = "a" * 32
CLIENT_B = "b" * 32


async def _database() -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    settings = get_settings()
    try:
        await asyncio.to_thread(upgrade_to_head, settings.database_url)
    except Exception as exc:  # pragma: no cover - depends on the machine
        pytest.skip(f"PostgreSQL is unavailable: {exc}")
    engine = make_engine(settings)
    return engine, make_session_factory(engine)


def _document(collection_id: str) -> Document:
    return Document(
        doc_id=f"upload:{collection_id}",
        title="Visitor document",
        filename="visitor.pdf",
        source="visitor upload",
        source_url="",
        licence="visitor-supplied",
        licence_url="",
        retrieved_at=date(2026, 10, 29),
        sha256=hashlib.sha256(collection_id.encode()).hexdigest(),
        byte_size=10,
        page_count=1,
        embedder="test",
        collection_id=collection_id,
    )


async def _insert_upload(
    session: AsyncSession,
    client_id: str,
    *,
    status: str = "ready",
    collection_id: str | None = None,
    expires_delta: timedelta = timedelta(hours=24),
    created_delta: timedelta = timedelta(),
    started_delta: timedelta | None = None,
) -> Upload:
    now = datetime.now(UTC)
    collection = collection_id or f"u_{uuid.uuid4().hex}"
    upload = Upload(
        id=uuid.uuid4(),
        client_id=client_id,
        filename="visitor.pdf",
        content_type="application/pdf",
        byte_size=9,
        sha256="d" * 64,
        status=status,
        collection_id=collection,
        pages_total=1,
        pages_done=1 if status == "ready" else 0,
        data=b"%PDF-1.7",
        created_at=now + created_delta,
        started_at=None if started_delta is None else now + started_delta,
        expires_at=now + expires_delta,
    )
    session.add(upload)
    await session.flush()
    return upload


async def test_ownership_is_scoped_to_the_signing_client() -> None:
    engine, factory = await _database()
    try:
        async with factory() as session, session.begin():
            upload = await _insert_upload(session, CLIENT_A)

        async with factory() as session:
            assert await owned_upload(session, upload.id, CLIENT_A) is not None
            assert await owned_upload(session, upload.id, CLIENT_B) is None
            assert await owned_upload(session, uuid.uuid4(), CLIENT_A) is None
            assert (
                await ready_upload_for_collection(
                    session, CLIENT_A, upload.collection_id
                )
                is not None
            )
            assert (
                await ready_upload_for_collection(
                    session, CLIENT_B, upload.collection_id
                )
                is None
            )
    finally:
        async with factory() as session, session.begin():
            await session.execute(delete(Upload).where(Upload.client_id == CLIENT_A))
            await session.execute(
                delete(Document).where(Document.doc_id.like("upload:u_%"))
            )
        await engine.dispose()


async def test_only_a_ready_upload_backs_a_collection() -> None:
    engine, factory = await _database()
    try:
        async with factory() as session, session.begin():
            upload = await _insert_upload(session, CLIENT_A, status="parsing")
        async with factory() as session:
            assert (
                await ready_upload_for_collection(
                    session, CLIENT_A, upload.collection_id
                )
                is None
            )
    finally:
        async with factory() as session, session.begin():
            await session.execute(delete(Upload).where(Upload.id == upload.id))
        await engine.dispose()


async def test_quota_counts_are_windowed_per_client_and_host() -> None:
    engine, factory = await _database()
    since = datetime.now(UTC) - timedelta(hours=1)
    try:
        async with factory() as session, session.begin():
            await _insert_upload(session, CLIENT_A, created_delta=timedelta(minutes=-5))
            await _insert_upload(session, CLIENT_A, created_delta=timedelta(minutes=-10))
            await _insert_upload(session, CLIENT_B, created_delta=timedelta(minutes=-15))
            await _insert_upload(
                session, CLIENT_A, created_delta=timedelta(hours=-2)
            )
        async with factory() as session:
            assert await count_client_uploads_since(session, CLIENT_A, since) == 2
            assert await count_host_uploads_since(session, since) == 3
    finally:
        async with factory() as session, session.begin():
            await session.execute(
                delete(Upload).where(Upload.client_id.in_([CLIENT_A, CLIENT_B]))
            )
        await engine.dispose()


async def test_active_upload_blocks_a_second_ingest_per_client() -> None:
    engine, factory = await _database()
    try:
        async with factory() as session, session.begin():
            first = await _insert_upload(session, CLIENT_A, status="embedding")
        async with factory() as session:
            active = await active_upload_for(session, CLIENT_A)
            assert active is not None and active.id == first.id
            assert await active_upload_for(session, CLIENT_B) is None
    finally:
        async with factory() as session, session.begin():
            await session.execute(delete(Upload).where(Upload.id == first.id))
        await engine.dispose()


async def test_sweep_removes_the_upload_its_pages_document_and_vectors() -> None:
    engine, factory = await _database()
    expired_collection = f"u_{uuid.uuid4().hex}"
    live_collection = f"u_{uuid.uuid4().hex}"
    try:
        async with factory() as session, session.begin():
            expired = await _insert_upload(
                session,
                CLIENT_A,
                collection_id=expired_collection,
                expires_delta=timedelta(hours=-1),
            )
            live = await _insert_upload(
                session,
                CLIENT_A,
                collection_id=live_collection,
                expires_delta=timedelta(hours=1),
            )
            session.add(
                UploadPage(
                    upload_id=expired.id,
                    page_no=1,
                    text="expired page",
                    page_text_sha256="e" * 64,
                )
            )
            session.add(
                UploadPage(
                    upload_id=live.id,
                    page_no=1,
                    text="live page",
                    page_text_sha256="f" * 64,
                )
            )
            for collection in (expired_collection, live_collection):
                document = _document(collection)
                session.add(document)
                await session.flush()
                session.add(
                    ChunkRow(
                        document_id=document.id,
                        chunk_index=0,
                        page_no=1,
                        char_start=0,
                        char_end=5,
                        page_text_sha256="0" * 64,
                        breadcrumb="Visitor document | p.1",
                        text="visitor text",
                        token_count=2,
                        embedding=[0.0] * 384,
                        collection_id=collection,
                    )
                )

        async with factory() as session, session.begin():
            report = await sweep_expired(session)
        assert report.uploads >= 1
        assert report.documents >= 1
        assert report.chunks >= 1
        assert report.pages >= 1

        async with factory() as session:
            assert await session.scalar(
                select(func.count()).select_from(Upload).where(
                    Upload.collection_id == expired_collection
                )
            ) == 0
            assert await session.scalar(
                select(func.count()).select_from(UploadPage).where(
                    UploadPage.upload_id == expired.id
                )
            ) == 0
            assert await session.scalar(
                select(func.count()).select_from(Document).where(
                    Document.collection_id == expired_collection
                )
            ) == 0
            assert await session.scalar(
                select(func.count()).select_from(ChunkRow).where(
                    ChunkRow.collection_id == expired_collection
                )
            ) == 0
            assert await session.scalar(
                select(func.count()).select_from(Upload).where(
                    Upload.id == live.id
                )
            ) == 1
            assert await session.scalar(
                select(func.count()).select_from(ChunkRow).where(
                    ChunkRow.collection_id == live_collection
                )
            ) == 1

        async with factory() as session, session.begin():
            again = await sweep_expired(session)
        assert again.uploads == 0
    finally:
        async with factory() as session, session.begin():
            await session.execute(
                delete(Upload).where(
                    Upload.collection_id.in_([expired_collection, live_collection])
                )
            )
            await session.execute(
                delete(ChunkRow).where(
                    ChunkRow.collection_id.in_([expired_collection, live_collection])
                )
            )
            await session.execute(
                delete(Document).where(
                    Document.collection_id.in_([expired_collection, live_collection])
                )
            )
        await engine.dispose()


async def test_page_text_is_served_only_for_its_upload() -> None:
    engine, factory = await _database()
    try:
        async with factory() as session, session.begin():
            upload = await _insert_upload(session, CLIENT_A)
            session.add(
                UploadPage(
                    upload_id=upload.id,
                    page_no=1,
                    text="the visitor page",
                    page_text_sha256="1" * 64,
                )
            )
        async with factory() as session:
            owned = await owned_upload(session, upload.id, CLIENT_A)
            assert owned is not None
            page = await page_for(session, owned, 1)
            assert page is not None and page.text == "the visitor page"
            assert await page_for(session, owned, 2) is None
            assert await owned_upload(session, upload.id, CLIENT_B) is None
    finally:
        async with factory() as session, session.begin():
            await session.execute(delete(Upload).where(Upload.id == upload.id))
        await engine.dispose()


async def test_recovery_fails_an_upload_interrupted_by_a_restart() -> None:
    engine, factory = await _database()
    try:
        async with factory() as session, session.begin():
            stale = await _insert_upload(
                session,
                CLIENT_A,
                status="parsing",
                created_delta=timedelta(hours=-2),
                started_delta=timedelta(hours=-2),
            )
        async with factory() as session, session.begin():
            recovered = await recover_interrupted(
                session, older_than=datetime.now(UTC) - timedelta(minutes=10)
            )
            assert recovered >= 1
        async with factory() as session:
            row = await session.get(Upload, stale.id)
            assert row is not None
            assert row.status == "failed"
            assert row.error_code == "interrupted"
            assert await active_upload_for(session, CLIENT_A) is None
    finally:
        async with factory() as session, session.begin():
            await session.execute(delete(Upload).where(Upload.id == stale.id))
        await engine.dispose()

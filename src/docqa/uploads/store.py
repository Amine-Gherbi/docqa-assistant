"""Database operations for visitor uploads: quota, ownership, TTL.

The request loop only reads and inserts rows here; parsing belongs to the
worker. The sweep deletes chunks explicitly (the vectors) before the document
and the upload row, so an expired collection can never outlive its bytes.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

from sqlalchemy import CursorResult, delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from docqa.db import Chunk as ChunkRow
from docqa.db import Document, Upload, UploadPage

RUNNING_STATUSES = ("queued", "parsing", "embedding")


@dataclass(frozen=True)
class SweepReport:
    """How many rows one sweep removed."""

    uploads: int
    documents: int
    chunks: int
    pages: int


def utcnow() -> datetime:
    """Return a timezone-aware ``now``; the database columns are timestamptz."""
    return datetime.now(UTC)


async def count_client_uploads_since(
    session: AsyncSession, client_id: str, since: datetime
) -> int:
    """Count uploads this client made since ``since`` (every attempt counts)."""
    return int(
        await session.scalar(
            select(func.count())
            .select_from(Upload)
            .where(Upload.client_id == client_id, Upload.created_at >= since)
        )
        or 0
    )


async def count_host_uploads_since(session: AsyncSession, since: datetime) -> int:
    """Count uploads the whole host accepted since ``since``."""
    return int(
        await session.scalar(
            select(func.count()).select_from(Upload).where(Upload.created_at >= since)
        )
        or 0
    )


async def active_upload_for(session: AsyncSession, client_id: str) -> Upload | None:
    """Return the one queued/parsing/embedding upload for a client, if any."""
    return await session.scalar(
        select(Upload)
        .where(Upload.client_id == client_id, Upload.status.in_(RUNNING_STATUSES))
        .order_by(Upload.created_at.desc())
        .limit(1)
    )


async def owned_upload(
    session: AsyncSession, upload_id: uuid.UUID, client_id: str
) -> Upload | None:
    """Return an upload only when this signed client owns it."""
    return await session.scalar(
        select(Upload).where(
            Upload.id == upload_id, Upload.client_id == client_id
        )
    )


async def ready_upload_for_collection(
    session: AsyncSession, client_id: str, collection_id: str
) -> Upload | None:
    """Return a ready upload that authorises ``/ask`` for this client and collection."""
    return await session.scalar(
        select(Upload).where(
            Upload.collection_id == collection_id,
            Upload.client_id == client_id,
            Upload.status == "ready",
        )
    )


async def page_for(
    session: AsyncSession, upload: Upload, page_no: int
) -> UploadPage | None:
    """Return the stored text for one page of one upload."""
    return await session.scalar(
        select(UploadPage).where(
            UploadPage.upload_id == upload.id, UploadPage.page_no == page_no
        )
    )


async def recover_interrupted(session: AsyncSession, older_than: datetime) -> int:
    """Fail uploads whose worker died with the previous instance.

    Without this, a scale-to-zero restart during ingest would leave the client
    permanently blocked by the one-at-a-time rule.
    """
    result = await session.execute(
        update(Upload)
        .where(
            Upload.status.in_(RUNNING_STATUSES),
            func.coalesce(Upload.started_at, Upload.created_at) < older_than,
        )
        .values(
            status="failed",
            error_code="interrupted",
            error_detail=(
                "The instance restarted while this upload was being processed; "
                "upload it again."
            ),
            finished_at=func.now(),
        )
    )
    return int(cast(CursorResult[Any], result).rowcount or 0)


async def sweep_expired(
    session: AsyncSession, *, now: datetime | None = None
) -> SweepReport:
    """Delete every expired upload, its pages, document, chunks and vectors.

    Idempotent: a second call finds nothing and returns zeros.
    """
    moment = now or utcnow()
    rows = (
        await session.execute(
            select(Upload.id, Upload.collection_id).where(
                Upload.expires_at <= moment
            )
        )
    ).all()
    if not rows:
        return SweepReport(uploads=0, documents=0, chunks=0, pages=0)

    upload_ids = [row[0] for row in rows]
    collections = [row[1] for row in rows]
    pages = int(
        await session.scalar(
            select(func.count())
            .select_from(UploadPage)
            .where(UploadPage.upload_id.in_(upload_ids))
        )
        or 0
    )
    chunks = int(
        await session.scalar(
            select(func.count())
            .select_from(ChunkRow)
            .where(ChunkRow.collection_id.in_(collections))
        )
        or 0
    )
    documents = int(
        await session.scalar(
            select(func.count())
            .select_from(Document)
            .where(Document.collection_id.in_(collections))
        )
        or 0
    )

    await session.execute(
        delete(ChunkRow).where(ChunkRow.collection_id.in_(collections))
    )
    await session.execute(
        delete(Document).where(Document.collection_id.in_(collections))
    )
    await session.execute(
        delete(UploadPage).where(UploadPage.upload_id.in_(upload_ids))
    )
    deleted = await session.execute(
        delete(Upload).where(Upload.id.in_(upload_ids))
    )
    return SweepReport(
        uploads=int(cast(CursorResult[Any], deleted).rowcount or 0),
        documents=documents,
        chunks=chunks,
        pages=pages,
    )

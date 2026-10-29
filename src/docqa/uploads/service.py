"""Accept, track, and sweep visitor uploads without parsing in the request loop.

``submit`` checks the byte/format/page limits it can see cheaply, then takes a
transaction-scoped advisory lock so two concurrent requests cannot both pass the
quota check. It inserts a ``queued`` row and returns; a background task takes it
from there.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from contextlib import AbstractAsyncContextManager, suppress
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from docqa.config import Settings
from docqa.db import Upload, session_scope
from docqa.ingest.embedder import Embedder
from docqa.uploads.limits import check_upload_bytes
from docqa.uploads.store import (
    SweepReport,
    active_upload_for,
    count_client_uploads_since,
    count_host_uploads_since,
    owned_upload,
    page_for,
    ready_upload_for_collection,
    recover_interrupted,
    sweep_expired,
    utcnow,
)
from docqa.uploads.worker import Extractor, extract_in_thread, process_upload

LOGGER = logging.getLogger(__name__)

# One lock serialises submit, not the worker: the quota check and the insert
# must see the same row set. Submissions are rare enough that a host-wide lock
# costs nothing, and a single key removes any lock-ordering question.
UPLOAD_LOCK_KEY = 7_102_201
DAY = timedelta(hours=24)
HOUR = timedelta(hours=1)
STALE_AFTER = timedelta(minutes=10)


class UploadBusyError(Exception):
    """An upload for this client is already queued or running."""

    def __init__(self, upload_id: str, status: str) -> None:
        """Name the upload that must finish first."""
        super().__init__(f"upload {upload_id} is still {status}")
        self.upload_id = upload_id
        self.status = status


class UploadQuotaError(Exception):
    """A quota window is full; ``retry_after_s`` says when it reopens."""

    status_code = 429

    def __init__(
        self,
        code: str,
        detail: str,
        *,
        retry_after_s: int,
        reset_at: datetime,
    ) -> None:
        """Carry the machine-readable reason and the reset moment."""
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.retry_after_s = retry_after_s
        self.reset_at = reset_at


@dataclass(frozen=True)
class UploadView:
    """Everything a client may see about its own upload."""

    id: str
    status: str
    collection: str
    filename: str
    byte_size: int
    pages_done: int
    pages_total: int | None
    error_code: str | None
    error_detail: str | None
    created_at: datetime
    expires_at: datetime


@dataclass(frozen=True)
class UploadFile:
    """The original bytes, served only to the owning client."""

    filename: str
    content_type: str
    sha256: str
    data: bytes


def _view(upload: Upload) -> UploadView:
    """Project a row onto what the API may return."""
    return UploadView(
        id=str(upload.id),
        status=upload.status,
        collection=upload.collection_id,
        filename=upload.filename,
        byte_size=upload.byte_size,
        pages_done=upload.pages_done,
        pages_total=upload.pages_total,
        error_code=upload.error_code,
        error_detail=upload.error_detail,
        created_at=upload.created_at,
        expires_at=upload.expires_at,
    )


class UploadService:
    """Owns the upload table, the worker tasks, and the TTL sweep."""

    def __init__(
        self,
        settings: Settings,
        factory: async_sessionmaker[AsyncSession],
        embedder: Embedder,
        *,
        extractor: Extractor | None = None,
    ) -> None:
        """Bind the database, the shared embedder, and the parse path."""
        self._settings = settings
        self._factory = factory
        self._embedder = embedder
        self._extractor = extractor or extract_in_thread
        self._tasks: set[asyncio.Task[None]] = set()
        self._sweeper: asyncio.Task[None] | None = None
        self._started = False

    # -- request path -----------------------------------------------------

    async def submit(
        self, client_id: str, filename: str, content_type: str, raw: bytes
    ) -> UploadView:
        """Validate, enforce quotas, store the bytes, and schedule the worker."""
        check_upload_bytes(
            raw,
            max_bytes=self._settings.upload_max_bytes,
            max_pages=self._settings.upload_max_pages,
        )
        now = utcnow()
        upload_id = uuid.uuid4()
        upload = Upload(
            id=upload_id,
            client_id=client_id,
            filename=_safe_name(filename),
            content_type=(content_type or "application/pdf")[:200],
            byte_size=len(raw),
            sha256=_sha256(raw),
            status="queued",
            collection_id=f"u_{upload_id.hex}",
            pages_total=None,
            pages_done=0,
            data=raw,
            created_at=now,
            expires_at=now + timedelta(hours=self._settings.upload_ttl_hours),
        )
        async with session_scope(self._factory) as session:
            await session.execute(
                text("select pg_advisory_xact_lock(:key)"),
                {"key": UPLOAD_LOCK_KEY},
            )
            running = await active_upload_for(session, client_id)
            if running is not None:
                raise UploadBusyError(str(running.id), running.status)
            await self._check_client_day(session, client_id, now)
            await self._check_host_hour(session, now)
            session.add(upload)
        self._schedule(upload_id)
        return _view(upload)

    async def status(
        self, client_id: str, upload_id: uuid.UUID
    ) -> UploadView | None:
        """Return this client's upload, or ``None`` for a foreign or unknown id."""
        async with session_scope(self._factory) as session:
            upload = await owned_upload(session, upload_id, client_id)
            return None if upload is None else _view(upload)

    async def file(self, client_id: str, upload_id: uuid.UUID) -> UploadFile | None:
        """Return the original PDF bytes, only for the owning client."""
        async with session_scope(self._factory) as session:
            upload = await owned_upload(session, upload_id, client_id)
            if upload is None:
                return None
            return UploadFile(
                filename=upload.filename,
                content_type=upload.content_type,
                sha256=upload.sha256,
                data=bytes(upload.data),
            )

    async def page(
        self, client_id: str, upload_id: uuid.UUID, page_no: int
    ) -> str | None:
        """One stored page's exact text, only for the owning client."""
        async with session_scope(self._factory) as session:
            upload = await owned_upload(session, upload_id, client_id)
            if upload is None:
                return None
            page = await page_for(session, upload, page_no)
            return None if page is None else page.text

    async def collection_for(
        self, client_id: str, collection_id: str
    ) -> UploadView | None:
        """Authorise ``/ask``: a ready upload owned by this client."""
        async with session_scope(self._factory) as session:
            upload = await ready_upload_for_collection(
                session, client_id, collection_id
            )
            return None if upload is None else _view(upload)

    def session(self) -> AbstractAsyncContextManager[AsyncSession]:
        """Open a transaction-scoped session for callers that need one."""
        return session_scope(self._factory)

    # -- background work --------------------------------------------------

    def _schedule(self, upload_id: uuid.UUID) -> None:
        """Start one worker task and forget it when it finishes."""
        task = asyncio.create_task(
            process_upload(
                upload_id,
                factory=self._factory,
                embedder=self._embedder,
                max_pages=self._settings.upload_max_pages,
                timeout_s=self._settings.upload_timeout_s,
                extractor=self._extractor,
            ),
            name=f"upload:{upload_id}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def start(self) -> None:
        """Sweep expired rows at boot, fail interrupted work, then sweep on a loop."""
        if self._started:
            return
        self._started = True
        try:
            async with session_scope(self._factory) as session:
                await recover_interrupted(session, older_than=utcnow() - STALE_AFTER)
            await self.sweep()
        except Exception:
            LOGGER.exception("upload boot sweep failed")
        self._sweeper = asyncio.create_task(self._sweep_loop(), name="upload-sweep")

    async def stop(self) -> None:
        """Stop the sweep loop and wait for in-flight workers."""
        if self._sweeper is not None:
            self._sweeper.cancel()
            with suppress(asyncio.CancelledError):
                await self._sweeper
            self._sweeper = None
        await self.drain()
        self._started = False

    async def drain(self) -> None:
        """Wait until every scheduled worker task has finished."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    async def sweep(self) -> SweepReport:
        """Delete every expired upload inside one transaction."""
        async with session_scope(self._factory) as session:
            return await sweep_expired(session)

    async def _sweep_loop(self) -> None:
        """Sweep on an interval until ``stop`` cancels it."""
        while True:
            await asyncio.sleep(self._settings.upload_sweep_interval_s)
            try:
                report = await self.sweep()
                if report.uploads:
                    LOGGER.info(
                        "upload sweep removed %s uploads, %s chunks, %s documents",
                        report.uploads,
                        report.chunks,
                        report.documents,
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                LOGGER.exception("upload sweep failed")

    # -- quota internals --------------------------------------------------

    async def _check_client_day(
        self, session: AsyncSession, client_id: str, now: datetime
    ) -> None:
        """Refuse a fourth upload in 24 hours, naming when the first expires."""
        since = now - DAY
        used = await count_client_uploads_since(session, client_id, since)
        if used < self._settings.upload_per_client_day:
            return
        oldest = await session.scalar(
            select(func.min(Upload.created_at)).where(
                Upload.client_id == client_id, Upload.created_at >= since
            )
        )
        reset_at = _reset_at(oldest, now, DAY)
        raise UploadQuotaError(
            "client_daily_limit",
            f"Each visitor may upload {self._settings.upload_per_client_day} "
            "documents per 24 hours.",
            retry_after_s=_seconds_until(reset_at, now),
            reset_at=reset_at,
        )

    async def _check_host_hour(
        self, session: AsyncSession, now: datetime
    ) -> None:
        """Refuse the 21st upload in an hour across the whole host."""
        since = now - HOUR
        used = await count_host_uploads_since(session, since)
        if used < self._settings.upload_per_host_hour:
            return
        oldest = await session.scalar(
            select(func.min(Upload.created_at)).where(Upload.created_at >= since)
        )
        reset_at = _reset_at(oldest, now, HOUR)
        raise UploadQuotaError(
            "host_hourly_limit",
            "The shared demo is taking a short break from new uploads; "
            "try again in a few minutes.",
            retry_after_s=_seconds_until(reset_at, now),
            reset_at=reset_at,
        )


def _reset_at(oldest: datetime | None, now: datetime, window: timedelta) -> datetime:
    """When the oldest row leaves its window; a full window if it is unknown."""
    return (oldest + window) if oldest is not None else (now + window)


def _seconds_until(reset_at: datetime, now: datetime) -> int:
    """Whole seconds until ``reset_at``, never zero or negative."""
    return max(1, int((reset_at - now).total_seconds()))


def _sha256(raw: bytes) -> str:
    """Hash the stored bytes for provenance."""
    return hashlib.sha256(raw).hexdigest()


def _safe_name(filename: str) -> str:
    """Keep only the final path component of a client-supplied name."""
    name = Path(filename or "").name.strip()
    return (name or "upload.pdf")[:200]

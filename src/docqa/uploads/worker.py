"""Parse and embed a visitor upload outside the request loop.

The request that accepts an upload only stores its bytes. This module owns the
slow half: PyMuPDF extraction in a worker thread under a hard timeout, then
chunking and embedding page by page so a poller can watch ``pages_done`` move.
Any failure deletes the partial collection and marks the upload ``failed`` with
a code the UI can explain.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import replace
from functools import partial
from pathlib import Path

import anyio
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from docqa.db import Chunk as ChunkRow
from docqa.db import Document, Upload, UploadPage
from docqa.ingest.chunking import build_chunks
from docqa.ingest.embedder import Embedder, normalise
from docqa.ingest.extract import EncryptedPdfError, ExtractedPdf, extract_pdf_bytes
from docqa.uploads.store import RUNNING_STATUSES, utcnow

LOGGER = logging.getLogger(__name__)

EMBED_BATCH = 16
DETAIL_LIMIT = 500

Extractor = Callable[[bytes], Awaitable[ExtractedPdf]]


class UploadPageLimitError(ValueError):
    """The parsed document has more pages than the configured limit."""


class UploadNoTextError(ValueError):
    """No page contains selectable text, so there is nothing to cite."""


async def extract_in_thread(raw: bytes) -> ExtractedPdf:
    """Run the pinned PyMuPDF extraction without blocking the event loop.

    ``abandon_on_cancel`` matters: when the timeout fires, the await is
    cancelled but a stuck C call cannot take the process with it.
    """
    return await anyio.to_thread.run_sync(
        partial(extract_pdf_bytes, raw, "<upload>"), abandon_on_cancel=True
    )


async def process_upload(
    upload_id: uuid.UUID,
    *,
    factory: async_sessionmaker[AsyncSession],
    embedder: Embedder,
    max_pages: int,
    timeout_s: float,
    extractor: Extractor = extract_in_thread,
) -> None:
    """Take one queued upload through parsing and embedding to ready or failed."""
    async with factory() as session, session.begin():
        upload = await session.get(Upload, upload_id)
        if upload is None or upload.status not in RUNNING_STATUSES:
            return
        raw = bytes(upload.data)
        title = _title(upload.filename)
        upload.status = "parsing"
        upload.started_at = utcnow()

    parse_started = time.perf_counter()
    try:
        async with asyncio.timeout(timeout_s):
            extracted = await extractor(raw)
            if extracted.page_count > max_pages:
                raise UploadPageLimitError(
                    f"The PDF has {extracted.page_count} pages; the limit is "
                    f"{max_pages}."
                )
            if not any(page.text.strip() for page in extracted.pages):
                raise UploadNoTextError(
                    "No selectable text was found in this PDF; it may be a scan."
                )
        parse_ms = int((time.perf_counter() - parse_started) * 1000)
        embed_started = time.perf_counter()
        await _store_pages(factory, upload_id, extracted, embedder, title)
        await _embed_pages(factory, upload_id, extracted, embedder, title)
        async with factory() as session, session.begin():
            upload = await session.get(Upload, upload_id)
            if upload is None:
                return
            upload.status = "ready"
            upload.parse_ms = parse_ms
            upload.embed_ms = int((time.perf_counter() - embed_started) * 1000)
            upload.finished_at = utcnow()
            upload.error_code = None
            upload.error_detail = None
    except TimeoutError:
        await _fail(
            factory,
            upload_id,
            "timeout",
            f"Parsing took longer than {timeout_s:g} s; try a smaller file.",
        )
    except EncryptedPdfError:
        await _fail(
            factory,
            upload_id,
            "encrypted",
            "The PDF is password-protected, so its text cannot be read.",
        )
    except UploadPageLimitError as exc:
        await _fail(factory, upload_id, "page_limit", str(exc))
    except UploadNoTextError as exc:
        await _fail(factory, upload_id, "no_text", str(exc))
    except Exception as exc:
        LOGGER.exception("upload %s failed", upload_id)
        await _fail(factory, upload_id, "failed", f"{type(exc).__name__}: {exc}")


def _title(filename: str) -> str:
    """Return a human title for citations: the file's stem, or its name."""
    stem = Path(filename).stem.strip()
    return stem or filename or "Uploaded document"


async def _store_pages(
    factory: async_sessionmaker[AsyncSession],
    upload_id: uuid.UUID,
    extracted: ExtractedPdf,
    embedder: Embedder,
    title: str,
) -> None:
    """Create the document and persist every page's exact text in one commit."""
    async with factory() as session, session.begin():
        upload = await session.get(Upload, upload_id)
        if upload is None:
            return
        document = Document(
            doc_id=f"upload:{upload.collection_id}",
            title=title,
            filename=upload.filename,
            source="visitor upload",
            source_url="",
            licence="visitor-supplied",
            licence_url="",
            retrieved_at=upload.created_at.date(),
            sha256=upload.sha256,
            byte_size=upload.byte_size,
            page_count=extracted.page_count,
            embedder=embedder.name,
            collection_id=upload.collection_id,
        )
        session.add(document)
        await session.flush()
        session.add_all(
            [
                UploadPage(
                    upload_id=upload_id,
                    page_no=page.page_no,
                    text=page.text,
                    page_text_sha256=page.sha256,
                )
                for page in extracted.pages
            ]
        )
        upload.document_id = document.id
        upload.pages_total = extracted.page_count
        upload.pages_done = 0
        upload.status = "embedding"


async def _embed_pages(
    factory: async_sessionmaker[AsyncSession],
    upload_id: uuid.UUID,
    extracted: ExtractedPdf,
    embedder: Embedder,
    title: str,
) -> None:
    """Chunk and embed one page at a time, committing progress after each."""
    offset = 0
    for page in extracted.pages:
        if page.text.strip():
            chunks = await anyio.to_thread.run_sync(
                partial(
                    build_chunks,
                    replace(extracted, pages=(page,), page_count=1),
                    embedder.counter,
                    title,
                )
            )
            vectors = await _embed_chunks(embedder, [chunk.text for chunk in chunks])
            async with factory() as session, session.begin():
                upload = await session.get(Upload, upload_id)
                if upload is None or upload.status == "failed":
                    return
                session.add_all(
                    [
                        ChunkRow(
                            document_id=upload.document_id,
                            chunk_index=offset + chunk.chunk_index,
                            page_no=chunk.page_no,
                            char_start=chunk.char_start,
                            char_end=chunk.char_end,
                            page_text_sha256=chunk.page_text_sha256,
                            breadcrumb=chunk.breadcrumb,
                            text=chunk.text,
                            token_count=chunk.token_count,
                            collection_id=upload.collection_id,
                            embedding=vector,
                        )
                        for chunk, vector in zip(chunks, vectors, strict=True)
                    ]
                )
                upload.pages_done = page.page_no
            offset += len(chunks)
        else:
            async with factory() as session, session.begin():
                upload = await session.get(Upload, upload_id)
                if upload is None or upload.status == "failed":
                    return
                upload.pages_done = page.page_no


async def _embed_chunks(embedder: Embedder, texts: list[str]) -> list[list[float]]:
    """Embed in bounded batches and L2-normalise every vector."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH):
        batch = texts[start : start + EMBED_BATCH]
        embedded = await embedder.embed(batch)
        if len(embedded) != len(batch):
            raise RuntimeError(
                f"embedder returned {len(embedded)} vectors for {len(batch)} chunks"
            )
        for vector in embedded:
            unit, _ = normalise(vector)
            vectors.append(unit)
    return vectors


async def _fail(
    factory: async_sessionmaker[AsyncSession],
    upload_id: uuid.UUID,
    code: str,
    detail: str,
) -> None:
    """Delete a failed upload's partial collection, then record why it failed."""
    async with factory() as session, session.begin():
        upload = await session.get(Upload, upload_id)
        if upload is None:
            return
        await session.execute(
            delete(ChunkRow).where(ChunkRow.collection_id == upload.collection_id)
        )
        await session.execute(
            delete(Document).where(Document.collection_id == upload.collection_id)
        )
        await session.execute(
            delete(UploadPage).where(UploadPage.upload_id == upload_id)
        )
        upload.document_id = None
        upload.status = "failed"
        upload.error_code = code
        upload.error_detail = detail[:DETAIL_LIMIT]
        upload.finished_at = utcnow()

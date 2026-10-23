"""End-to-end ingest: verify the lock, extract, chunk, embed, upsert."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import anyio
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from docqa.config import Settings
from docqa.db import Chunk as ChunkRow
from docqa.db import Document, make_engine, make_session_factory
from docqa.ingest.chunking import Chunk, build_chunks
from docqa.ingest.embedder import EmbedderChoice, normalise
from docqa.ingest.extract import ExtractedPdf, extract_pdf
from docqa.ingest.lock import LockEntry, load_lock, verify_file


@dataclass(frozen=True)
class FileStats:
    """Per-file ingest report."""

    path: Path
    doc_id: str
    title: str
    pages: int
    chunks: int
    p50_tokens: int
    p95_tokens: int
    max_tokens: int
    clipped_vectors: int
    updated: bool


@dataclass(frozen=True)
class IngestResult:
    """Result for a whole ingest run."""

    files: list[FileStats]
    embedder_name: str
    dummy_embeddings: bool


def percentile(values: Sequence[int], fraction: float) -> int:
    """Nearest-rank percentile (fraction in [0, 1])."""
    if not values:
        return 0
    ordered = sorted(values)
    rank = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[rank]


def _prepare(
    paths: Sequence[Path], settings: Settings
) -> list[tuple[Path, LockEntry]]:
    """Verify every input against the lock before touching the database."""
    lock = load_lock(settings.corpus_lock)
    prepared = [(path, verify_file(path, lock)) for path in paths]
    seen_sha: set[str] = set()
    for _, entry in prepared:
        if entry.sha256 in seen_sha:
            raise ValueError(f"duplicate input file for doc_id {entry.doc_id}")
        seen_sha.add(entry.sha256)
    return prepared


async def ingest_files(
    paths: Sequence[Path],
    settings: Settings,
    choice: EmbedderChoice,
    *,
    batch_size: int = 32,
    embedder_name: str | None = None,
) -> IngestResult:
    """Ingest PDFs idempotently, one transaction per document."""
    prepared = _prepare(paths, settings)
    extracted: list[tuple[Path, LockEntry, ExtractedPdf]] = []
    for path, entry in prepared:
        parsed = await anyio.to_thread.run_sync(extract_pdf, path)
        if parsed.page_count != entry.pages:
            raise ValueError(
                f"{path.name}: extracted {parsed.page_count} pages, lock pins {entry.pages}"
            )
        extracted.append((path, entry, parsed))

    engine = make_engine(settings)
    factory = make_session_factory(engine)
    stored_embedder = embedder_name or choice.embedder.name
    reports: list[FileStats] = []
    try:
        for path, entry, parsed in extracted:
            chunks = await anyio.to_thread.run_sync(
                partial(
                    build_chunks,
                    parsed,
                    choice.embedder.counter,
                    entry.title,
                )
            )
            vectors: list[list[float]] = []
            clipped = 0
            for offset in range(0, len(chunks), batch_size):
                batch = chunks[offset : offset + batch_size]
                embedded = await choice.embedder.embed([chunk.text for chunk in batch])
                if len(embedded) != len(batch):
                    raise RuntimeError(
                        f"embedder returned {len(embedded)} vectors for {len(batch)} chunks"
                    )
                for vector in embedded:
                    unit, was_clipped = normalise(vector)
                    vectors.append(unit)
                    clipped += int(was_clipped)

            async with factory() as session, session.begin():
                updated = await _upsert_document(
                    session,
                    entry,
                    parsed,
                    chunks,
                    vectors,
                    stored_embedder,
                )
            token_counts = [chunk.token_count for chunk in chunks]
            reports.append(
                FileStats(
                    path=path,
                    doc_id=entry.doc_id,
                    title=entry.title,
                    pages=parsed.page_count,
                    chunks=len(chunks),
                    p50_tokens=percentile(token_counts, 0.50),
                    p95_tokens=percentile(token_counts, 0.95),
                    max_tokens=max(token_counts, default=0),
                    clipped_vectors=clipped,
                    updated=updated,
                )
            )
    finally:
        await engine.dispose()
    return IngestResult(
        files=reports,
        embedder_name=stored_embedder,
        dummy_embeddings=choice.embedder.is_dummy,
    )


async def _upsert_document(
    session: AsyncSession,
    entry: LockEntry,
    parsed: ExtractedPdf,
    chunks: Sequence[Chunk],
    vectors: Sequence[Sequence[float]],
    embedder_name: str,
) -> bool:
    """Replace a document's chunks atomically; returns True when it already existed."""
    existing = await session.scalar(
        select(Document).where(Document.sha256 == entry.sha256)
    )
    if existing is None:
        existing = await session.scalar(
            select(Document).where(Document.doc_id == entry.doc_id)
        )
    updated = existing is not None
    if existing is None:
        document = Document(
            doc_id=entry.doc_id,
            title=entry.title,
            filename=entry.filename,
            source=entry.source,
            source_url=entry.url,
            licence=entry.licence,
            licence_url=entry.licence_url,
            retrieved_at=entry.retrieved_at,
            sha256=entry.sha256,
            byte_size=parsed.byte_size,
            page_count=parsed.page_count,
            embedder=embedder_name,
        )
        session.add(document)
        await session.flush()
    else:
        document = existing
        document.title = entry.title
        document.filename = entry.filename
        document.source = entry.source
        document.source_url = entry.url
        document.licence = entry.licence
        document.licence_url = entry.licence_url
        document.retrieved_at = entry.retrieved_at
        document.sha256 = entry.sha256
        document.byte_size = parsed.byte_size
        document.page_count = parsed.page_count
        document.embedder = embedder_name
        await session.execute(
            delete(ChunkRow).where(ChunkRow.document_id == document.id)
        )
        await session.flush()

    session.add_all(
        [
            ChunkRow(
                document_id=document.id,
                chunk_index=chunk.chunk_index,
                page_no=chunk.page_no,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                page_text_sha256=chunk.page_text_sha256,
                breadcrumb=chunk.breadcrumb,
                text=chunk.text,
                token_count=chunk.token_count,
                embedding=vector,
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
    )
    return updated

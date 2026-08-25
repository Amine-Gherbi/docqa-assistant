"""Integration test: a second ingest replaces chunks instead of duplicating them.

Uses a throwaway lock plus five generated PDFs so the real corpus and its
embedding vectors are never touched. Skipped when PostgreSQL is unreachable.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
from pathlib import Path

import pymupdf
import pytest
from sqlalchemy import delete, func, select, text

from docqa.config import get_settings
from docqa.db import Chunk as ChunkRow
from docqa.db import Document, make_engine, make_session_factory, upgrade_to_head
from docqa.ingest.embedder import EmbedderChoice, HashEmbedder
from docqa.ingest.extract import extract_pdf
from docqa.ingest.pipeline import ingest_files

SENTENCE = (
    "Dense retrieval maps questions and passages into one embedding space so "
    "that a retriever can rank evidence before a generator writes an answer. "
)


def _make_pdf(path: Path, marker: str) -> None:
    document = pymupdf.open()
    for page_number in range(1, 3):
        page = document.new_page()
        body = f"{marker} page {page_number}. " + " ".join([SENTENCE] * 40)
        page.insert_textbox(
            pymupdf.Rect(50, 50, 545, 790),
            body,
            fontsize=9,
            fontname="helv",
        )
    document.save(path)
    document.close()


def _make_lock(directory: Path) -> Path:
    files = []
    for index in range(5):
        path = directory / f"synthetic-{index}.pdf"
        _make_pdf(path, marker=f"synthetic paper {index}")
        raw = path.read_bytes()
        extracted = extract_pdf(path)
        files.append(
            {
                "doc_id": f"synthetic-{index}",
                "filename": path.name,
                "title": f"Synthetic Test Paper {index}",
                "source": "self-generated",
                "url": f"file://synthetic-{index}.pdf",
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
                "pages": extracted.page_count,
                "licence": "self-generated",
                "licence_url": "https://example.invalid/self-generated",
                "retrieved_at": "2026-10-23",
            }
        )
    lock_path = directory / "corpus.lock.json"
    lock_path.write_text(
        json.dumps(
            {
                "version": 1,
                "corpus": "synthetic-test",
                "generated_at": "2026-10-23",
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    return lock_path


async def test_reingest_same_file_is_idempotent(tmp_path: Path) -> None:
    lock_path = _make_lock(tmp_path)
    settings = dataclasses.replace(
        get_settings(),
        corpus_lock=lock_path,
        corpus_dir=tmp_path,
        embedder_mode="hash",
    )
    try:
        await asyncio.to_thread(upgrade_to_head, settings.database_url)
    except Exception as exc:  # pragma: no cover - depends on the machine
        pytest.skip(f"PostgreSQL is unavailable: {exc}")

    choice = EmbedderChoice(HashEmbedder(), "hash", None)
    pdf = tmp_path / "synthetic-0.pdf"
    engine = make_engine(settings)
    factory = make_session_factory(engine)
    document_pk: int | None = None
    try:
        async with factory() as session, session.begin():
            await session.execute(
                delete(Document).where(Document.doc_id.like("synthetic-%"))
            )
        first = await ingest_files([pdf], settings, choice)
        second = await ingest_files([pdf], settings, choice)

        assert first.files[0].updated is False
        assert second.files[0].updated is True
        assert first.files[0].chunks == second.files[0].chunks > 0

        async with factory() as session:
            document_pk = await session.scalar(
                select(Document.id).where(Document.doc_id == "synthetic-0")
            )
            assert document_pk is not None
            chunk_count = await session.scalar(
                select(func.count())
                .select_from(ChunkRow)
                .where(ChunkRow.document_id == document_pk)
            )
            distinct_slots = await session.scalar(
                select(func.count(func.distinct(ChunkRow.chunk_index))).where(
                    ChunkRow.document_id == document_pk
                )
            )
            duplicates = (
                await session.execute(
                    select(ChunkRow.chunk_index, func.count())
                    .where(ChunkRow.document_id == document_pk)
                    .group_by(ChunkRow.chunk_index)
                    .having(func.count() > 1)
                )
            ).all()
            max_tokens = await session.scalar(
                select(func.max(ChunkRow.token_count)).where(
                    ChunkRow.document_id == document_pk
                )
            )
            correct_dims = await session.scalar(
                text(
                    "SELECT count(*) FROM chunks "
                    "WHERE document_id = :doc AND vector_dims(embedding) = 384"
                ),
                {"doc": document_pk},
            )

        assert chunk_count == first.files[0].chunks
        assert distinct_slots == chunk_count
        assert duplicates == []
        assert max_tokens is not None and max_tokens <= 448
        assert correct_dims == chunk_count
    finally:
        if document_pk is not None:
            async with factory() as session, session.begin():
                await session.execute(
                    delete(Document).where(Document.id == document_pk)
                )
        await engine.dispose()

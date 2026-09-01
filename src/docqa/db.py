"""Database schema and connection helpers.

Two tables matter for R1:

* ``documents`` — one row per pinned PDF, keyed by SHA-256.
* ``chunks`` — retrievable units with page/character provenance, a 384-d embedding,
  and a generated ``tsvector`` for the lexical half of hybrid retrieval.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Computed,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import TSVECTOR, UUID
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from docqa.config import EMBEDDING_DIM, PROJECT_ROOT, Settings

MAX_CHUNK_TOKENS = 448

# One API process answers one question at a time on the free instance. The
# default pool (5 + 10 overflow) keeps up to 15 asyncpg connection buffers for
# no benefit; bound it and recycle old connections so a proxied host connection
# cannot pin a closed socket.
POOL_SIZE = 2
MAX_OVERFLOW = 2
POOL_RECYCLE_S = 300


class Base(DeclarativeBase):
    """Declarative base for every docqa table."""


class Document(Base):
    """A PDF pinned by SHA-256 in ``corpus.lock.json``."""

    __tablename__ = "documents"
    __table_args__ = (
        Index(
            "uq_documents_collection_sha256",
            "collection_id",
            "sha256",
            unique=True,
        ),
        Index("ix_documents_collection_id", "collection_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    doc_id: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    licence: Mapped[str] = mapped_column(Text, nullable=False)
    licence_url: Mapped[str] = mapped_column(Text, nullable=False)
    retrieved_at: Mapped[date] = mapped_column(Date, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    byte_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    page_count: Mapped[int] = mapped_column(Integer, nullable=False)
    embedder: Mapped[str] = mapped_column(Text, nullable=False)
    collection_id: Mapped[str] = mapped_column(
        Text, nullable=False, server_default="demo"
    )
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Chunk(Base):
    """A contiguous span of one page, prefixed with its breadcrumb."""

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_chunks_doc_index"),
        CheckConstraint("char_start <= char_end", name="ck_chunks_span_order"),
        CheckConstraint(
            f"token_count BETWEEN 1 AND {MAX_CHUNK_TOKENS}", name="ck_chunks_token_cap"
        ),
        Index("ix_chunks_document_id", "document_id"),
        Index("ix_chunks_page_no", "document_id", "page_no"),
        Index("ix_chunks_collection_id", "collection_id"),
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_chunks_tsv_gin", "tsv", postgresql_using="gin"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    document_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    page_no: Mapped[int] = mapped_column(Integer, nullable=False)
    char_start: Mapped[int] = mapped_column(Integer, nullable=False)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False)
    page_text_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    breadcrumb: Mapped[str] = mapped_column(Text, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    collection_id: Mapped[str] = mapped_column(
        Text, nullable=False, server_default="demo"
    )
    embedding: Mapped[Any] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
    tsv: Mapped[Any] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', coalesce(text, ''))", persisted=True),
        nullable=True,
    )

    document: Mapped[Document] = relationship(back_populates="chunks")


class Upload(Base):
    """A visitor PDF held in Postgres until its 24-hour TTL expires."""

    __tablename__ = "uploads"
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued','parsing','embedding','ready','failed')",
            name="ck_uploads_status",
        ),
        CheckConstraint("byte_size > 0", name="ck_uploads_size"),
        CheckConstraint("pages_done >= 0", name="ck_uploads_pages_done"),
        Index("ix_uploads_client_created", "client_id", "created_at"),
        Index("ix_uploads_expires_at", "expires_at"),
        Index("ix_uploads_status", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    client_id: Mapped[str] = mapped_column(Text, nullable=False)
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    content_type: Mapped[str] = mapped_column(Text, nullable=False)
    byte_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    collection_id: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    document_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("documents.id", ondelete="SET NULL"), nullable=True
    )
    pages_total: Mapped[int | None] = mapped_column(Integer, nullable=True)
    pages_done: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    parse_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    embed_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_code: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    data: Mapped[bytes] = mapped_column("bytea", LargeBinary, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    pages: Mapped[list[UploadPage]] = relationship(
        back_populates="upload",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class UploadPage(Base):
    """One page's exact PyMuPDF text, owned by an upload."""

    __tablename__ = "upload_pages"
    __table_args__ = (
        UniqueConstraint("upload_id", "page_no", name="uq_upload_pages_upload_page"),
        CheckConstraint("page_no >= 1", name="ck_upload_pages_page_no"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    upload_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("uploads.id", ondelete="CASCADE"),
        nullable=False,
    )
    page_no: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    page_text_sha256: Mapped[str] = mapped_column(String(64), nullable=False)

    upload: Mapped[Upload] = relationship(back_populates="pages")


def make_engine(settings: Settings) -> AsyncEngine:
    """Create a bounded async engine for the configured database."""
    return create_async_engine(
        settings.database_url,
        pool_pre_ping=True,
        pool_size=POOL_SIZE,
        max_overflow=MAX_OVERFLOW,
        pool_recycle=POOL_RECYCLE_S,
    )


def make_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Create a session factory that does not expire objects after commit."""
    return async_sessionmaker(engine, expire_on_commit=False)


@asynccontextmanager
async def session_scope(
    factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """Yield a session inside a transaction, rolling back on error."""
    async with factory() as session, session.begin():
        yield session


def alembic_config(database_url: str) -> Config:
    """Build an Alembic config pointed at this project's migrations."""
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def upgrade_to_head(database_url: str) -> None:
    """Apply migrations. Synchronous; run before entering the event loop."""
    command.upgrade(alembic_config(database_url), "head")


def downgrade_to_base(database_url: str) -> None:
    """Drop everything the migrations created. Used by tests and fresh starts."""
    command.downgrade(alembic_config(database_url), "base")


def project_path(relative: str) -> Path:
    """Resolve a path relative to the repository root."""
    return PROJECT_ROOT / relative

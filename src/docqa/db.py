"""Database schema and connection helpers.

Two tables matter for R1:

* ``documents`` — one row per pinned PDF, keyed by SHA-256.
* ``chunks`` — retrievable units with page/character provenance, a 384-d embedding,
  and a generated ``tsvector`` for the lexical half of hybrid retrieval.
"""

from __future__ import annotations

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
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from docqa.config import EMBEDDING_DIM, PROJECT_ROOT, Settings

MAX_CHUNK_TOKENS = 448


class Base(DeclarativeBase):
    """Declarative base for every docqa table."""


class Document(Base):
    """A PDF pinned by SHA-256 in ``corpus.lock.json``."""

    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("sha256", name="uq_documents_sha256"),)

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
    embedding: Mapped[Any] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
    tsv: Mapped[Any] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', coalesce(text, ''))", persisted=True),
        nullable=True,
    )

    document: Mapped[Document] = relationship(back_populates="chunks")


def make_engine(settings: Settings) -> AsyncEngine:
    """Create an async engine for the configured database."""
    return create_async_engine(settings.database_url, pool_pre_ping=True)


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

"""Initial documents and chunks schema.

Revision ID: 0001_initial
Revises:
Create Date: 2026-10-23
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EMBEDDING_DIM = 384


def upgrade() -> None:
    """Create the pgvector extension, document table, and chunk table."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "documents",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("doc_id", sa.Text(), nullable=False, unique=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("licence", sa.Text(), nullable=False),
        sa.Column("licence_url", sa.Text(), nullable=False),
        sa.Column("retrieved_at", sa.Date(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("byte_size", sa.BigInteger(), nullable=False),
        sa.Column("page_count", sa.Integer(), nullable=False),
        sa.Column("embedder", sa.Text(), nullable=False),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("sha256", name="uq_documents_sha256"),
    )
    op.create_table(
        "chunks",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column(
            "document_id",
            sa.BigInteger(),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("page_no", sa.Integer(), nullable=False),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.Column("page_text_sha256", sa.String(length=64), nullable=False),
        sa.Column("breadcrumb", sa.Text(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
        sa.Column(
            "tsv",
            postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('english', coalesce(text, ''))", persisted=True),
            nullable=True,
        ),
        sa.UniqueConstraint("document_id", "chunk_index", name="uq_chunks_doc_index"),
        sa.CheckConstraint("char_start <= char_end", name="ck_chunks_span_order"),
        sa.CheckConstraint("token_count BETWEEN 1 AND 448", name="ck_chunks_token_cap"),
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])
    op.create_index("ix_chunks_page_no", "chunks", ["document_id", "page_no"])
    op.create_index(
        "ix_chunks_embedding_hnsw",
        "chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.create_index("ix_chunks_tsv_gin", "chunks", ["tsv"], postgresql_using="gin")


def downgrade() -> None:
    """Drop the tables (the vector extension may be shared)."""
    op.drop_index("ix_chunks_tsv_gin", table_name="chunks")
    op.drop_index("ix_chunks_embedding_hnsw", table_name="chunks")
    op.drop_index("ix_chunks_page_no", table_name="chunks")
    op.drop_index("ix_chunks_document_id", table_name="chunks")
    op.drop_table("chunks")
    op.drop_table("documents")

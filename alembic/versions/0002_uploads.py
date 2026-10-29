"""Visitor uploads: collections, bytes, pages, and the 24-hour TTL.

Revision ID: 0002_uploads
Revises: 0001_initial
Create Date: 2026-10-29
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002_uploads"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add collection ownership and the two upload tables."""
    op.add_column(
        "documents",
        sa.Column(
            "collection_id",
            sa.Text(),
            nullable=False,
            server_default="demo",
        ),
    )
    op.add_column(
        "chunks",
        sa.Column(
            "collection_id",
            sa.Text(),
            nullable=False,
            server_default="demo",
        ),
    )
    op.create_index("ix_documents_collection_id", "documents", ["collection_id"])
    op.create_index("ix_chunks_collection_id", "chunks", ["collection_id"])

    op.create_table(
        "uploads",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("client_id", sa.Text(), nullable=False),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("content_type", sa.Text(), nullable=False),
        sa.Column("byte_size", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("collection_id", sa.Text(), nullable=False, unique=True),
        sa.Column(
            "document_id",
            sa.BigInteger(),
            sa.ForeignKey("documents.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("pages_total", sa.Integer(), nullable=True),
        sa.Column("pages_done", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("parse_ms", sa.Integer(), nullable=True),
        sa.Column("embed_ms", sa.Integer(), nullable=True),
        sa.Column("error_code", sa.Text(), nullable=True),
        sa.Column("error_detail", sa.Text(), nullable=True),
        sa.Column("bytea", sa.LargeBinary(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status IN ('queued','parsing','embedding','ready','failed')",
            name="ck_uploads_status",
        ),
        sa.CheckConstraint("byte_size > 0", name="ck_uploads_size"),
        sa.CheckConstraint("pages_done >= 0", name="ck_uploads_pages_done"),
    )
    op.create_index("ix_uploads_client_created", "uploads", ["client_id", "created_at"])
    op.create_index("ix_uploads_expires_at", "uploads", ["expires_at"])
    op.create_index("ix_uploads_status", "uploads", ["status"])

    op.create_table(
        "upload_pages",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column(
            "upload_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("uploads.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("page_no", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("page_text_sha256", sa.String(length=64), nullable=False),
        sa.UniqueConstraint("upload_id", "page_no", name="uq_upload_pages_upload_page"),
        sa.CheckConstraint("page_no >= 1", name="ck_upload_pages_page_no"),
    )


def downgrade() -> None:
    """Remove the upload tables and collection ownership."""
    op.drop_table("upload_pages")
    op.drop_index("ix_uploads_status", table_name="uploads")
    op.drop_index("ix_uploads_expires_at", table_name="uploads")
    op.drop_index("ix_uploads_client_created", table_name="uploads")
    op.drop_table("uploads")
    op.drop_index("ix_chunks_collection_id", table_name="chunks")
    op.drop_index("ix_documents_collection_id", table_name="documents")
    op.drop_column("chunks", "collection_id")
    op.drop_column("documents", "collection_id")

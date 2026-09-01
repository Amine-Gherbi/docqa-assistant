"""Scope a document's SHA-256 to its collection.

A visitor may upload the same PDF twice, or a file whose bytes match a pinned
corpus paper. The digest is still unique inside one collection — that is what
makes demo re-ingest idempotent — but no longer across collections.

Revision ID: 0003_collection_sha
Revises: 0002_uploads
Create Date: 2026-10-29
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0003_collection_sha"
down_revision: str | None = "0002_uploads"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Replace the global digest constraint with a per-collection one."""
    op.drop_constraint("uq_documents_sha256", "documents", type_="unique")
    op.create_index(
        "uq_documents_collection_sha256",
        "documents",
        ["collection_id", "sha256"],
        unique=True,
    )


def downgrade() -> None:
    """Restore the global constraint; fails if duplicate digests already exist."""
    op.drop_index("uq_documents_collection_sha256", table_name="documents")
    op.create_unique_constraint("uq_documents_sha256", "documents", ["sha256"])

"""PDF text extraction with provenance that survives later page rendering.

The character offsets stored for a chunk index the exact string returned by
``page.get_text("text", sort=True)`` for the PyMuPDF version pinned in
``pyproject.toml`` (2.0). ``page_text_sha256`` lets a later reader prove it is
looking at the same string before trusting those offsets.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pymupdf

PINNED_PYMUPDF_VERSION = "2.0"


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase hex SHA-256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """Return the SHA-256 of a page string's UTF-8 encoding."""
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    """Hash a file in 1 MiB blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Page:
    """One page's exact extracted text and its digest."""

    page_no: int
    text: str
    sha256: str


@dataclass(frozen=True)
class ExtractedPdf:
    """A PDF's bytes, pages, and file-level digest."""

    path: Path
    sha256: str
    byte_size: int
    page_count: int
    pages: tuple[Page, ...]
    pymupdf_version: str


def extract_pdf(path: Path) -> ExtractedPdf:
    """Extract every page's text using the pinned PyMuPDF text mode."""
    raw = path.read_bytes()
    version = str(pymupdf.__version__)
    if version != PINNED_PYMUPDF_VERSION:
        raise RuntimeError(
            f"PyMuPDF {version} is not the pinned {PINNED_PYMUPDF_VERSION}; "
            "character offsets would not be reproducible"
        )
    document = pymupdf.open(path)  # type: ignore[no-untyped-call]
    try:
        if not document.is_pdf:
            raise ValueError(f"{path} is not a PDF")
        pages: list[Page] = []
        for index, page in enumerate(cast(Any, document), start=1):
            text = str(page.get_text("text", sort=True))
            pages.append(Page(page_no=index, text=text, sha256=sha256_text(text)))
        return ExtractedPdf(
            path=path,
            sha256=sha256_bytes(raw),
            byte_size=len(raw),
            page_count=document.page_count,
            pages=tuple(pages),
            pymupdf_version=version,
        )
    finally:
        document.close()  # type: ignore[no-untyped-call]

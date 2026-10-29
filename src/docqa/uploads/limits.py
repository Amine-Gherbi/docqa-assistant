"""Refuse an upload from its bytes, before PyMuPDF parses anything.

The request loop never opens the document. Size and magic bytes are checked
directly; the page count and the encryption key are read from the trailer, and
a file with no ``startxref``/``%%EOF`` is not a complete PDF. The worker repeats
every check against the real parser, so a trailer that lies still cannot store
an over-limit document.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_UPLOAD_PAGES = 80
PDF_MAGIC = b"%PDF-"
TAIL_WINDOW = 256 * 1024

_COUNT_RE = re.compile(rb"/Count\s+(\d+)")
_PAGE_OBJECT_RE = re.compile(rb"/Type\s*/Page\b")
_ENCRYPT_RE = re.compile(rb"/Encrypt\s+\d+\s+\d+\s+R")


class UploadLimitError(Exception):
    """A refused upload, with the HTTP status and machine-readable code."""

    def __init__(
        self,
        status_code: int,
        code: str,
        detail: str,
        *,
        limit: int | None = None,
    ) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.code = code
        self.detail = detail
        self.limit = limit


@dataclass(frozen=True)
class PdfInspection:
    """What the trailer told us without parsing the document."""

    page_count: int | None
    encrypted: bool
    parseable: bool


def _megabytes(value: int) -> str:
    """Render a byte limit the way the page does."""
    if value % (1024 * 1024) == 0:
        return f"{value // (1024 * 1024)} MB"
    return f"{value} bytes"


def inspect_pdf(raw: bytes) -> PdfInspection:
    """Read the page count, encryption marker, and trailer completeness."""
    tail = raw[-TAIL_WINDOW:]
    counts = [int(match) for match in _COUNT_RE.findall(tail)]
    page_count = max(counts) if counts else None
    if page_count is None:
        objects = len(_PAGE_OBJECT_RE.findall(raw))
        page_count = objects or None
    return PdfInspection(
        page_count=page_count,
        encrypted=_ENCRYPT_RE.search(raw) is not None,
        parseable=b"startxref" in tail and b"%%EOF" in tail,
    )


def check_upload_bytes(
    raw: bytes,
    *,
    max_bytes: int = MAX_UPLOAD_BYTES,
    max_pages: int = MAX_UPLOAD_PAGES,
) -> PdfInspection:
    """Return the inspection or raise :class:`UploadLimitError`.

    Size is checked first so a large body is refused before the magic scan; the
    magic scan is checked before the trailer because a non-PDF has no trailer.
    """
    if len(raw) > max_bytes:
        raise UploadLimitError(
            413,
            "too_large",
            (
                f"The file is {len(raw)} bytes; the limit is "
                f"{max_bytes} bytes ({_megabytes(max_bytes)})."
            ),
            limit=max_bytes,
        )
    if len(raw) < len(PDF_MAGIC) or not raw.startswith(PDF_MAGIC):
        raise UploadLimitError(
            415,
            "not_pdf",
            "The file does not start with %PDF-; only PDF files are accepted.",
        )

    inspection = inspect_pdf(raw)
    if inspection.encrypted:
        raise UploadLimitError(
            422,
            "encrypted",
            "The PDF is password-protected, so its text cannot be read.",
        )
    if not inspection.parseable:
        raise UploadLimitError(
            422,
            "not_parseable",
            "The bytes are not a complete PDF: no startxref/%%EOF trailer was found.",
        )
    if inspection.page_count is not None and inspection.page_count > max_pages:
        raise UploadLimitError(
            422,
            "page_limit",
            (
                f"The PDF has {inspection.page_count} pages; the limit is "
                f"{max_pages} pages."
            ),
            limit=max_pages,
        )
    return inspection

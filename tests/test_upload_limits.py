"""Upload limits are enforced on bytes and structure, before any parse."""

from __future__ import annotations

import pymupdf
import pytest

from docqa.uploads.limits import (
    MAX_UPLOAD_BYTES,
    UploadLimitError,
    check_upload_bytes,
    inspect_pdf,
)


def _pdf_bytes(pages: int = 1) -> bytes:
    document = pymupdf.open()
    for index in range(pages):
        page = document.new_page()
        page.insert_text((72, 72), f"page {index + 1}")
    raw = document.tobytes()
    document.close()
    return raw


def _error(raw: bytes, **kwargs: int) -> UploadLimitError:
    with pytest.raises(UploadLimitError) as raised:
        check_upload_bytes(raw, **kwargs)
    return raised.value


def test_accepts_a_real_pdf_and_reads_its_page_count() -> None:
    raw = _pdf_bytes(3)
    inspection = check_upload_bytes(raw)
    assert inspection.page_count == 3
    assert inspection.encrypted is False
    assert inspection.parseable is True


def test_rejects_oversized_bytes_before_the_magic_check() -> None:
    error = _error(b"not a pdf", max_bytes=4)
    assert error.status_code == 413
    assert error.code == "too_large"
    assert "25" not in error.detail  # the configured limit is reported, not a constant


def test_rejects_a_file_that_does_not_start_with_the_pdf_magic() -> None:
    error = _error(b"<html>not a pdf</html>")
    assert error.status_code == 415
    assert error.code == "not_pdf"


def test_rejects_a_trailer_page_count_above_the_limit() -> None:
    raw = b"%PDF-1.7\n1 0 obj\n<< /Type /Pages /Count 81 >>\nendobj\n%%EOF\nstartxref\n0\n"
    error = _error(raw, max_pages=80)
    assert error.status_code == 422
    assert error.code == "page_limit"
    assert "81" in error.detail
    assert "80" in error.detail


def test_rejects_encrypted_pdfs() -> None:
    raw = (
        b"%PDF-1.7\n5 0 obj\n<< /Filter /Standard >>\nendobj\n"
        b"trailer\n<< /Root 1 0 R /Encrypt 5 0 R >>\n%%EOF\nstartxref\n0\n"
    )
    error = _error(raw)
    assert error.status_code == 422
    assert error.code == "encrypted"


def test_rejects_an_inline_encryption_dictionary() -> None:
    """PyMuPDF writes `/Encrypt<<...>>` in the trailer rather than a reference."""
    raw = (
        b"%PDF-1.7\ntrailer\n<< /Root 1 0 R /Encrypt<< /Filter/Standard"
        b" /R 6 /V 5 /Length 256 >> >>\n%%EOF\nstartxref\n0\n"
    )
    error = _error(raw)
    assert error.status_code == 422
    assert error.code == "encrypted"


def test_rejects_a_pymupdf_encrypted_document() -> None:
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), "protected")
    raw = document.tobytes(
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="owner",
        user_pw="user",
    )
    document.close()
    assert inspect_pdf(raw).encrypted is True
    error = _error(raw)
    assert error.status_code == 422
    assert error.code == "encrypted"


def test_rejects_bytes_without_a_complete_trailer() -> None:
    error = _error(b"%PDF-1.7\n1 0 obj\n<< /Type /Pages /Count 1 >>\nendobj\n")
    assert error.status_code == 422
    assert error.code == "not_parseable"


def test_counts_page_objects_when_no_count_key_is_visible() -> None:
    raw = (
        b"%PDF-1.7\n1 0 obj\n<< /Type /Page >>\nendobj\n"
        b"2 0 obj\n<< /Type /Page >>\nendobj\n%%EOF\nstartxref\n0\n"
    )
    assert inspect_pdf(raw).page_count == 2


def test_default_byte_limit_is_twenty_five_mebibytes() -> None:
    assert MAX_UPLOAD_BYTES == 25 * 1024 * 1024

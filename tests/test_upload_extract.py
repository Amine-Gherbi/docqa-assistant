"""Extraction from the bytes held in Postgres, with the same provenance."""

from __future__ import annotations

import hashlib

import pymupdf

from docqa.ingest.extract import extract_pdf_bytes, sha256_text


def _pdf_bytes(pages: int, *, encryption: int | None = None) -> bytes:
    document = pymupdf.open()
    for index in range(pages):
        page = document.new_page()
        page.insert_text((72, 72), f"page {index + 1} of the visitor document")
    if encryption is None:
        raw = document.tobytes()
    else:
        raw = document.tobytes(
            encryption=encryption,
            owner_pw="owner",
            user_pw="visitor",
        )
    document.close()
    return raw


def test_extract_pdf_bytes_keeps_page_text_and_offsets() -> None:
    raw = _pdf_bytes(2)
    extracted = extract_pdf_bytes(raw, "upload")

    assert extracted.byte_size == len(raw)
    assert extracted.sha256 == hashlib.sha256(raw).hexdigest()
    assert extracted.page_count == 2
    assert [page.page_no for page in extracted.pages] == [1, 2]
    assert extracted.pages[0].text.startswith("page 1 of")
    assert extracted.pages[0].sha256 == sha256_text(extracted.pages[0].text)


def test_extract_pdf_bytes_refuses_a_password_protected_file() -> None:
    raw = _pdf_bytes(1, encryption=pymupdf.PDF_ENCRYPT_AES_256)
    try:
        extract_pdf_bytes(raw, "upload")
    except Exception as exc:
        assert type(exc).__name__ == "EncryptedPdfError"
    else:
        raise AssertionError("an encrypted PDF must not be extracted")

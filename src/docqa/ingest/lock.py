"""The corpus contract: ``corpus.lock.json`` is the only source of truth.

A PDF is eligible for ingest only when its filename appears in the lock and its
byte length and SHA-256 match the pinned values. The PDFs themselves are not
committed; ``scripts/fetch-corpus`` materialises them from the lock.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from docqa.ingest.extract import sha256_file


class CorpusLockError(RuntimeError):
    """The lock file is malformed."""


class LockMismatch(RuntimeError):
    """A file does not match its pinned size or hash."""


class LockEntry(BaseModel):
    """One pinned PDF."""

    doc_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    title: str = Field(min_length=1)
    source: str = Field(min_length=1)
    url: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(gt=0)
    pages: int = Field(gt=0)
    licence: str = Field(min_length=1)
    licence_url: str = Field(min_length=1)
    retrieved_at: date


class CorpusLock(BaseModel):
    """The full lock file."""

    version: Literal[1]
    corpus: str
    generated_at: date
    note: str = ""
    files: list[LockEntry]

    @model_validator(mode="after")
    def _check_uniqueness(self) -> CorpusLock:
        if len(self.files) < 5:
            raise ValueError("the corpus must pin at least 5 files")
        for attribute in ("filename", "doc_id", "sha256"):
            values = [getattr(entry, attribute) for entry in self.files]
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate {attribute} in corpus lock")
        return self

    def by_filename(self, filename: str) -> LockEntry | None:
        """Look up an entry by the PDF's file name."""
        return next((entry for entry in self.files if entry.filename == filename), None)


def load_lock(path: Path) -> CorpusLock:
    """Load and validate the lock file."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CorpusLockError(f"corpus lock not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CorpusLockError(f"corpus lock is not valid JSON: {path}: {exc}") from exc
    try:
        return CorpusLock.model_validate(raw)
    except ValueError as exc:
        raise CorpusLockError(f"corpus lock is invalid: {exc}") from exc


def verify_file(path: Path, lock: CorpusLock) -> LockEntry:
    """Refuse a PDF that is not pinned, or whose bytes differ from the pin."""
    entry = lock.by_filename(path.name)
    if entry is None:
        raise LockMismatch(f"{path.name} is not listed in {lock.corpus} corpus.lock.json")
    size = path.stat().st_size
    if size != entry.bytes:
        raise LockMismatch(
            f"{path.name}: size {size} != locked {entry.bytes}; refusing ingest"
        )
    digest = sha256_file(path)
    if digest != entry.sha256:
        raise LockMismatch(
            f"{path.name}: sha256 {digest} != locked {entry.sha256}; refusing ingest"
        )
    return entry

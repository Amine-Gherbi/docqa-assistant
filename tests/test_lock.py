from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from docqa.ingest.lock import (
    CorpusLock,
    CorpusLockError,
    LockMismatch,
    load_lock,
    verify_file,
)


def _entry(index: int, payload: bytes) -> dict[str, object]:
    return {
        "doc_id": f"doc-{index}",
        "filename": f"doc-{index}.pdf",
        "title": f"Document {index}",
        "source": f"test:{index}",
        "url": f"https://example.invalid/doc-{index}.pdf",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "pages": 1,
        "licence": "self-generated",
        "licence_url": "https://example.invalid/licence",
        "retrieved_at": "2026-10-23",
    }


def _write_lock(path: Path, payloads: list[bytes]) -> None:
    data = {
        "version": 1,
        "corpus": "test-corpus",
        "generated_at": "2026-10-23",
        "files": [_entry(index, payload) for index, payload in enumerate(payloads)],
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def test_load_and_verify(tmp_path: Path) -> None:
    payloads = [f"file-{index}".encode() for index in range(5)]
    lock_path = tmp_path / "corpus.lock.json"
    _write_lock(lock_path, payloads)
    target = tmp_path / "doc-0.pdf"
    target.write_bytes(payloads[0])

    lock = load_lock(lock_path)
    entry = verify_file(target, lock)
    assert entry.doc_id == "doc-0"


def test_hash_mismatch_is_refused(tmp_path: Path) -> None:
    payloads = [f"file-{index}".encode() for index in range(5)]
    lock_path = tmp_path / "corpus.lock.json"
    _write_lock(lock_path, payloads)
    target = tmp_path / "doc-0.pdf"
    tampered = bytearray(payloads[0])
    tampered[0] ^= 0xFF
    target.write_bytes(bytes(tampered))

    with pytest.raises(LockMismatch, match="sha256"):
        verify_file(target, load_lock(lock_path))


def test_size_mismatch_is_refused(tmp_path: Path) -> None:
    payloads = [f"file-{index}".encode() for index in range(5)]
    lock_path = tmp_path / "corpus.lock.json"
    _write_lock(lock_path, payloads)
    target = tmp_path / "doc-0.pdf"
    target.write_bytes(payloads[0] + b"extra")

    with pytest.raises(LockMismatch, match="size"):
        verify_file(target, load_lock(lock_path))


def test_unlisted_file_is_refused(tmp_path: Path) -> None:
    payloads = [f"file-{index}".encode() for index in range(5)]
    lock_path = tmp_path / "corpus.lock.json"
    _write_lock(lock_path, payloads)
    target = tmp_path / "unlisted.pdf"
    target.write_bytes(b"anything")

    with pytest.raises(LockMismatch, match="not listed"):
        verify_file(target, load_lock(lock_path))


def test_duplicate_entries_are_rejected(tmp_path: Path) -> None:
    payloads = [f"file-{index}".encode() for index in range(4)]
    lock_path = tmp_path / "corpus.lock.json"
    _write_lock(lock_path, payloads)
    raw = json.loads(lock_path.read_text(encoding="utf-8"))
    raw["files"].append(_entry(4, payloads[0]))
    lock_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(CorpusLockError, match="invalid"):
        load_lock(lock_path)


def test_bad_digest_is_rejected(tmp_path: Path) -> None:
    payloads = [f"file-{index}".encode() for index in range(5)]
    lock_path = tmp_path / "corpus.lock.json"
    _write_lock(lock_path, payloads)
    raw = json.loads(lock_path.read_text(encoding="utf-8"))
    raw["files"][0]["sha256"] = "not-a-digest"
    lock_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(CorpusLockError):
        load_lock(lock_path)


def test_fewer_than_five_files_is_rejected(tmp_path: Path) -> None:
    lock_path = tmp_path / "corpus.lock.json"
    _write_lock(lock_path, [b"one", b"two"])
    with pytest.raises(CorpusLockError):
        load_lock(lock_path)


def test_missing_lock_is_reported(tmp_path: Path) -> None:
    with pytest.raises(CorpusLockError, match="not found"):
        load_lock(tmp_path / "absent.json")


def test_committed_lock_is_valid() -> None:
    from docqa.config import PROJECT_ROOT

    lock = load_lock(PROJECT_ROOT / "corpus.lock.json")
    assert len(lock.files) >= 5
    assert all(len(entry.sha256) == 64 for entry in lock.files)
    assert isinstance(lock, CorpusLock)

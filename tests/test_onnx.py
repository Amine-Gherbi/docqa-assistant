"""The process-wide ONNX limiter honours the hosted concurrency bound."""

from __future__ import annotations

import pytest

import docqa.onnx as onnx


def test_default_concurrency_stays_two(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invariant 7 keeps ``Semaphore(2)`` unless a deploy lowers it."""
    monkeypatch.delenv("DOCQA_ONNX_CONCURRENCY", raising=False)
    monkeypatch.setattr(onnx, "_SEMAPHORE", None)
    assert onnx.onnx_semaphore().value == 2


def test_hosted_concurrency_can_be_lowered_to_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """The 512 MB free instance serialises ONNX work across requests."""
    monkeypatch.setenv("DOCQA_ONNX_CONCURRENCY", "1")
    monkeypatch.setattr(onnx, "_SEMAPHORE", None)
    assert onnx.onnx_semaphore().value == 1


@pytest.mark.parametrize("raw", ["0", "-1", "two"])
def test_bad_concurrency_fails_at_first_use(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    """Zero would deadlock every ONNX call; reject it explicitly."""
    monkeypatch.setenv("DOCQA_ONNX_CONCURRENCY", raw)
    monkeypatch.setattr(onnx, "_SEMAPHORE", None)
    with pytest.raises(ValueError, match="DOCQA_ONNX_CONCURRENCY"):
        onnx.onnx_semaphore()

"""The reranker's memory bounds reach fastembed unchanged."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from docqa.rerank import CrossEncoderReranker


class _FakeCrossEncoder:
    """Records construction kwargs and the batch size used for each call."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.calls: list[tuple[str, list[str], int]] = []

    def rerank(self, query: str, documents: list[str], batch_size: int = 64) -> list[float]:
        self.calls.append((query, list(documents), batch_size))
        return [float(index) for index, _ in enumerate(documents)]


def _install_fake_fastembed(monkeypatch: pytest.MonkeyPatch) -> None:
    """fastembed is installed, but these tests must not load ONNX."""
    module = types.ModuleType("fastembed.rerank.cross_encoder")
    module.TextCrossEncoder = _FakeCrossEncoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed.rerank.cross_encoder", module)


def test_scores_one_pair_at_a_time_and_disables_the_cpu_arena(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 20-passage batch measured +439 MB RSS; one pair measured +31 MB."""
    _install_fake_fastembed(monkeypatch)
    reranker = CrossEncoderReranker(
        "model", cache_dir="/cache", threads=1, batch_size=1, memory_arena=False
    )

    scores = reranker._score_sync("question", ["first", "second"])

    assert scores == [0.0, 1.0]
    model = reranker._model
    assert isinstance(model, _FakeCrossEncoder)
    assert model.kwargs == {
        "model_name": "model",
        "enable_cpu_mem_arena": False,
        "cache_dir": "/cache",
        "threads": 1,
    }
    assert [call[2] for call in model.calls] == [1]


def test_batch_size_and_arena_can_be_raised_for_a_larger_machine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bound is a host setting, not a hard-coded constant."""
    _install_fake_fastembed(monkeypatch)
    reranker = CrossEncoderReranker("model", batch_size=4, memory_arena=True)

    reranker._score_sync("question", ["first", "second"])

    model = reranker._model
    assert isinstance(model, _FakeCrossEncoder)
    assert model.kwargs["enable_cpu_mem_arena"] is True
    assert model.calls[0][2] == 4


def test_rejects_a_batch_size_below_one() -> None:
    """A zero batch would silently score nothing."""
    with pytest.raises(ValueError, match="batch size"):
        CrossEncoderReranker("model", batch_size=0)


def test_non_finite_scores_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A NaN must not reach the calibration or the answer."""
    _install_fake_fastembed(monkeypatch)

    class _NaNEncoder(_FakeCrossEncoder):
        def rerank(self, query: str, documents: list[str], batch_size: int = 64) -> list[float]:
            return [float("nan") for _ in documents]

    module = sys.modules["fastembed.rerank.cross_encoder"]
    module.TextCrossEncoder = _NaNEncoder  # type: ignore[attr-defined]
    reranker = CrossEncoderReranker("model")

    with pytest.raises(ValueError, match="non-finite"):
        reranker._score_sync("question", ["first"])

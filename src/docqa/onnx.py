"""Process-wide limiter for ONNX inference (invariant 7).

Embedding and reranking both run ONNX graphs on CPU. They share one
``Semaphore(2)`` so the process never runs more than two inference calls at
once, regardless of which path issued them.
"""

from __future__ import annotations

import anyio

_SEMAPHORE: anyio.Semaphore | None = None


def onnx_semaphore() -> anyio.Semaphore:
    """Return the process-wide ``Semaphore(2)`` guarding ONNX work."""
    global _SEMAPHORE
    if _SEMAPHORE is None:
        _SEMAPHORE = anyio.Semaphore(2)
    return _SEMAPHORE

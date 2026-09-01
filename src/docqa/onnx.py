"""Process-wide limiter for ONNX inference (invariant 7).

Embedding and reranking both run ONNX graphs on CPU. They share one
``Semaphore(2)`` by default so the process never runs more than two inference
calls at once, regardless of which path issued them.

``DOCQA_ONNX_CONCURRENCY`` lowers that bound. The hosted free instance is a
single 512 MB container: two concurrent cross-encoder calls were measured at
+69 MB over one (461 MB with the models resident) and pushed the whole process
over the cgroup limit, so it runs with one.
"""

from __future__ import annotations

import os

import anyio

_SEMAPHORE: anyio.Semaphore | None = None
DEFAULT_CONCURRENCY = 2


def _concurrency_from_environment() -> int:
    """Read ``DOCQA_ONNX_CONCURRENCY``; unset keeps the documented two."""
    raw = os.environ.get("DOCQA_ONNX_CONCURRENCY", str(DEFAULT_CONCURRENCY)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("DOCQA_ONNX_CONCURRENCY must be an integer") from exc
    if value < 1:
        raise ValueError("DOCQA_ONNX_CONCURRENCY must be at least 1")
    return value


def onnx_semaphore() -> anyio.Semaphore:
    """Return the process-wide ONNX limiter (``Semaphore(2)`` by default)."""
    global _SEMAPHORE
    if _SEMAPHORE is None:
        _SEMAPHORE = anyio.Semaphore(_concurrency_from_environment())
    return _SEMAPHORE

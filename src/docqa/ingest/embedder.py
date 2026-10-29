"""Embedding backends behind one async interface.

The real backend is fastembed's bge-small-en-v1.5 (384-d, ONNX, CPU). If the
weights cannot be downloaded the pipeline can fall back to a deterministic hash
embedder; that fallback is never silent — it is named in the CLI banner, stored on
every document row, and recorded in ``docs/ENV.md``.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast, runtime_checkable

import anyio

from docqa.config import EMBEDDING_DIM, Settings
from docqa.ingest.tokens import HashTokenCounter, ModelTokenCounter, TokenCounter
from docqa.onnx import onnx_semaphore

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _embedding_semaphore() -> anyio.Semaphore:
    """Return the shared ONNX concurrency limiter."""
    return onnx_semaphore()


@runtime_checkable
class Embedder(Protocol):
    """Minimal interface shared by the real and dummy embedders."""

    name: str
    is_dummy: bool
    dim: int
    counter: TokenCounter

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of passages, preserving order."""
        ...

    async def embed_query(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of queries, preserving order."""
        ...


class FastEmbedEmbedder:
    """bge-small-en-v1.5 through fastembed's ONNX runtime."""

    def __init__(
        self, model_name: str, cache_dir: str | None = None, threads: int | None = None
    ) -> None:
        """Load the ONNX model and bind its tokenizer for counting."""
        from fastembed import TextEmbedding

        self.name = f"fastembed:{model_name}"
        self.is_dummy = False
        self.dim = EMBEDDING_DIM
        kwargs: dict[str, Any] = {"model_name": model_name}
        if cache_dir is not None:
            kwargs["cache_dir"] = cache_dir
        if threads is not None:
            kwargs["threads"] = threads
        self._model = TextEmbedding(**kwargs)
        inner = cast(Any, self._model).model
        self.counter: TokenCounter = ModelTokenCounter(inner.tokenizer)

    def _collect(self, vectors: Iterable[Iterable[float]]) -> list[list[float]]:
        collected: list[list[float]] = []
        for vector in vectors:
            values = [float(value) for value in vector]
            if len(values) != self.dim:
                raise ValueError(f"expected {self.dim}-d embedding, got {len(values)}")
            collected.append(values)
        return collected

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        return self._collect(self._model.embed(texts))

    def _embed_query_sync(self, texts: list[str]) -> list[list[float]]:
        return self._collect(self._model.query_embed(texts))

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Run ONNX inference in a worker thread, at most two batches at once."""
        if not texts:
            return []
        async with _embedding_semaphore():
            return await anyio.to_thread.run_sync(self._embed_sync, list(texts))

    async def embed_query(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed queries through the model's query path (bge adds its prefix)."""
        if not texts:
            return []
        async with _embedding_semaphore():
            return await anyio.to_thread.run_sync(self._embed_query_sync, list(texts))


class HashEmbedder:
    """Deterministic, semantics-free feature hashing.

    This exists only so ingest still works when model weights are unreachable.
    It is labelled ``dummy`` and must never be presented as a semantic result.
    """

    is_dummy = True
    dim = EMBEDDING_DIM

    def __init__(self) -> None:
        """Bind the fallback name and its deterministic counter."""
        self.name = "dummy:blake2b-hash-384"
        self.counter: TokenCounter = HashTokenCounter()

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = _TOKEN_RE.findall(text.lower())
        features = tokens + [f"{a}_{b}" for a, b in itertools.pairwise(tokens)]
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            vector[0] = 1.0
            return vector
        return [value / norm for value in vector]

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Hash embeddings run in a thread like the real path, for one interface."""
        if not texts:
            return []
        async with _embedding_semaphore():
            return await anyio.to_thread.run_sync(self._embed_sync, list(texts))

    async def embed_query(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed queries; the hash fallback has no query/document asymmetry."""
        return await self.embed(texts)


@dataclass(frozen=True)
class EmbedderChoice:
    """The embedder actually selected, plus why."""

    embedder: Embedder
    requested: str
    fallback_reason: str | None

    @property
    def banner(self) -> str:
        """Human-readable line printed before ingest."""
        if self.embedder.is_dummy:
            return (
                "WARNING: dummy embeddings (deterministic hash embedder) — "
                "vectors are NOT semantic; retrieval quality is not meaningful."
            )
        return f"Embeddings: {self.embedder.name} ({self.embedder.dim}-d, ONNX CPU)"


def build_embedder(settings: Settings) -> EmbedderChoice:
    """Select the configured embedding backend, falling back only in ``auto``."""
    mode = settings.embedder_mode
    if mode == "hash":
        return EmbedderChoice(HashEmbedder(), requested=mode, fallback_reason=None)
    try:
        fastembedder = FastEmbedEmbedder(
            settings.embedding_model,
            str(settings.hf_cache_dir) if settings.hf_cache_dir else None,
            threads=settings.onnx_threads,
        )
    except Exception as exc:  # Download and ONNX failures are heterogeneous.
        if mode == "fastembed":
            raise
        return EmbedderChoice(
            HashEmbedder(),
            requested=mode,
            fallback_reason=f"{type(exc).__name__}: {exc}",
        )
    return EmbedderChoice(fastembedder, requested=mode, fallback_reason=None)


def normalise(vector: Sequence[float]) -> tuple[list[float], bool]:
    """L2-normalise a vector; returns (unit_vector, was_clipped)."""
    values = [float(value) for value in vector]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("embedding contained a non-finite value")
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        raise ValueError("embedding had zero norm")
    clipped = abs(norm - 1.0) > 1e-3
    if not clipped:
        return values, False
    return [value / norm for value in values], True

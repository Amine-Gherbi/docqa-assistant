from __future__ import annotations

import math

import pytest

from docqa.ingest.embedder import HashEmbedder, normalise


def test_hash_embedder_is_deterministic_and_normalised() -> None:
    embedder = HashEmbedder()
    first = embedder._vector("retrieval augmented generation")
    second = embedder._vector("retrieval augmented generation")
    assert first == second
    assert len(first) == 384
    assert math.isclose(sum(value * value for value in first), 1.0, rel_tol=1e-9)


def test_hash_embedder_distinguishes_documents() -> None:
    embedder = HashEmbedder()
    a = embedder._vector("dense passage retrieval")
    b = embedder._vector("sentence embeddings with siamese networks")
    assert a != b


def test_hash_embedder_is_labelled_dummy() -> None:
    embedder = HashEmbedder()
    assert embedder.is_dummy
    assert embedder.name.startswith("dummy:")


def test_normalise_marks_unnormalised_vectors_as_clipped() -> None:
    unit, clipped = normalise([3.0, 4.0] + [0.0] * 382)
    assert clipped
    assert math.isclose(unit[0], 0.6)
    assert math.isclose(unit[1], 0.8)


def test_normalise_leaves_unit_vectors_alone() -> None:
    unit, clipped = normalise([0.0, 1.0] + [0.0] * 382)
    assert not clipped
    assert unit[1] == 1.0


def test_normalise_rejects_zero_and_non_finite() -> None:
    with pytest.raises(ValueError, match="zero norm"):
        normalise([0.0] * 384)
    with pytest.raises(ValueError, match="non-finite"):
        normalise([float("nan")] + [0.0] * 383)


async def test_hash_embedder_async_batch() -> None:
    embedder = HashEmbedder()
    vectors = await embedder.embed(["one", "two"])
    assert len(vectors) == 2
    assert all(len(vector) == 384 for vector in vectors)

"""Token counting interfaces.

The real path uses the exact tokenizer shipped with the bge-small-en-v1.5
embedding model, so the 448-token invariant is measured in the same units as the
encoder's context window. The hash fallback has its own deterministic counter and
is labelled as such everywhere it is used.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class Encoding(Protocol):
    """The slice of ``tokenizers.Encoding`` count() needs."""

    @property
    def ids(self) -> list[int]:
        """Token ids, including special tokens."""
        ...


class TokenizerLike(Protocol):
    """The slice of ``tokenizers.Tokenizer`` count() needs."""

    def encode(self, text: str) -> Encoding:
        """Encode one string."""
        ...


@runtime_checkable
class TokenCounter(Protocol):
    """Counts model tokens for a string."""

    name: str

    def count(self, text: str) -> int:
        """Return the number of tokens in ``text``."""
        ...


class ModelTokenCounter:
    """Wraps a Hugging Face ``tokenizers.Tokenizer``."""

    def __init__(self, tokenizer: TokenizerLike) -> None:
        """Wrap a concrete tokenizer instance."""
        self._tokenizer = tokenizer
        self.name = "bge-small-en-v1.5"

    def count(self, text: str) -> int:
        """Count including the tokenizer's special tokens, matching the invariant."""
        return len(self._tokenizer.encode(text).ids)


class HashTokenCounter:
    """Deterministic word/punctuation counter for the hash-embedder fallback.

    It is *not* the model tokenizer. It exists so the fallback path still refuses
    pathological chunks and still computes a stable overlap length.
    """

    name = "hash-words"

    def count(self, text: str) -> int:
        """Count word and punctuation pieces."""
        return len(_WORD_RE.findall(text))


def make_hash_counter() -> HashTokenCounter:
    """Create the fallback counter."""
    return HashTokenCounter()

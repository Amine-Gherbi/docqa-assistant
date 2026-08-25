from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from docqa.ingest.chunking import (
    HARD_CAP,
    OVERLAP_TOKENS,
    TARGET_MAX,
    TARGET_MIN,
    Chunk,
    TokenCapViolation,
    assert_token_cap,
    build_chunks,
)
from docqa.ingest.extract import ExtractedPdf, Page, sha256_text
from docqa.ingest.tokens import HashTokenCounter


class CharCounter:
    """Counts characters; makes one 600-char word unbreakable."""

    name = "chars"

    def count(self, text: str) -> int:
        return len(text)


def _pdf(pages: list[str]) -> ExtractedPdf:
    page_objects = tuple(
        Page(page_no=index, text=text, sha256=sha256_text(text))
        for index, text in enumerate(pages, start=1)
    )
    return ExtractedPdf(
        path=Path("synthetic.pdf"),
        sha256="0" * 64,
        byte_size=1,
        page_count=len(pages),
        pages=page_objects,
        pymupdf_version="2.0",
    )


def _technical_page() -> str:
    sentence = (
        "Retrieval augmented generation combines a dense retriever with a "
        "sequence to sequence generator and conditions every answer on evidence. "
    )
    paragraphs = [" ".join([sentence] * 8) for _ in range(12)]
    return "\n\n".join(paragraphs).strip()


def test_chunk_text_is_exactly_prefix_plus_page_span() -> None:
    page_text = _technical_page()
    counter = HashTokenCounter()
    chunks = build_chunks(_pdf([page_text]), counter, "A Technical Paper")

    assert chunks
    for chunk in chunks:
        prefix = chunk.breadcrumb + "\n\n"
        assert chunk.breadcrumb == "A Technical Paper | p.1"
        assert chunk.text == prefix + page_text[chunk.char_start : chunk.char_end]
        assert chunk.token_count == counter.count(chunk.text)


def test_every_chunk_is_under_the_hard_cap() -> None:
    chunks = build_chunks(_pdf([_technical_page()]), HashTokenCounter(), "Paper")
    assert all(chunk.token_count <= HARD_CAP for chunk in chunks)
    assert max(chunk.token_count for chunk in chunks) <= TARGET_MAX


def test_non_final_chunks_hit_the_target_range() -> None:
    chunks = build_chunks(_pdf([_technical_page()]), HashTokenCounter(), "Paper")
    assert len(chunks) > 3
    for chunk in chunks[:-1]:
        assert TARGET_MIN <= chunk.token_count <= TARGET_MAX, chunk.token_count


def test_overlap_is_at_most_48_tokens() -> None:
    page_text = _technical_page()
    counter = HashTokenCounter()
    chunks = build_chunks(_pdf([page_text]), counter, "Paper")
    for previous, following in itertools.pairwise(chunks):
        assert following.page_no == previous.page_no
        assert following.char_start >= previous.char_start
        if following.char_start < previous.char_end:
            overlap = page_text[following.char_start : previous.char_end]
            assert counter.count(overlap) <= OVERLAP_TOKENS


def test_spans_cover_the_page_without_gaps() -> None:
    page_text = _technical_page()
    chunks = build_chunks(_pdf([page_text]), HashTokenCounter(), "Paper")
    assert chunks[0].char_start == 0
    assert chunks[-1].char_end == len(page_text)
    for previous, following in itertools.pairwise(chunks):
        gap = page_text[previous.char_end : following.char_start]
        assert not gap.strip(), gap


def test_pages_are_chunked_independently() -> None:
    first = _technical_page()
    second = "A short second page with one sentence about calibration."
    chunks = build_chunks(_pdf([first, second]), HashTokenCounter(), "Paper")
    pages = {chunk.page_no for chunk in chunks}
    assert pages == {1, 2}
    assert all(chunk.char_end <= len(first) for chunk in chunks if chunk.page_no == 1)
    assert all(chunk.char_end <= len(second) for chunk in chunks if chunk.page_no == 2)


def test_chunk_over_448_tokens_is_rejected_by_assertion() -> None:
    with pytest.raises(AssertionError, match="448"):
        assert_token_cap(HARD_CAP + 1, "too long")


def test_unbreakable_word_over_cap_trips_the_assertion() -> None:
    # 600 characters under CharCounter is 600 tokens, and there is no whitespace
    # to split on: the chunker must refuse rather than store a clipped vector.
    with pytest.raises((TokenCapViolation, AssertionError)):
        build_chunks(_pdf(["x" * 600]), CharCounter(), "Paper")


def test_short_page_may_be_short() -> None:
    chunks = build_chunks(_pdf(["A final short remark."]), HashTokenCounter(), "Paper")
    assert len(chunks) == 1
    assert chunks[0].token_count < TARGET_MIN


def test_empty_pages_produce_no_chunks() -> None:
    assert build_chunks(_pdf(["", "   \n\n  "]), HashTokenCounter(), "Paper") == []


def test_chunk_is_immutable() -> None:
    chunk = Chunk(0, 1, 0, 1, "0" * 64, "b", "t", 1)
    with pytest.raises(AttributeError):
        chunk.text = "other"  # type: ignore[misc]

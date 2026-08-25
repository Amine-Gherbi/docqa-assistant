"""Structure-aware chunking with a hard, asserted token ceiling.

Rules enforced here (see ``docs/INVARIANTS.md``):

* every stored chunk is at most ``HARD_CAP`` tokens under the model tokenizer;
* the target size is 320-420 tokens (short page tails are the only exceptions);
* consecutive chunks overlap by at most 48 tokens;
* each chunk carries a breadcrumb prefix, included in its token count;
* ``char_start``/``char_end`` index the pinned page string, so the stored text is
  exactly ``prefix + page_text[start:end]``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from docqa.ingest.extract import ExtractedPdf
from docqa.ingest.tokens import TokenCounter

TARGET_MIN = 320
TARGET_MAX = 420
HARD_CAP = 448
OVERLAP_TOKENS = 48
SEGMENT_MARGIN = 12

_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"\S+")


class TokenCapViolation(AssertionError):
    """Raised when a chunk cannot be made to fit the 448-token ceiling."""


@dataclass(frozen=True)
class Chunk:
    """A chunk with page-level provenance."""

    chunk_index: int
    page_no: int
    char_start: int
    char_end: int
    page_text_sha256: str
    breadcrumb: str
    text: str
    token_count: int


@dataclass
class _Atom:
    """A structure-aligned span that may be split further if it overflows."""

    start: int
    end: int


def assert_token_cap(token_count: int, chunk_text: str) -> None:
    """Reject an over-long chunk instead of storing a truncated vector."""
    assert token_count <= HARD_CAP, (
        f"chunk has {token_count} tokens (cap {HARD_CAP}); refusing to store it: "
        f"{chunk_text[:120]!r}"
    )


def breadcrumb_for(title: str, page_no: int) -> str:
    """Build the page breadcrumb stored with every chunk."""
    return f"{title} | p.{page_no}"


def build_chunks(
    extracted: ExtractedPdf, counter: TokenCounter, title: str
) -> list[Chunk]:
    """Chunk every page independently; chunks never cross page boundaries."""
    chunks: list[Chunk] = []
    index = 0
    for page in extracted.pages:
        if not page.text.strip():
            continue
        breadcrumb = breadcrumb_for(title, page.page_no)
        prefix = f"{breadcrumb}\n\n"
        prefix_tokens = counter.count(prefix)
        if prefix_tokens >= TARGET_MIN:
            raise ValueError(f"breadcrumb leaves no body budget: {breadcrumb!r}")
        page_chunks, index = _chunk_page(
            page.text,
            page.sha256,
            page.page_no,
            breadcrumb,
            prefix,
            prefix_tokens,
            counter,
            index,
        )
        chunks.extend(page_chunks)
    return chunks


def _chunk_page(
    text: str,
    page_sha256: str,
    page_no: int,
    breadcrumb: str,
    prefix: str,
    prefix_tokens: int,
    counter: TokenCounter,
    first_index: int,
) -> tuple[list[Chunk], int]:
    body_max = TARGET_MAX - prefix_tokens
    hard_body = HARD_CAP - prefix_tokens
    body_total = counter.count(text)
    segments = _segments(text, counter, body_total, body_max, hard_body)

    chunks: list[Chunk] = []
    index = first_index
    for position, (start, end) in enumerate(segments):
        if position > 0:
            previous_start, previous_end = segments[position - 1]
            candidate = _overlap_start(text, previous_start, previous_end, counter)
            if candidate < start and counter.count(text[candidate:end]) <= body_max:
                start = candidate
        chunk_text = prefix + text[start:end]
        token_count = counter.count(chunk_text)
        assert_token_cap(token_count, chunk_text)
        chunks.append(
            Chunk(
                chunk_index=index,
                page_no=page_no,
                char_start=start,
                char_end=end,
                page_text_sha256=page_sha256,
                breadcrumb=breadcrumb,
                text=chunk_text,
                token_count=token_count,
            )
        )
        index += 1
    return chunks, index


def _segments(
    text: str,
    counter: TokenCounter,
    body_total: int,
    body_max: int,
    hard_body: int,
) -> list[tuple[int, int]]:
    """Partition a page into near-equal segments, without overlap.

    Splitting evenly (rather than always filling to 420) avoids leaving a tiny
    tail at the end of every page. The page gets the fewest segments that fit
    under the hard cap, then each segment is aimed at >=320 tokens where the
    arithmetic allows; the remainder lands in the final segment. Overlap is
    added afterwards only where the extended segment still fits 420.
    """
    body_min = body_max - (TARGET_MAX - TARGET_MIN)
    if body_total <= hard_body:
        target = max(1, body_total)
    else:
        count = 1
        while (count - 1) * body_max + hard_body < body_total:
            count += 1
        # The first segment gets no overlap, so aim a little above the floor to
        # keep it inside 320-420 after rounding down to a word boundary.
        target = min(
            body_max, max(body_min + SEGMENT_MARGIN, math.ceil(body_total / count))
        )
    atoms = _atoms(text)
    segments: list[tuple[int, int]] = []
    i = 0
    while i < len(atoms):
        atom = atoms[i]
        start = atom.start
        end = atom.end
        if counter.count(text[start:end]) > target:
            cut = _cut_to_budget(text, start, end, target, counter)
            if cut > start:
                end = cut
                atom.start = _skip_ws(text, cut, atom.end)
                next_i = i
            else:
                # An unbreakable word. Keep it whole; the 448-token assertion
                # in the caller decides whether it is legal.
                end = atom.end
                next_i = i + 1
        else:
            next_i = i + 1
            while next_i < len(atoms):
                candidate_end = atoms[next_i].end
                if counter.count(text[start:candidate_end]) <= target:
                    end = candidate_end
                    next_i += 1
                    continue
                cut = _cut_to_budget(text, start, candidate_end, target, counter)
                if cut > end:
                    end = cut
                    atoms[next_i].start = _skip_ws(text, cut, atoms[next_i].end)
                break
        segments.append((start, end))
        i = next_i
    return segments


def _atoms(text: str) -> list[_Atom]:
    """Split a page into paragraph atoms, then sentence atoms when needed."""
    atoms: list[_Atom] = []
    for start, end in _paragraph_spans(text):
        paragraph = text[start:end]
        if len(paragraph) <= 0:
            continue
        atoms.extend(_Atom(s, e) for s, e in _sentence_spans(text, start, end))
    return atoms


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    """Return trimmed spans for runs of non-blank lines."""
    spans: list[tuple[int, int]] = []
    start: int | None = None
    cursor = 0
    for line in text.splitlines(keepends=True):
        if line.strip():
            if start is None:
                start = cursor
            end = cursor + len(line)
        else:
            if start is not None:
                spans.append(_trim(text, start, end))
                start = None
        cursor += len(line)
    if start is not None:
        spans.append(_trim(text, start, cursor))
    return [(s, e) for s, e in spans if e > s and text[s:e].strip()]


def _sentence_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Split one paragraph at sentence boundaries, tracking exact offsets."""
    spans: list[tuple[int, int]] = []
    cursor = start
    for match in _SENTENCE_BREAK.finditer(text, start, end):
        spans.append(_trim(text, cursor, match.start()))
        cursor = match.end()
    spans.append(_trim(text, cursor, end))
    return [(s, e) for s, e in spans if e > s and text[s:e].strip()]


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    """Strip whitespace from a span without losing its position in the page."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _skip_ws(text: str, position: int, limit: int) -> int:
    """Advance past whitespace, never beyond ``limit``."""
    while position < limit and text[position].isspace():
        position += 1
    return position


def _cut_to_budget(
    text: str, start: int, limit_end: int, budget: int, counter: TokenCounter
) -> int:
    """Largest word-boundary end in ``[start, limit_end]`` within ``budget`` tokens.

    Returns ``start`` when even the first word overflows, which signals an
    unbreakable atom to the caller.
    """
    if counter.count(text[start:limit_end]) <= budget:
        return limit_end
    ends = [match.end() for match in _WORD.finditer(text, start, limit_end)]
    if not ends:
        return start
    low, high = 0, len(ends) - 1
    best = start
    while low <= high:
        middle = (low + high) // 2
        candidate = ends[middle]
        if counter.count(text[start:candidate]) <= budget:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    return best


def _overlap_start(
    text: str, previous_start: int, previous_end: int, counter: TokenCounter
) -> int:
    """Return the start of the longest tail of the previous chunk that is ≤48 tokens."""
    if previous_end <= previous_start:
        return previous_end
    body = text[previous_start:previous_end]
    if counter.count(body) <= OVERLAP_TOKENS:
        return previous_start
    starts = [previous_start + match.start() for match in _WORD.finditer(body)]
    low, high = 0, len(starts) - 1
    best = previous_end
    while low <= high:
        middle = (low + high) // 2
        candidate = starts[middle]
        if counter.count(text[candidate:previous_end]) <= OVERLAP_TOKENS:
            best = candidate
            high = middle - 1
        else:
            low = middle + 1
    return best

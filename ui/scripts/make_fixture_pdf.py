"""Build the committed citation fixture PDF and its PyMuPDF provenance.

The PDF is deliberately small (four pages, five sentences each) and every
citation quote is one whole line, so PyMuPDF's ``search_for`` can produce real
``rects`` without guessing. PDF.js text extraction is verified separately by
``verify_fixtures.mjs`` before ``citations.json`` is written.

Run from the repository root:

    uv run python ui/scripts/make_fixture_pdf.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT / "public" / "fixtures" / "fixture-corpus.pdf"
SOURCE_PATH = Path(__file__).resolve().parent / "fixture-source.json"
FONT_DIR = ROOT / "node_modules" / "pdfjs-dist" / "standard_fonts"
REGULAR_FONT = FONT_DIR / "LiberationSans-Regular.ttf"
BOLD_FONT = FONT_DIR / "LiberationSans-Bold.ttf"
DOC_ID = "fixture-corpus"

PAGES: list[tuple[str, list[str]]] = [
    (
        "1  Retrieval",
        [
            "Dense retrieval maps each passage to a single vector.",
            "The vector index returns nearest neighbours by cosine distance.",
            "Lexical retrieval matches rare terms that embeddings often miss.",
            "Reciprocal rank fusion merges two lists without score scaling.",
            "A cross encoder reranks the fused candidates in a second pass.",
        ],
    ),
    (
        "2  Calibration",
        [
            "The refusal threshold is fitted on held out query passage pairs.",
            "A calibrated probability below the threshold triggers a refusal.",
            "Every citation must quote a verbatim span from its passage.",
            "A claim whose quote is absent is dropped before it is displayed.",
            "Each page is hashed so an extractor upgrade fails loudly.",
        ],
    ),
    (
        "3  Citations",
        [
            "This fixture corpus exists to exercise the citation interface.",
            "Hybrid retrieval finds the rare term “quokka” but vector only misses it.",
            "The lexical ranker uses ts_rank_cd rather than BM25.",
            "Reranking can move a passage from rank nine to rank one.",
            "The final context contains at most eight passages.",
        ],
    ),
    (
        "4  Rendering",
        [
            "The highlight layer rebuilds the page string before it searches.",
            "Whitespace is collapsed and typographic quotes are normalised.",
            "On a miss, the interface draws the stored rectangles instead.",
            "The citation render miss counter is reported with every answer.",
            "An answer with no resolvable citation is never shown to the reader.",
        ],
    ),
]


def build_pdf() -> None:
    """Write the fixture PDF with real text (not an image or a fake canvas)."""
    if not REGULAR_FONT.is_file() or not BOLD_FONT.is_file():
        raise SystemExit(
            "Liberation Sans is missing; run `npm --prefix ui install` first "
            f"(expected at {FONT_DIR})"
        )
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open()
    for heading, lines in PAGES:
        page = document.new_page(width=612, height=792)
        page.insert_font(fontname="docqa-bold", fontfile=str(BOLD_FONT))
        page.insert_font(fontname="docqa-regular", fontfile=str(REGULAR_FONT))
        page.insert_text(
            (72, 78), "Fixture Corpus", fontsize=18, fontname="docqa-bold"
        )
        page.insert_text((72, 112), heading, fontsize=13, fontname="docqa-bold")
        y = 168
        for line in lines:
            page.insert_text((72, y), line, fontsize=11, fontname="docqa-regular")
            y += 30
        page.insert_text(
            (72, 740),
            f"Committed fixture for the Marginalia citation UI. Page {page.number + 1} of 4.",
            fontsize=8,
            fontname="docqa-regular",
        )
    document.set_metadata(
        {
            "title": "Marginalia citation fixture",
            "author": "Marginalia",
            "subject": "Committed offline fixture for citation resolution",
            "producer": "PyMuPDF 2.0",
        }
    )
    document.save(PDF_PATH, garbage=4, deflate=True)
    document.close()


def build_source() -> None:
    """Record PyMuPDF page text, hashes, char offsets, quotes, and rects."""
    document = pymupdf.open(PDF_PATH)
    pages: list[dict[str, object]] = []
    citations: list[dict[str, object]] = []
    number = 0
    for page_index, (_, lines) in enumerate(PAGES):
        page = document[page_index]
        text = page.get_text("text", sort=True)
        pages.append(
            {
                "page_no": page_index + 1,
                "text": text,
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        )
        for line in lines:
            start = text.find(line)
            if start < 0:
                raise SystemExit(f"line not found in PyMuPDF text: {line!r}")
            quads = page.search_for(line, quads=True)
            if not quads:
                raise SystemExit(f"no rects for line: {line!r}")
            rects = [
                [
                    round(quad.rect.x0, 2),
                    round(quad.rect.y0, 2),
                    round(quad.rect.x1, 2),
                    round(quad.rect.y1, 2),
                ]
                for quad in quads
            ]
            number += 1
            citations.append(
                {
                    "id": f"fx-{number:02d}",
                    "claim": line.rstrip("."),
                    "doc_id": DOC_ID,
                    "page_no": page_index + 1,
                    "char_start": start,
                    "char_end": start + len(line),
                    "page_text_sha256": pages[-1]["sha256"],
                    "quote": line,
                    "rects": rects,
                }
            )
    document.close()
    SOURCE_PATH.write_text(
        json.dumps(
            {
                "doc_id": DOC_ID,
                "pdf": "/fixtures/fixture-corpus.pdf",
                "pinned_with": f"PyMuPDF {pymupdf.__version__}",
                "pages": pages,
                "citations": citations,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    """Generate the PDF, then its provenance file."""
    build_pdf()
    build_source()
    print(f"wrote {PDF_PATH} ({PDF_PATH.stat().st_size} bytes)")
    print(f"wrote {SOURCE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

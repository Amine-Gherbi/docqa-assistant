# docqa-assistant (P4) — agent context

Purpose: ask a question about a PDF corpus and get an answer whose every claim points at a highlighted span, with an inspectable retrieval trace, a calibrated refusal, and an eval harness that prints confidence intervals.

## Architecture
- Python 3.12 (`uv`), FastAPI, async SQLAlchemy, PostgreSQL 16 + pgvector 0.8, `fastembed` bge-small-en-v1.5 (384d, ONNX, CPU), React + Vite UI, PDF.js.
- Ingest: PyMuPDF blocks → structure-aware chunker (320–420 tok, hard cap 448, overlap ≤48, breadcrumb prefix) → fastembed → pgvector HNSW + `tsvector` GIN.
- Retrieve: vector top-20 ⊕ lexical top-20 (`ts_rank_cd`, not BM25) → RRF k=60 → rerank top-24 (ms-marco-MiniLM-L-6-v2) → top-8 into the prompt.
- Generate: **one forced tool call** `submit_answer`; validation rejects uncited claims before the user sees them; SSE streams validated claims, not tokens.
- Citation contract: `(doc_id, page_no, char_start, char_end, page_text_sha256, quote, rects[])`; the UI resolves the verbatim `quote` via the CSS Custom Highlight API and falls back to `rects[]`.

## Commands
- `uv run docqa ingest <pdf…>` · `uv run docqa eval` · `uv run docqa judge-check`
- `./scripts/verify.sh` → ruff + mypy --strict + pytest + the k6 concurrency scenario
- API 8100 · UI 5173 · local pgvector on 5433 (hosted Neon/Supabase if native build fails)

## Invariants
1. `len(tokenizer.encode(chunk).ids) <= 448`, asserted at ingest; a batch that exceeds it fails instead of storing a truncated vector.
2. Every claim carries ≥1 citation id that is in the context, and every `quote` is a normalised substring of the cited chunk. Otherwise the claim is dropped.
3. `corpus.lock.json` is enforced by SHA-256; ingest refuses a mismatched file. The corpus is never committed.
4. Refusal threshold comes from `eval/calibration.json` (`{a, b, threshold, fit_set_sha256}`), loaded at boot — never hand-tuned.
5. Every rate is printed with a Wilson 95% CI and its denominator. `stage_failed` is recorded per question.
6. The judge model differs from the generator and is pinned by exact id in `eval/judge.yaml`.
7. ONNX work runs in `anyio.to_thread.run_sync` behind `Semaphore(2)`.

## Pitfalls
- pgvector has no Windows binary and Docker is unavailable here: R0 chooses hosted pgvector first; CI always uses the `pgvector/pgvector:pg16` service container.
- Python 3.14 has no ML wheels — pin 3.12 in `.python-version` and use ONNX, not torch.
- PyMuPDF char offsets are meaningless inside PDF.js: resolve by `quote` first, `rects[]` as fallback, count `citation_render_miss`.
- The gateway may rewrite `tool_choice`: the CI probe runs before R3; the documented fallback is a JSON-only prompt plus `json-repair` (ADR 0004).
- Streaming generation would let an uncited claim reach the user; do not add it.

## Decisions
ADRs: `docs/adr/NNNN-*.md`. Environment: `docs/ENV.md`. Corpus: `docs/CORPUS.md`. Invariants: `docs/INVARIANTS.md`.

# docqa-assistant

[![eval](https://img.shields.io/badge/eval-3%20of%205%20gates%20fail-red)](docs/EVAL.md)

**What it is.** Ask a question about a PDF corpus and get answers whose claims point at highlighted spans, with an inspectable retrieval trace and a calibrated refusal. **Hardest problem.** Keeping the citation honest: a claim whose quote is not a verbatim substring of the cited chunk is dropped before the UI sees it, and PyMuPDF offsets are never trusted inside PDF.js. **Live URL.** Local only: UI `http://127.0.0.1:5173`, API `http://127.0.0.1:8788`. **Demo path.** `docs/DEMO.md` — one click loads an offline demo; there is no login. **Proof.** `docs/PROOF.md` — every claim with the command that reproduces it. **Eval status.** `uv run docqa eval` exits 6; 3 of 5 production gates fail because this build has no generator model key, so the labelled extractive fallback answers only 5/24 answerable questions. The numbers are reported, not tuned away. **Synthetic fixtures.** The UI citation demo uses a committed synthetic PDF (`ui/public/fixtures/fixture-corpus.pdf`) and a seeded offline answer (`ui/src/fixtures/demo-answer.json`), labelled in the UI. The eval numbers use the real locked corpus and real bge-small embeddings/cross-encoder; the corpus PDFs are not committed.

## Commands

```sh
uv run docqa ingest data/corpus/*.pdf     # fetch first: scripts/fetch-corpus
uv run docqa ingest --check               # verify the corpus against corpus.lock.json
uv run docqa serve --port 8788            # FastAPI answer API
uv run docqa eval                         # 30-question harness; exits 6 on a gate failure
uv run docqa eval --ablate vector-only    # weakened retriever must miss a gate
uv run docqa judge-check                  # pinned judge protocol; reliability UNVERIFIED
uv run python scripts/capture-assets      # 4 WebP + manifest.json for the hub
./scripts/verify.sh                       # ruff + mypy --strict + pytest (k6 opt-in)
```

The UI is a separate package. Windows PowerShell 5.1 does not accept `&&`, so use two statements:

```powershell
uv run docqa serve --port 8788            # terminal 1
Set-Location ui; npx vite preview --port 5173     # terminal 2 → http://localhost:5173
```

`./scripts/verify.sh` is a bash script: in PowerShell `bash` resolves to WSL, which has no `node` or
`uv`. Call Git bash instead: `& 'C:/Program Files/Git/bin/bash.exe' scripts/verify.sh`.


## Architecture

Ingest: PyMuPDF blocks → structure-aware chunker (320–420 tok, hard cap 448, overlap ≤48, breadcrumb prefix) → fastembed `BAAI/bge-small-en-v1.5` (384-d, ONNX CPU) → PostgreSQL 16 + pgvector HNSW, plus a generated `tsvector` GIN index.

Retrieve: vector top-20 ⊕ lexical top-20 (`ts_rank_cd`, **not BM25**) → RRF `k=60` → rerank top-24 with `Xenova/ms-marco-MiniLM-L-6-v2` → top-8 into the prompt.

Generate: one forced tool call `submit_answer`; schema and citation validation drop uncited claims before the response is returned. SSE streams validated claims, not tokens.

Citation contract: `(doc_id, page_no, char_start, char_end, page_text_sha256, quote, rects[])`. The UI resolves the verbatim `quote` through the CSS Custom Highlight API and falls back to `rects[]`, counting `citation_render_miss`.

## Eval, honestly

| configuration | n | metric | result | 95% Wilson CI | gate |
|---|---:|---|---:|---|---|
| production (hybrid+RRF+rerank, top-8) | 24 | recall@8 | 18/24 (75.0%) | [55.1%, 88.0%] | PASS |
| production | 24 | answer rate | 5/24 (20.8%) | [9.2%, 40.5%] | FAIL |
| production | 6 | unanswerable refusal | 6/6 (100%) | [61.0%, 100%] | PASS |
| production | 15 | citation precision* | 3/15 (20.0%) | [7.0%, 45.2%] | FAIL |
| production | 15 | faithfulness* | 2/15 (13.3%) | [3.7%, 37.9%] | FAIL |
| weakened (vector-only, top-3) | 24 | recall@3 | 15/24 (62.5%) | [42.7%, 78.8%] | FAIL |

\* rule-based, not a model judge. Full definitions, stage attribution and the calibration limitation are in `docs/EVAL.md`; the regenerated artifacts are `eval-report.json` and `eval-report-ablation.json`.

**Headline limitation.** The refusal threshold in `eval/calibration.json` was fitted on machine-generated self-retrieval sentences, not human question phrasing. On real questions it refuses 19/24 answerable questions at `low_score` before generation. Refitting on a larger human gold set is the remaining R5 action; the threshold was not hand-tuned to make a gate pass.

## Further reading

- `docs/PROOF.md` — reproducible claims and the unverified ones.
- `docs/INVARIANTS.md` — rules that must never break, with `file:line` and tests.
- `docs/INTERVIEW.md` — 20 hard questions with source citations.
- `docs/CORPUS.md` — corpus provenance and ingest stats.
- `docs/ENV.md` — the local environment, including the dummy-embedding fallback.
- `docs/ABLATION.md` — retrieval ablation and warm latency.
- `docs/adr/0004-structured-output.md` — forced-tool probe and JSON fallback.

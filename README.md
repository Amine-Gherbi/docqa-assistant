# Marginalia

[![eval](https://img.shields.io/badge/eval-3%20of%205%20gates%20fail-red)](docs/EVAL.md)

**What it is.** Ask a question about a PDF corpus and get answers whose claims point at highlighted spans, with an inspectable retrieval trace and a calibrated refusal. **Hardest problem.** Keeping the citation honest: a claim whose quote is not a verbatim substring of the cited chunk is dropped before the UI sees it, and PyMuPDF offsets are never trusted inside PDF.js. **Live URL.** Local only: UI `http://127.0.0.1:5173`, API `http://127.0.0.1:8788`. **Demo path.** `docs/DEMO.md` — one click loads an offline demo; there is no login. **Proof.** `docs/PROOF.md` — every claim with the command that reproduces it. **Eval status.** `bash scripts/live.sh eval` ran 30 questions against the live, keyless `openai-fast` endpoint on 2026-10-23 in 778.6 s (26 requests, 0 retries, 0 rate limits) and exited 6: 3 of 5 gates fail. Answer rate is 8/24 (33.3%, [18.0%, 53.3%]); citation precision and faithfulness are 7/9 (77.8%, [45.3%, 93.7%]); retrieval passes exactly (18/24) and unanswerable refusal is 6/6. The refusal threshold is now fitted on the human gold questions and admits all 24 answerable questions, so the model is the binding constraint: a forced-generation diagnostic (`--force-generation`) also answers 8/24, with 13 replies dropped by the citation contract because their quotes were not verbatim. The numbers are reported, not tuned away. **Synthetic fixtures.** The UI citation demo uses a committed synthetic PDF (`ui/public/fixtures/fixture-corpus.pdf`) and a seeded offline answer (`ui/src/fixtures/demo-answer.json`), labelled in the UI. The eval numbers use the real locked corpus and real bge-small embeddings/cross-encoder; the corpus PDFs are not committed.

## Commands

```sh
uv run docqa ingest data/corpus/*.pdf     # fetch first: scripts/fetch-corpus
uv run docqa ingest --check               # verify the corpus against corpus.lock.json
uv run docqa serve --port 8788            # FastAPI answer API (serves the demo cache)
bash scripts/live.sh probe --json   # real endpoint; records tool/JSON mode
bash scripts/live.sh ask --question "How many identical layers …?"
bash scripts/live.sh warm           # six demo answers → data/demo-answers.json
bash scripts/live.sh eval           # 30 questions, 778.6 s, exits 6 (3 gates fail)
bash scripts/live.sh eval --force-generation  # diagnostic, not a gate run; exits 0
uv run docqa eval --ablate vector-only    # weakened retriever misses recall (extractive)
uv run docqa judge-check                  # pinned judge protocol; reliability UNVERIFIED
./scripts/verify.sh                       # ruff + mypy --strict + pytest (k6 opt-in)
```

`scripts/live.sh` only exports the public, keyless endpoint
(`https://text.pollinations.ai/openai`, model `openai-fast`) and runs
`uv run <args>`. That endpoint answers about once per 30 s per IP; the client
serialises calls and waits at least 31 s between request starts.

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
| production | 24 | answer rate | 8/24 (33.3%) | [18.0%, 53.3%] | FAIL |
| production | 6 | unanswerable refusal | 6/6 (100%) | [61.0%, 100.0%] | PASS |
| production | 9 | citation precision* | 7/9 (77.8%) | [45.3%, 93.7%] | FAIL |
| production | 9 | faithfulness* | 7/9 (77.8%) | [45.3%, 93.7%] | FAIL |
| forced diagnostic (not a gate run) | 24 | answer rate | 8/24 (33.3%) | [18.0%, 53.3%] | — |
| weakened (vector-only, top-3) | 24 | recall@3 | 15/24 (62.5%) | [42.7%, 78.8%] | FAIL |

\* rule-based, not a model judge. Full definitions, stage attribution and the calibration limitation are in `docs/EVAL.md`; the run artifacts are `eval-report.json` (final production), `eval-report-baseline.json` (old threshold), `eval-report-forced.json` (diagnostic) and `eval-report-ablation.json`.

**Headline limitation.** Refitting the threshold on the human gold questions removed the threshold as the main blocker, but it did not make the model answer: even with refusal bypassed, the free endpoint returns a contract-valid answer for 8/24 answerable questions. The other 13 replies that reached validation paraphrased or overran their quote and were correctly dropped. Citation precision and faithfulness fail at 7/9 because R4 cites two non-gold chunks. The committed calibration is **in-sample** — fitted on the same 30 gold questions the eval runs — so its 0.833 balanced accuracy is optimistic; the id-parity held-out folds also give 0.833, with only three unanswerable questions each. The threshold was not hand-tuned to make a gate pass.

## Further reading

- `docs/PROOF.md` — reproducible claims and the unverified ones.
- `docs/INVARIANTS.md` — rules that must never break, with `file:line` and tests.
- `docs/INTERVIEW.md` — 20 hard questions with source citations.
- `docs/CORPUS.md` — corpus provenance and ingest stats.
- `docs/ENV.md` — the local environment, including the dummy-embedding fallback.
- `docs/ABLATION.md` — retrieval ablation and warm latency.
- `docs/adr/0004-structured-output.md` — forced-tool probe and JSON fallback.

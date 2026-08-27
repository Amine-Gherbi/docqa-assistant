# Marginalia

**Case study:** https://gherbi.pages.dev/projects/docqa-assistant/ — the app runs locally, not on a host.

**What it is.** Ask a question about a PDF corpus and get an answer whose every claim is tied to a verbatim quote from the cited page, highlighted in the reader, with the retrieval trace and the calibrated refusal that produced it.

**What it does not do.** No hosted URL, no login, no video, no unvalidated token streaming, and no committed corpus PDFs; the free endpoint's answer quality is measured, not promised — 3 of 5 production gates fail (2026-10-23).

**Run it.** Two commands (Git Bash; Windows PowerShell users should call Git Bash for `verify.sh`):
```sh
uv run docqa serve --port 8788       # API: /ask, /eval/report, /health
npm --prefix ui run dev              # UI: http://127.0.0.1:5173
```

**Free-endpoint caveat.** The generator is the public, keyless `https://text.pollinations.ai/openai` model `openai-fast` (the response reports `gpt-oss-20b`); it allows about one answer per 30 s per IP, has no SLA, and may change its backing model. Measured 2026-10-23: the 30-question production eval took 778.6 s with 26 requests, 0 retries and 0 rate limits.

**Where the numbers live.** `docs/EVAL.md` (rates, Wilson intervals, denominators, gate rules) and `docs/ENV.md` (endpoint, versions, measured run cost); `docs/PROOF.md` maps each claim below to the command that reproduces it.

## Status, measured 2026-10-23

The production run used the live keyless endpoint, the real `BAAI/bge-small-en-v1.5` embedder and the real `Xenova/ms-marco-MiniLM-L-6-v2` reranker over 30 fixed questions (24 answerable, 6 unanswerable). Two of the five gates pass, so `bash scripts/live.sh eval` exits 6. The failing gates stay failing; they were not tuned away.

| gate | n | measured | Wilson 95% | result |
|---|---:|---|---:|---|
| retrieval recall@8 | 24 | 18/24 (75.0%) | [55.1%, 88.0%] | PASS |
| answer rate | 24 | 8/24 (33.3%) | [18.0%, 53.3%] | FAIL |
| unanswerable refusal | 6 | 6/6 (100.0%) | [61.0%, 100.0%] | PASS |
| citation precision* | 9 | 7/9 (77.8%) | [45.3%, 93.7%] | FAIL |
| faithfulness* | 9 | 7/9 (77.8%) | [45.3%, 93.7%] | FAIL |

\* rule-based, not a model judge.

**Why the answer-rate gate fails.** The model is the binding constraint, not the threshold. A forced-generation diagnostic (`bash scripts/live.sh eval --force-generation`, `eval-report-forced.json`, `gate_run: false`) also answers 8/24 (33.3%, [18.0%, 53.3%]). Of the 16 answerable questions that did not answer, 13 model replies made a claim whose quote was not a verbatim substring of the cited chunk, so the validator dropped it; 11 of those 13 had the gold chunk in the top eight. Two replies declined with `insufficient_context` and one returned no `submit_answer` call. The committed threshold admits all 24 answerable questions in-sample.

## Architecture

Ingest: PyMuPDF blocks → structure-aware chunker (hard token cap enforced at ingest) → fastembed `BAAI/bge-small-en-v1.5` (384-d, ONNX CPU) → PostgreSQL 16 + pgvector HNSW, plus a generated `tsvector` GIN index.

Retrieve: vector ⊕ lexical (`ts_rank_cd`, not BM25) → reciprocal rank fusion → cross-encoder rerank → top-8 into the prompt.

Generate: one forced `submit_answer` tool call; schema and citation validation drop uncited claims before the response is returned. Generation is non-streaming — the API returns one validated JSON response and has no SSE route — so an uncited claim cannot reach the UI.

Citation contract: `(doc_id, page_no, char_start, char_end, page_text_sha256, quote, rects[])`. The UI resolves the verbatim `quote` through the CSS Custom Highlight API and falls back to `rects[]`, counting `citation_render_miss`.

## Demo data

The UI opens the real Sentence-BERT paper from the corpus (`ui/public/papers/sentence-bert.pdf`; licence in `docs/CORPUS.md`) with recorded extractive answers in `ui/src/fixtures/real-paper.json`; every quote is re-checked against the committed PDF by `ui/scripts/verify_demo_paper.mjs`. `ui/src/fixtures/demo-answer.json` is only the offline fallback when the API is unreachable. The screenshots use this seeded data; no live model call was made. The eval numbers use the real locked corpus, not this fixture.

## Commands

```sh
scripts/fetch-corpus                     # materialise corpus PDFs (not committed)
uv run docqa ingest data/corpus/*.pdf
uv run docqa ingest --check              # verify SHA-256 against corpus.lock.json
uv run docqa serve --port 8788
bash scripts/live.sh probe --json        # real endpoint; records tool/JSON mode
bash scripts/live.sh eval                # 30 questions; 778.6 s on 2026-10-23
bash scripts/live.sh eval --force-generation
uv run docqa eval --ablate vector-only
uv run docqa judge-check
./scripts/verify.sh                      # ruff + mypy + pytest (k6 opt-in)
npm --prefix ui run build
```

`scripts/live.sh` only exports the public, keyless endpoint and model id; it contains no secret. The client serialises calls and spaces request starts by at least 31 s.

## What is deliberately not claimed

- The eval set is 30 questions (24 answerable, 6 unanswerable), not 100. The Wilson intervals are wide and the ordering between vector-only, RRF and rerank is not statistically established.
- Citation precision and faithfulness are rule-based; no model judge ran. Their 77.8% point estimates rest on n=9, so the interval reaches down to 45.3%.
- The committed calibration is fitted in-sample on the same 30 gold questions the eval runs. The held-out id-parity folds each contain only three unanswerable questions.
- The answer-quality numbers are a snapshot of the free third-party endpoint on 2026-10-23. It may change its backing model or rate limit without notice.
- The build is local only: UI `http://127.0.0.1:5173`, API `http://127.0.0.1:8788`. There is no hosted URL, and the k6 concurrency scenario is opt-in and unverified where k6 is absent.
- The corpus PDFs are not committed. `scripts/fetch-corpus` materialises them from `corpus.lock.json` before any ingest claim can be reproduced.
- The UI demo is seeded and offline-labelled. It is not evidence that the live free endpoint answers well; the eval above is.

## Further reading

- `docs/PROOF.md` — each claim with the command that reproduces it, plus what stays unverified.
- `docs/INVARIANTS.md` — rules that must never break, with `file:line` and tests.
- `docs/INTERVIEW.md` — the hard questions with source citations.
- `docs/CORPUS.md` — corpus provenance and ingest stats.
- `docs/ENV.md` — the local environment, the free endpoint, and the dummy-embedding fallback.
- `docs/ABLATION.md` — retrieval ablation and warm latency.
- `docs/adr/0004-structured-output.md` — forced-tool probe and JSON fallback.

# Marginalia

**Live:** UI <https://marginalia-ui.onrender.com> · API <https://marginalia-api-4pvl.onrender.com> (`/health`, `/ready`, `/ask`)
**Case study:** <https://gherbi.pages.dev/projects/docqa-assistant/>

**What it is.** Ask a question about a PDF corpus and get an answer whose every claim is tied to a verbatim quote from the cited page, highlighted in the reader, with the retrieval trace and the calibrated refusal that produced it.

**What it does not do.** No login, no video, no unvalidated token streaming, and no committed corpus PDFs. Answer quality is measured, not promised: all five production gates pass on the 30-question set below.

**The hosted pair.** The API is a Docker image on Render's free plan in Frankfurt; the UI is a static site on the same plan; the corpus is Neon PostgreSQL 16 with pgvector (5 documents, 260 chunks). The API runs as one 512 MB, 0.1-CPU container that a five-minute HTTP monitor keeps warm, so a visitor does not pay a wake penalty. It answers one question at a time: the model client keeps one request in flight and spaces request starts by `DOCQA_MODEL_MIN_INTERVAL_S` (20 s). The UI warms `/health` on load and retries one 502/503/504 `/ask` after the warm-up.

**Models.** The generator is `gemini-flash-lite-latest` on the keyed Gemini free tier, with `gemini-3.8-flash` as the first documented retry. The judge is pinned to `gemini-3.8-flash` in `eval/judge.yaml`; the eval probe could not get a judge answer (HTTP 429/429/503), so the generator is never substituted and citation precision and faithfulness are reported by a deterministic rule-based checker, not a model judge.

**Run it locally.** Two commands (Git Bash; Windows PowerShell users should call Git Bash for `verify.sh`):
```sh
uv run docqa serve --port 8788       # API: /ask, /eval/report, /health, /ready
npm --prefix ui run dev              # UI: http://127.0.0.1:5173
```

**Where the numbers live.** `docs/EVAL.md` (rates, Wilson intervals, denominators, gate rules) and `docs/ENV.md` (local and hosted environments, the free instance's limits, measured run cost); `docs/PROOF.md` maps each claim below to the command that reproduces it, including the hosted `/ask` transcript.

## Status

The production eval ran on the build machine against the locked corpus, the real `BAAI/bge-small-en-v1.5` embedder, the real `Xenova/ms-marco-MiniLM-L-6-v2` reranker and the keyed generator, over 30 fixed questions (24 answerable, 6 unanswerable). All five gates pass, so `bash scripts/live.sh eval` exits 0. The refusal threshold in `eval/calibration.json` was not changed. The live pair is separate evidence: its `/ask` transcript is in `docs/PROOF.md`.

| gate | n | measured | Wilson 95% | result |
|---|---:|---|---:|---|
| retrieval recall@8 | 24 | 18/24 (75.0%) | [55.1%, 88.0%] | PASS |
| answer rate | 24 | 15/24 (62.5%) | [42.7%, 78.8%] | PASS |
| unanswerable refusal | 6 | 6/6 (100.0%) | [61.0%, 100.0%] | PASS |
| citation precision* | 15 | 13/15 (86.7%) | [62.1%, 96.3%] | PASS |
| faithfulness* | 15 | 12/15 (80.0%) | [54.8%, 93.0%] | PASS |

\* rule-based, not a model judge.

**Reading the pass.** Retrieval and faithfulness pass at their thresholds (75.0% and 80.0%), so both are point-estimate passes and the intervals reach below the gate. Nine answerable questions do not answer: five do not have the gold chunk in the top eight, and four return a quote that crosses the interleaved columns of the two-column PDF extraction, so the citation contract drops the claim. The success count rose from 11/24 to 15/24 when the generation prompt began stating that passages are interleaved extraction and that a quote must not span a column gap; the JSON fallback fired zero times. The four remaining column-spanning quotes need column-aware extraction at ingest, which changes chunk text, retrieval and the gold chunk ids, so it stays a separate ingest task rather than a quiet change to this measurement.

## Architecture

Ingest: PyMuPDF blocks → structure-aware chunker (hard token cap enforced at ingest) → fastembed `BAAI/bge-small-en-v1.5` (384-d, ONNX CPU) → PostgreSQL 16 + pgvector HNSW, plus a generated `tsvector` GIN index.

Retrieve: vector ⊕ lexical (`ts_rank_cd`, not BM25) → reciprocal rank fusion → cross-encoder rerank → top-8 into the prompt.

Generate: one forced `submit_answer` tool call; schema and citation validation drop uncited claims before the response is returned. Generation is non-streaming — the API returns one validated JSON response and has no SSE route — so an uncited claim cannot reach the UI.

Citation contract: `(doc_id, page_no, char_start, char_end, page_text_sha256, quote, rects[])`. The UI resolves the verbatim `quote` through the CSS Custom Highlight API and falls back to `rects[]`, counting `citation_render_miss`.

## Demo data

The UI opens the real Sentence-BERT paper from the corpus (`ui/public/papers/sentence-bert.pdf`; licence in `docs/CORPUS.md`) with recorded extractive answers in `ui/src/fixtures/real-paper.json`; every quote is re-checked against the committed PDF by `ui/scripts/verify_demo_paper.mjs`. `ui/src/fixtures/demo-answer.json` is only the offline fallback when the API is unreachable. The screenshots use this seeded data; no live model call was made. The eval numbers use the real locked corpus, not this fixture.

## Visitor uploads

A visitor can drop a PDF on the page instead of asking the seeded paper. The
bytes are stored in Postgres and parsed outside the request loop by a worker
under a 90 s timeout; the page polls `queued → parsing (n/m) → embedding →
ready`, then opens the visitor's own PDF in the same two-pane reader and asks
against that document's collection only. Limits: 25 MB, 80 pages, three
uploads per browser per 24 hours, one at a time, and one store-wide upload
window of 20 per hour.

**Uploaded text is sent to a third-party model on a free tier. The provider may
use it to improve its products. Do not upload confidential documents.**
Uploaded documents are deleted after 24 hours.

## Commands

```sh
scripts/fetch-corpus                     # materialise corpus PDFs (not committed)
uv run docqa ingest data/corpus/*.pdf
uv run docqa ingest --check              # verify SHA-256 against corpus.lock.json
uv run docqa serve --port 8788
bash scripts/live.sh probe --json        # real endpoint; records tool/JSON mode
bash scripts/live.sh eval                # 30 questions; 710.3 s on the keyed free tier
bash scripts/live.sh eval --force-generation
uv run docqa eval --ablate vector-only
uv run docqa judge-check
./scripts/verify.sh                      # ruff + mypy + pytest (k6 opt-in)
npm --prefix ui run build
```

`scripts/live.sh` reads the key from `P:/portfolio/.secrets/model.env` (outside every repository) and exports only `DOCQA_MODEL_API_KEY` and `DOCQA_JUDGE_API_KEY`; it contains no secret and refuses to run without the key. The client serialises calls and spaces request starts by at least `DOCQA_MODEL_MIN_INTERVAL_S`.

## What is deliberately not claimed

- The eval set is 30 questions (24 answerable, 6 unanswerable), not 100. The Wilson intervals are wide and the ordering between vector-only, RRF and rerank is not statistically established.
- Citation precision and faithfulness are rule-based; no model judge ran. Their 86.7% and 80.0% point estimates rest on n=15, and the faithfulness interval reaches down to 54.8%.
- The committed calibration is fitted in-sample on the same 30 gold questions the eval runs. The held-out id-parity folds each contain only three unanswerable questions.
- The answer-quality numbers are a snapshot of the keyed Gemini free tier, which has no SLA and returned HTTP 429/503 during the judge probe.
- The hosted free instance scales to zero and answers one request at a time. A warm `/ask` measured 17–25 s in reproduction samples; a cold start waits about a minute. The k6 concurrency scenario is opt-in and unverified where k6 is absent.
- The corpus PDFs are not committed. `scripts/fetch-corpus` materialises them from `corpus.lock.json` before any ingest claim can be reproduced.
- The UI demo is seeded and offline-labelled. It is not evidence that the live generator answers well; the eval above is.

## Further reading

- `docs/PROOF.md` — each claim with the command that reproduces it, plus the hosted transcript and the 512 MB measurements.
- `docs/INVARIANTS.md` — rules that must never break, with `file:line` and tests.
- `docs/INTERVIEW.md` — the hard questions with source citations.
- `docs/CORPUS.md` — corpus provenance and ingest stats.
- `docs/ENV.md` — the local and hosted environments, the keyed endpoint, and the dummy-embedding fallback.
- `docs/ABLATION.md` — retrieval ablation and warm latency.
- `docs/adr/0004-structured-output.md` — forced-tool probe and JSON fallback.

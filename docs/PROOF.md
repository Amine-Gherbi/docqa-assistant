# Proof

Every row maps a README claim to the command that reproduces it and the output
to expect. Numbers are repeated from `docs/EVAL.md` and `docs/ENV.md`, both
recorded **2026-10-23**; nothing here is hand-tuned or estimated. Rows marked
**unverified** were not run and say why.

Build machine: Windows 11 Pro 10.0.26200 (Git Bash), Python 3.12.15,
PostgreSQL 16 + pgvector 0.8.7 on port 5433, ONNX Runtime on CPU.

## README claims

| # | Claim | Command | Expected output | Status |
|---|---|---|---|---|
| 1 | Every claim carries an in-context citation id and a verbatim quote; otherwise it is dropped | `uv run pytest tests/test_citations.py` | all tests pass; `validate_claims` at `src/docqa/answer.py:542` | verified |
| 2 | A chunk over the 448-token cap fails ingest instead of storing a truncated vector | `uv run pytest tests/test_chunking.py` | all tests pass; `HARD_CAP = 448` at `src/docqa/ingest/chunking.py:24` | verified |
| 3 | Ingest refuses a PDF whose SHA-256 differs from `corpus.lock.json` | `uv run docqa ingest --check` | five `OK <file>.pdf` lines, exit 0; `verify_file` at `src/docqa/ingest/lock.py:82` | verified 2026-10-23 |
| 4 | The refusal threshold is loaded from `eval/calibration.json` at boot and never hand-tuned | `uv run pytest tests/test_refusal.py`; read `eval/calibration.json` | tests pass; threshold `0.6361`, fit `gold-human-v1`; `load_calibration` at `src/docqa/answer.py:245` | verified |
| 5 | Every rate carries a Wilson 95% interval and a denominator; `stage_failed` is recorded per question | `uv run pytest tests/test_eval.py` | tests pass; `wilson_interval` at `src/docqa/eval.py:104`, counts at `:588` | verified |
| 6 | The judge is pinned to a different exact model id from the generator, is never substituted by the generator, and its reliability is unverified | `bash scripts/live.sh eval` (its judge probe) | judge `gemini-3.8-flash` against generator `gemini-flash-lite-latest`; `provenance.judge.status: judge_model_unavailable`, attempts `[timeout, 503, timeout]`, `model_judge_used: false`; artifact `eval/judge-status.json` | verified 2026-10-29 (judge unavailable; κ not computed) |
| 7 | ONNX work runs in a worker thread behind `Semaphore(2)` | `uv run pytest tests/test_embedder.py` | tests pass; `src/docqa/onnx.py:15` | verified |
| 8 | Production retrieval recall@8 is 18/24 (75.0%) | `bash scripts/live.sh eval` | `retrieval_recall_at_8` rate `0.750`, n=24, Wilson `[0.551, 0.880]`; artifact `eval-report.json` | verified 2026-10-23 |
| 9 | Production answer rate is 8/24 (33.3%) and the gate fails | `bash scripts/live.sh eval` | `answer_rate_answerable` rate `0.333`, n=24, Wilson `[0.180, 0.533]`, gate row `FAIL` | verified 2026-10-23 |
| 10 | Unanswerable refusal is 6/6 (100.0%) | `bash scripts/live.sh eval` | `unanswerable_refusal_rate` rate `1.000`, n=6, Wilson `[0.610, 1.000]` | verified 2026-10-23 |
| 11 | Citation precision and faithfulness are 7/9 (77.8%) each and are rule-based, not a model judge | `bash scripts/live.sh eval` | both rates `0.778`, n=9, Wilson `[0.453, 0.937]`; `provenance.judge.model_judge_used: false` | verified 2026-10-23 |
| 12 | The model, not the threshold, caps the answer rate; forced generation also answers 8/24 | `bash scripts/live.sh eval --force-generation` | `run.gate_run: false`; answer rate 8/24 (33.3%) `[18.0%, 53.3%]`; exit 0 by design; artifact `eval-report-forced.json` | verified 2026-10-23 |
| 13 | A deliberately weakened vector-only retriever fails the recall gate | `uv run docqa eval --ablate vector-only` | recall@3 15/24 (62.5%) `[42.7%, 78.8%]` FAIL; exit 6; artifact `eval-report-ablation.json` | verified (committed artifact) |
| 14 | The production eval fits the free endpoint's rate budget | `bash scripts/live.sh eval` | `run.wall_clock_s: 778.6`, 30 questions, `model_calls {requests: 26, retries: 0, rate_limited: 0}`, `blocked: null`; `docs/ENV.md` | verified 2026-10-23 |
| 15 | The gateway honours the forced `submit_answer` call | `bash scripts/live.sh probe --json` | `mode: "tool"`, `tool_call_returned: true`, `wall_clock_s: 14.531`; artifact `eval/structured-output.json` | verified 2026-10-29 |
| 16 | The generator is the keyed Gemini free tier, `gemini-flash-lite-latest`, with one request at a time and backoff on 429/503/timeouts | read `docs/ENV.md`; `bash scripts/live.sh eval` | `provenance.generator.endpoint_host: generativelanguage.googleapis.com`, `requested_models` starts `gemini-flash-lite-latest`, per-question `model_answered` recorded; `scripts/live.sh` refuses without the key | verified 2026-10-29 |
| 17 | The UI demo uses the committed Sentence-BERT paper and recorded quotes, not a live call; every citation scrolls, highlights and resolves | `node ui/scripts/verify_demo_paper.mjs`; `npm --prefix ui run e2e` | 9/9 recorded claims resolve against `ui/public/papers/sentence-bert.pdf`; 4 Playwright tests pass with no render miss | verified 2026-10-23 |
| 18 | The submitted screenshots are four 1440x900 CSS-px shots at 2x DPR in WebP, with seeded data | `uv run python scripts/capture-assets` | `[assets-mirror] valid: 4 WebP at 2880x1800`; manifest has `previewCommand` and no `demo` block | verified 2026-10-23 |
| 19 | Lint, strict types and tests pass; k6 is opt-in | `bash scripts/verify.sh` | ruff, `mypy --strict`, 107 tests pass; prints `SKIP k6`; final line `verify: PASS` | verified 2026-10-23 (k6 skipped) |
| 20 | The UI builds | `npm --prefix ui run build` | `tsc -b && vite build` exits 0 | verified |
| 21 | The build is local only and the corpus PDFs are not committed | — | UI `127.0.0.1:5173`, API `127.0.0.1:8788`; no hosted URL; `scripts/fetch-corpus` must run first | **unverified by design** — no deployment exists and the corpus is not in git |
| 22 | The k6 concurrency scenario is green | `DOCQA_WITH_K6=1 bash scripts/verify.sh` | k6 p95 under the configured budget | **unverified** — k6 is not installed on this machine, so `verify.sh` prints `SKIP k6` |
| 23 | Generation is non-streaming: the API returns one validated JSON response and has no SSE route | `grep -rn "StreamingResponse\|text/event-stream" src/docqa` | no matches; `src/docqa/api.py:4` states the response is non-streaming | verified |

## Eval gates, measured 2026-10-23

**Generator:** `openai-compatible:openai-fast` (`https://text.pollinations.ai/openai`; the response reports `gpt-oss-20b`).
**Embedder:** `fastembed:BAAI/bge-small-en-v1.5` (384-d, ONNX CPU).
**Reranker:** `fastembed:Xenova/ms-marco-MiniLM-L-6-v2` (ONNX CPU).
**Run:** 30 fixed questions (24 answerable, 6 unanswerable), 778.6 s, 26 model requests, 0 retries, 0 rate limits. Artifact: `eval-report.json`.

| gate | n | measured | Wilson 95% | gate |
|---|---:|---|---:|---|
| retrieval recall@8 | 24 | 18/24 (75.0%) | [55.1%, 88.0%] | PASS |
| answer rate | 24 | 8/24 (33.3%) | [18.0%, 53.3%] | FAIL |
| unanswerable refusal | 6 | 6/6 (100.0%) | [61.0%, 100.0%] | PASS |
| citation precision* | 9 | 7/9 (77.8%) | [45.3%, 93.7%] | FAIL |
| faithfulness* | 9 | 7/9 (77.8%) | [45.3%, 93.7%] | FAIL |

\* rule-based, not a model judge. Rule: a citation is correct when it names the
gold chunk; a claim is faithful when every citation names the gold chunk and its
verbatim quote contains, or is contained in, the gold quote.

Two of five gates pass, so the command exits **6**. The forced-generation
diagnostic is **not** a gate run: it bypasses only the calibrated low-score
refusal, keeps the citation validator, records `gate_run: false`, and exits 0 by
design.

## What is deliberately not claimed

- The eval set is 30 questions (24 answerable, 6 unanswerable), not the planned
  100. The intervals are wide and the ordering between vector-only, RRF and
  rerank is not statistically established.
- Citation precision and faithfulness are rule-based. Their 77.8% point
  estimates rest on n=9, so the Wilson interval reaches down to 45.3%; they are
  not evidence of general precision.
- The committed calibration is fitted in-sample on the same 30 gold questions
  the eval runs. The held-out id-parity folds each contain only three
  unanswerable questions, so 0.833 is a direction, not a tight estimate.
- The answer-quality numbers are a snapshot of the free third-party endpoint on
  2026-10-23. It has no SLA and may change its backing model or rate limit.
- The UI screenshots are seeded/offline: the demo answers are recorded and the
  cites resolve against the committed Sentence-BERT PDF. No live model call was
  made while capturing them.
- There is no hosted URL and no video. `k6` is not installed, so the concurrency
  scenario is unverified on this machine.
- The corpus PDFs are not committed. `scripts/fetch-corpus` materialises them
  from `corpus.lock.json` before any ingest claim can be reproduced.

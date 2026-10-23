# Proof

Every row names the command that reproduces it and the output to expect. Rows
marked **unverified** were not run on this machine; the reason is stated rather
than hidden. The build machine is Windows 11 Pro 10.0.26200 (Git Bash), Python
3.12.15, PostgreSQL 16 + pgvector 0.8.7 on port 5433.

| # | Claim | Command | Expected output | Status |
|---|---|---|---|---|
| 1 | The CLI has exactly seven verbs | `uv run docqa --help` | usage shows `{ingest,serve,ask,warm,probe,judge-check,eval}` and no other verb | verified |
| 2 | The corpus matches `corpus.lock.json` | `uv run docqa ingest --check` | five `OK <file>.pdf` lines, exit 0 | verified |
| 3 | Lint, strict types and tests pass | `./scripts/verify.sh` | ruff/mypy/pytest sections end with `verify: PASS`; k6 prints `SKIP` because it is not installed | verified (k6 skipped) |
| 4 | Retrieval recall@8 is 18/24 | `bash scripts/live.sh eval` | `retrieval_recall_at_8` 0.750, n=24, Wilson [0.551, 0.880] | verified (regenerated `eval-report.json`, 2026-10-23) |
| 5 | Unanswerable refusal is 6/6 | `bash scripts/live.sh eval` | `unanswerable_refusal_rate` 1.000, n=6, Wilson [0.610, 1.000] | verified (regenerated `eval-report.json`, 2026-10-23) |
| 6 | Three production gates fail, so eval exits 6 | `bash scripts/live.sh eval` | 2 PASS / 3 FAIL: answer rate 8/24 (33.3%) [18.0%, 53.3%], citation precision 7/9 (77.8%) [45.3%, 93.7%] and faithfulness 7/9 (77.8%) fail; retrieval 18/24 and refusal 6/6 pass; `GATE FAILURE: exit 6` | verified (regenerated `eval-report.json`, 2026-10-23) |
| 7 | A weakened retriever fails a gate | `uv run docqa eval --ablate vector-only` | recall@3 15/24 (62.5%) FAIL; unanswerable refusal 1/6 FAIL; exit 6 | verified (committed `eval-report-ablation.json`) |
| 8 | The citation contract drops bad claims | `uv run pytest tests/test_citations.py` | all tests pass | verified via `verify.sh` |
| 9 | The 448-token cap fails ingest instead of truncating | `uv run pytest tests/test_chunking.py` | all tests pass | verified via `verify.sh` |
| 10 | The judge is pinned to a different model but reliability is unmeasured | `uv run docqa judge-check` | model `claude-opus-5-5` differs from generator `claude-sonnet-5-5`; `status: UNVERIFIED`; reason says no endpoint/kappa | verified |
| 11 | The asset snapshot has four 2880x1800 WebP files and a schema-valid manifest | `uv run python scripts/capture-assets` | `[assets-mirror] valid: 4 WebP at 2880x1800, 6 measurements, 7 invariants, 5 limits` | verified |
| 12 | The hub's own `check:assets` accepts the snapshot | `ASSET_PROJECTS_DIR=... ASSET_ROOT=P:/portfolio/_shared/assets npm --prefix P:/portfolio/portfolio-site run check:assets` | `[assets] docqa-assistant: manifest valid, 4 shots present` | **unverified** — the session sandbox permits reads only under `docqa-assistant`; `node` and `npm --prefix` against `portfolio-site` were denied. `capture-assets --validate-only` mirrors `manifest-schema.mjs` and `check-assets.mjs` instead. |
| 13 | The k6 concurrency p95 holds | `./scripts/verify.sh` (with k6 installed) | `k6 run scripts/k6-ask.js`; threshold `http_req_duration p(95)<5000` | **unverified** — `k6` is not installed on this machine; `verify.sh` prints `SKIP k6` |
| 14 | The demo video exists | `docs/DEMO.md` | an MP4 under `_shared/assets/docqa-assistant/` | **unverified** — no screen recorder is available; the click path and capture command are in `docs/DEMO.md` |
| 15 | A hosted live URL exists | — | a public URL | **unverified** — this build is local only (`127.0.0.1:5173`, API `:8788`); no deployment was made |
| 16 | The UI citation fixtures resolve 20/20 | `npm --prefix ui run e2e` | Playwright highlight spec passes | **unverified** in this session — the Playwright browser is installed, but the e2e suite was not re-run during R6; the live capture did resolve the demo quote through the CSS Custom Highlight API |
| 17 | The real embedding and reranker models were used for the eval | `uv run docqa eval` | banners `fastembed:BAAI/bge-small-en-v1.5` and `fastembed:Xenova/ms-marco-MiniLM-L-6-v2` | verified (recorded in `docs/ENV.md` and `eval-report.json`) |
| 18 | Ingest stats: 260 chunks, max 430 tokens, 0 clipped vectors | `uv run docqa ingest data/corpus/*.pdf` | per-file table; TOTAL 260 chunks | verified (recorded in `docs/CORPUS.md`) |
| 19 | Warm retrieval p95 is 2825.9 ms | `uv run python scripts/ablation.py` | latency table p95 2825.9 ms | verified (recorded in `docs/ABLATION.md`) |
| 20 | The gateway honours the forced `submit_answer` call | `bash scripts/live.sh probe --json` | `mode: "tool"`, `tool_call_returned: true`, 6.28 s; artifact `eval/structured-output.json` | verified 2026-10-23 |
| 21 | Six demo answers are precomputed with model and timestamp | `bash scripts/live.sh warm` | `data/demo-answers.json` holds exactly six questions, each with `model`, `generated_at`, `response` and `trace` | verified 2026-10-23 |
| 22 | The API serves a recorded answer without a model call | `curl -s localhost:8100/ask -H 'content-type: application/json' -d '{"question":"How many identical layers are in the Transformer encoder stack?"}'` | `cached: true`, `cache.model: openai-compatible:openai-fast`, `cache.generated_at`, 1 validated claim, trace | verified 2026-10-23 |
| 23 | The eval fits the endpoint's rate budget | `bash scripts/live.sh eval` | `run.wall_clock_s: 778.6`, `questions_measured: 30`, `model_calls: {requests: 26, retries: 0, rate_limited: 0}`, `blocked: null` | verified 2026-10-23 |
| 24 | Forcing generation measures the model, not the threshold | `bash scripts/live.sh eval --force-generation` | `run.gate_run: false`, 30 requests, 0 retries, 0 rate-limited, answer rate 8/24 (33.3%), exit 0 by design; artifact `eval-report-forced.json` | verified 2026-10-23 |
| 25 | The committed calibration is fitted on the gold questions with a recorded held-out check | `uv run python scripts/fit_calibration_gold.py` (local DB required) | `eval/calibration.json` has `provenance: gold-human-v1`, `fit_size: 30`, `positives: 24`, `negatives: 6`, two `held_out` folds at balanced accuracy 0.833, and `in_sample` marked as such; `tests/test_refusal.py` checks it | verified 2026-10-23 |

## What is deliberately not claimed

- The eval set is 30 questions, not the 100 in the plan. The intervals are wide
  and the ordering between vector-only, RRF and rerank is not statistically
  established.
- Citation precision and faithfulness are rule-based; no model judge ran. Their
  77.8% point estimates rest on n=9, so the Wilson interval reaches down to
  45.3%; they are not evidence of general precision.
- The committed calibration is fitted on the same 30 gold questions the eval
  runs, so run (c) is an in-sample test of the threshold. The held-out numbers
  in `eval/calibration.json` come from odd/even id-parity folds with only three
  unanswerable questions each.
- The answer-quality numbers are a snapshot of the free third-party endpoint on
  2026-10-23. It has no SLA and may change its backing model (`openai-fast`
  reported `gpt-oss-20b`) or its rate limit without notice.
- The corpus PDFs are not committed. `scripts/fetch-corpus` materialises them
  from `corpus.lock.json` before any ingest claim can be reproduced.

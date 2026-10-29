# Environments (P4)

Two environments: the build machine used for ingest, eval and the ablation, and
the hosted pair on Render and Neon. The hosted services read the variables in
"Hosted services"; the local defaults remain below.

| component | value |
|---|---|
| OS | Windows 11 Pro (10.0.26200), Git Bash |
| Python | CPython 3.12.15 (pinned in `.python-version`), managed by `uv` 0.13.5 |
| Package manager | `uv` only; `uv.lock` committed; no pip |
| Database | PostgreSQL 16.13 at `localhost:5433`, database `docqa` |
| Extension | pgvector 0.8.7, native Windows build (`CREATE EXTENSION vector`) |
| DSN | `postgresql+asyncpg://postgres@localhost:5433/docqa` |
| Schema | Alembic revision `0001_initial` (`uv run alembic upgrade head`; `docqa ingest` also applies it) |
| PDF parser | PyMuPDF **2.0** (pinned exactly in `pyproject.toml`) |
| Embedding | fastembed `BAAI/bge-small-en-v1.5`, 384-d, ONNX Runtime, CPU |
| Reranking | fastembed `Xenova/ms-marco-MiniLM-L-6-v2` (Apache-2.0), ONNX Runtime, CPU |
| Answer API | `uv run docqa serve` on `127.0.0.1:8788` (CLI default); `GET /health`, `GET /ready`, `POST /ask`, `GET /eval/report` |
| UI | `npm --prefix ui run dev` on `http://127.0.0.1:5173`; Vite proxies `/api` to `127.0.0.1:8788` |
| Screenshots | four 2880x1800 WebP files captured at 1440x900 CSS px with deviceScaleFactor 2, re-captured before the daylight theme; not yet re-captured for it |
| UI theme | "daylight reading room": cool page `#f6f7f9`, white panels, navy citations `#2f4f8f`, warm paper document `#fffdf8` |
| UI tests | `npm --prefix ui run e2e`: 4 Playwright tests pass — all three citations scroll and highlight, every recorded claim resolves with no render miss; `node ui/scripts/verify_demo_paper.mjs`: 9/9 demo claims resolve against the committed PDF |
| Judge model | `gemini-3.8-flash` pinned in `eval/judge.yaml`, different from the generator; reliability unverified |

## Hosted services

| service | value |
|---|---|
| API | <https://marginalia-api-4pvl.onrender.com> (`/health`, `/ready`, `/ask`) |
| UI | <https://marginalia-ui.onrender.com> |
| Render services | `marginalia-api` (`srv-dbho7udg1s2s73aj3kd0`), `marginalia-ui` (`srv-dbho81dg1s2s73aj3rlg`); branch `w2-b-hosting` |
| Plan | free: one 512 MB, 0.1-CPU API container; static UI; scale-to-zero after 15 minutes idle |
| Database | Neon PostgreSQL 16 + pgvector, database `docqa`, 5 documents / 260 chunks |

The free plan runs no pre-deploy command: `preDeployCommand` is ignored and the
blueprint declares none. Schema and corpus are an operator step from a machine
holding `DOCQA_DATABASE_URL`:

```sh
uv run alembic upgrade head          # or let `docqa ingest` apply it
uv run docqa ingest data/corpus/*.pdf
```

`GET /ready` returns 503 until the database reports the expected 5 documents,
260 chunks and the `0001_initial` revision.

### Variables the hosted services read

| variable | hosted value | meaning |
|---|---|---|
| `DOCQA_DATABASE_URL` | secret (`sync: false`) | Neon URL. `postgresql://…?sslmode=require` is normalised to `postgresql+asyncpg://…?ssl=require`; `channel_binding` is dropped because asyncpg rejects it. |
| `DOCQA_MODEL_API_KEY` | secret (`sync: false`) | Bearer token for the Gemini API; runtime only, never in the image. |
| `DOCQA_MODEL_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta/openai` | OpenAI-compatible endpoint. |
| `DOCQA_MODEL_NAME` | `gemini-flash-lite-latest` | Primary generator id. |
| `DOCQA_MODEL_FALLBACK_NAMES` | `gemini-3.8-flash` | First retry; after it fails the call fails loudly. |
| `DOCQA_MODEL_TIMEOUT_S` | `180` (default) | Per-request HTTP timeout. |
| `DOCQA_MODEL_MIN_INTERVAL_S` | `20` (default) | Minimum seconds between request starts; one call in flight. |
| `DOCQA_MODEL_MAX_RETRIES` | `2` (default) | Retries on 429/503/timeout before moving down the chain. |
| `DOCQA_MODEL_RETRY_BACKOFF_S` | `5` (default) | Base backoff; 5 s then 10 s, or `Retry-After` if larger. |
| `DOCQA_CORS_ORIGINS` | `https://marginalia-ui.onrender.com` | Browser origin allowed to call `/ask`; an unset value keeps the localhost origins. |
| `DOCQA_EMBEDDER` | `fastembed` | Fail instead of falling back to the hash embedder. |
| `DOCQA_RERANKER` | `fastembed` | Fail instead of falling back to lexical overlap. |
| `DOCQA_STRUCTURED_OUTPUT` | `tool` | Force the recorded forced-`submit_answer` mode; do not trust a stale probe artifact. |
| `DOCQA_ONNX_THREADS` | `1` | One ONNX thread; the default sizes arenas from the visible CPU count. |
| `DOCQA_RERANK_BATCH_SIZE` | `1` | One cross-encoder pair per batch; a batch of 20 measured +439 MB. |
| `DOCQA_ONNX_MEMORY_ARENA` | `0` | Do not keep ONNX CPU buffers alive after inference. |
| `DOCQA_ONNX_CONCURRENCY` | `1` | One ONNX call at a time; two concurrent cross-encoder calls crossed 512 MB. |
| `DOCQA_HF_CACHE` | `/app/.cache/fastembed` (image default) | Where the ONNX weights are baked into the image at build time. |
| `PORT` | injected by Render | The image binds `${PORT:-8788}`; the blueprint does not set it. |

The asyncpg pool is not an environment variable: `pool_size=2`,
`max_overflow=2`, `pool_recycle=300` and `pool_pre_ping=True` are fixed in
`src/docqa/db.py:52`. The default 5 + 10 pool would hold up to 15 connection
buffers for a process that answers one question at a time.

### What the free instance imposes

- 512 MB RAM and 0.1 CPU for the API. The ONNX models stay resident; the
  process-wide limiter is set to 1, so inference runs one call at a time.
- Scale-to-zero after 15 minutes idle; the first request after that waits about
  a minute. `GET /health` and `/ready` are the warm-up probes.
- No pre-deploy command and no persistent disk. The ONNX weights are baked into
  the image at build time; the corpus PDFs are fetched and hash-checked at
  build time, while the database is external (Neon).
- One instance, one uvicorn process. Ingest and eval are not served by the
  hosted API; they run from the build machine with `DOCQA_DATABASE_URL` set.

## Daylight reading-room palette and contrast

The Marginalia UI uses the light "daylight reading room" identity: page
`#f6f7f9`, panel `#ffffff`, raised panel `#eef1f5`, ink `#15181d`, muted
`#5c6370`, rule `#dfe3e9`, navy accent `#2f4f8f`, highlight
`rgba(47, 79, 143, 0.14)`, and warm paper `#fffdf8`. The PDF page is never
inverted or filtered; the paper tint is applied with
`mix-blend-mode: multiply`, not by altering the document itself.

Ratios are computed with the WCAG 2.x relative-luminance formula,
`(L_lighter + 0.05) / (L_darker + 0.05)`, not estimated:

| pair | ratio | requirement | result |
|---|---:|---:|---|
| ink `#15181d` on page `#f6f7f9` | 16.60:1 | >= 7:1 | pass |
| muted `#5c6370` on panel `#ffffff` | 6.05:1 | >= 4.5:1 | pass |
| navy `#2f4f8f` on page `#f6f7f9` (text) | 7.43:1 | >= 4.5:1 | pass |
| navy `#2f4f8f` on page `#f6f7f9` (UI) | 7.43:1 | >= 3:1 | pass |
| paper `#fffdf8` on page `#f6f7f9` | 1.05:1 | visible 1 px rule | pass: rule `#dfe3e9` on page is 1.20:1 |

Supporting pairs in the same theme: ink on panel 17.79:1, muted on page
5.64:1, muted on raised panel 5.34:1, navy on panel 7.97:1, navy on raised
panel 7.03:1, and panel-coloured text on navy 7.97:1. The paper page is the
only warm surface against the cool shell; its own 1 px rule keeps the page
edge legible without a shadow.

Source and trail type: source titles are 13 px ink, page labels
11 px mono muted, snippets 12 px muted, and the trail summary 12.5 px muted.
A selected source row sits on the 14% navy highlight, which composites to
`#e2e6ef` over the panel.

| pair | ratio | requirement | result |
|---|---:|---:|---|
| source title: ink `#15181d` on panel `#ffffff` | 17.79:1 | >= 4.5:1 | pass |
| page label: muted `#5c6370` on panel `#ffffff` | 6.05:1 | >= 4.5:1 | pass |
| snippet: muted `#5c6370` on panel `#ffffff` | 6.05:1 | >= 4.5:1 | pass |
| trail summary: muted `#5c6370` on panel `#ffffff` | 6.05:1 | >= 4.5:1 | pass |
| selected row: muted `#5c6370` on highlight `#e2e6ef` | 4.84:1 | >= 4.5:1 | pass |

Citation-mark placement was measured in Chromium at 1440x900 and 390x844:
0 of the 3 `sup` marks had a left edge within 2 px of its claim paragraph's
left edge, and the answer pane at 1440x900 measured `scrollHeight` 844 px
against `clientHeight` 844 px (no scroll).

## Embedding path used: real model

**This run used the real bge-small-en-v1.5 weights, not dummy embeddings.**
Hugging Face was reachable, fastembed downloaded the ONNX artifacts, and
`uv run docqa ingest` printed:

```
Embeddings: fastembed:BAAI/bge-small-en-v1.5 (384-d, ONNX CPU)
```

The model cache lives under the fastembed default
(`C:\Users\interview\AppData\Local\Temp\fastembed_cache` on this machine); set
`DOCQA_HF_CACHE` to relocate it. Inference runs in
`anyio.to_thread.run_sync` behind a process-wide semaphore (`Semaphore(2)` by
default; `DOCQA_ONNX_CONCURRENCY=1` on the hosted free instance), per invariant 7.
Vectors are L2-normalised before storage; the per-file `clipped-vector count`
reports vectors that had to be renormalised (0 for every file here).

### Dummy-embedding fallback

If weights cannot be fetched, `DOCQA_EMBEDDER=auto` (the default) falls back to
a deterministic blake2b feature-hash embedder (`dummy:blake2b-hash-384`). That
path is never silent:

- the CLI prints `WARNING: dummy embeddings ... vectors are NOT semantic`;
- every `documents.embedder` row stores `dummy:blake2b-hash-384`;
- this section of `docs/ENV.md` must be updated to say so.

Force it with `DOCQA_EMBEDDER=hash`, or make model failure fatal with
`DOCQA_EMBEDDER=fastembed`. The fallback uses a word/punctuation counter, not
the model tokenizer, so its 448-token cap is a structural guard rather than a
model-context guarantee.

## Reranker path used: real model

**The R2 ablation used the real `Xenova/ms-marco-MiniLM-L-6-v2` cross-encoder
weights, not the dummy scorer.** fastembed 0.9.0 does not export a cross-encoder
at the package top level, but the class exists at
`fastembed.rerank.cross_encoder.TextCrossEncoder`; the weights downloaded and
the harness printed:

```
Reranker: fastembed:Xenova/ms-marco-MiniLM-L-6-v2 (ONNX CPU)
```

The model cache is the same fastembed cache as the embedder (`DOCQA_HF_CACHE`
relocates it). Reranking runs in `anyio.to_thread.run_sync` behind the same
process-wide semaphore as embedding (`src/docqa/onnx.py`, invariant 7).

### Dummy-reranker fallback

`build_reranker(mode="auto")` catches import, download, and ONNX failures and
falls back to `dummy:lexical-overlap`, a deterministic query-term
coverage/frequency scorer. It is labelled in `RerankerChoice.banner` and in
`scripts/ablation.py`, and the ablation prints an explicit note when it is used.
It must never be reported as cross-encoder evidence. Force it with
`DOCQA_RERANKER=lexical` (read by the ablation harness) or make model failure
fatal with `DOCQA_RERANKER=fastembed`.

The reranker is the latency bottleneck: reranking 20–24 chunks costs roughly
2.1–2.7 s on this CPU, versus ~36 ms for query embedding and both SQL reads.
See `docs/ABLATION.md` for p50/p95.

## Database notes

- `documents` is keyed by `sha256`; `chunks` has a unique
  `(document_id, chunk_index)` constraint and `ON DELETE CASCADE`.
- Retrieval indexes are created with the schema: HNSW
  (`vector_cosine_ops`) on `chunks.embedding`, GIN on the generated
  `chunks.tsv` (`to_tsvector('english', text)`).
- Ingest is one transaction per document; a re-ingest deletes that document's
  chunks and reinserts the deterministic set, so counts stay stable.
- `psql -h localhost -p 5433 -U postgres -d docqa` for inspection.

## Commands used

```sh
scripts/fetch-corpus
uv run alembic upgrade head
uv run docqa ingest data/corpus/*.pdf
uv run docqa verify
uv run pytest
uv run ruff check src tests scripts
uv run mypy --strict src scripts/ablation.py
uv run python scripts/ablation.py
```

## Answer generation and structured output (R3, live endpoint)

The answer path reads its endpoint only from the
environment. No key is committed; `.env` is gitignored.

| variable | default | meaning |
|---|---|---|
| `DOCQA_MODEL_BASE_URL` | unset | OpenAI-compatible base URL; unset ⇒ extractive fallback |
| `DOCQA_MODEL_API_KEY` | unset | bearer token for the keyed endpoint |
| `DOCQA_JUDGE_API_KEY` | falls back to `DOCQA_MODEL_API_KEY` | bearer token for the judge probe |
| `DOCQA_MODEL_NAME` | `claude-sonnet-5-5` if unset | primary model id sent in the request body; `scripts/live.sh` and the hosted service set `gemini-flash-lite-latest` |
| `DOCQA_MODEL_FALLBACK_NAMES` | unset | comma-separated retry ids tried in order after the primary fails |
| `DOCQA_MODEL_TIMEOUT_S` | `180` | per-request HTTP timeout; a timeout is retried with backoff |
| `DOCQA_MODEL_MIN_INTERVAL_S` | `20` | minimum seconds between request **starts** to one endpoint (one call in flight) |
| `DOCQA_MODEL_MAX_RETRIES` | `2` | retries after HTTP 429/503 or a timeout before the call fails or moves down the model chain |
| `DOCQA_MODEL_RETRY_BACKOFF_S` | `5` | base backoff; retries wait 5 s then 10 s (or `Retry-After`, if larger) |
| `DOCQA_STRUCTURED_OUTPUT` | `auto` | `auto` = read the probe artifact; `tool`/`json` force a mode |
| `DOCQA_STRUCTURED_OUTPUT_PROBE` | `eval/structured-output.json` | recorded probe artifact |
| `DOCQA_DEMO_ANSWERS` | `data/demo-answers.json` | precomputed demo answers |
| `DOCQA_CALIBRATION` | `eval/calibration.json` | refusal calibration artifact |
| `DOCQA_FORCE_GENERATION` | `0` | **diagnostic only**: when `1`/`true`, `docqa eval`, `ask` and `warm` skip the calibrated low-score refusal so the model's own answer rate is measurable. Citation validation still runs. The API ignores it, and a forced eval is written to `eval-report-forced.json` with `gate_run: false` and exits 0 |

### Keyed endpoint used

| field | value |
|---|---|
| base URL | `https://generativelanguage.googleapis.com/v1beta/openai/` |
| primary model id | `gemini-flash-lite-latest` |
| first retry id | `gemini-3.8-flash` (then the run fails loudly) |
| auth | `Authorization: Bearer $DOCQA_MODEL_API_KEY`; the key lives in `P:/portfolio/.secrets/model.env`, never in a committed file or log |
| free-tier behaviour | one request at a time; request starts spaced by `DOCQA_MODEL_MIN_INTERVAL_S`; cold starts and demand spikes surface as 503 or a timeout, both retried with exponential backoff |
| probe | `bash scripts/live.sh probe --json` → `mode: "tool"`, `tool_call_returned: true`, 14.531 s; artifact `eval/structured-output.json` |

`scripts/live.sh` reads `P:/portfolio/.secrets/model.env` when present, strips
`\r` and surrounding quotes, exports `DOCQA_MODEL_API_KEY` and
`DOCQA_JUDGE_API_KEY`, and refuses with exit 2 when the key is absent. The
former keyless public endpoint was removed; it was a stand-in, not a fallback.

The client serialises every call and spaces request starts by at least 20 s.
HTTP 429/503 and timeouts are retried twice with 5 s / 10 s backoff; when the
primary id still fails, the documented retry id is tried with the same
discipline, and then the call fails loudly. There is no silent fallback to the
extractive model when a live endpoint is configured. The judge probe uses the
same endpoint, the same backoff and a different exact model id; if it cannot
answer, `provenance.judge.status` is `judge_model_unavailable`, the exact HTTP
status of each attempt is recorded, and the rule-based metric stays the
reported one.

### Measured run cost and demo cache

The 30-question keyed production eval took **710.3 s (11.84 min)** and made
**26 model requests** with **0 retries, 0 rate-limit failures and 0 server
errors**. The rate floor is 25 intervals × 20 s = 500 s; the remaining ~210 s is
provider latency. The gold-fitted threshold admits all 24 answerable questions
plus U1 and U6, so 26 calls are needed. The budget is recorded in
`eval-report.json` under `run`, with the earlier keyless gate run in
`eval-report-baseline.json` and the forced diagnostic in
`eval-report-forced.json`.

`bash scripts/live.sh warm` generated `data/demo-answers.json` with the six
demo questions. Of those, T1 and R3 are validated answers; S3, D3 and R2 are
`validation_failed` refusals (the model's quotes failed the citation contract);
U5 is a `low_score` refusal. Each entry carries the exact question text, the
model id recorded at generation time (the retired keyless model), and a
`generated_at` timestamp. The cache predates the keyed migration and the gold
calibration refit, so each cached response embeds the then-current calibration
snapshot (threshold `0.2552`); that is historical metadata, not the threshold now
loaded at boot. The API serves an exact-question hit with `cached: true` and
`cache: {model, generated_at}`, and a miss calls the live keyed endpoint. If the
provider still rate-limits after the documented retries, the API returns HTTP
429 `{"state": "rate_limited", …}` rather than a 500.

### Probe verdict (keyed endpoint)

`bash scripts/live.sh probe --json` against the keyed endpoint (the artifact
also records its own `checked_at`; abridged here):

```json
{
  "detail": "gateway returned the forced submit_answer call",
  "endpoint": "https://generativelanguage.googleapis.com/v1beta/openai",
  "json_mode_ok": null,
  "mode": "tool",
  "model": "gemini-flash-lite-latest",
  "tool_call_returned": true,
  "tool_choice_sent": true,
  "wall_clock_s": 14.531
}
```

The gateway honours the forced `submit_answer` call, so the recorded mode is
`tool`; JSON mode is not probed when tool mode already works. The artifact is
`eval/structured-output.json`, and `get_settings()` reads it only when
`DOCQA_STRUCTURED_OUTPUT=auto` and the artifact names the same endpoint and
model. A forced `tool`/`json` value in the environment always wins. If a future
endpoint ignores `tool_choice`, `docqa probe` tries JSON with a second real
request, records `mode: "json"`, and exits `5`. See
`docs/adr/0004-structured-output.md`.

### Live answer check

`bash scripts/live.sh ask --question "How many identical layers are in the Transformer encoder stack?" --json`
returned a validated answer against the keyed endpoint with the gold-fitted
threshold `0.6361`. The response carried both ids:

```
requested: gemini-flash-lite-latest
answered : gemini-flash-lite-latest
model    : openai-compatible:gemini-flash-lite-latest (retry gemini-3.8-flash)
Result   : answered with 1 validated claim(s)
Validation: returned=1 kept=1 dropped=0 schema_failures=0
```

The extractive fallback still exists and is used only when
`DOCQA_MODEL_BASE_URL` is unset; it is labelled `structured_output:
"extractive"` everywhere.

## Refusal calibration (R3)

`eval/calibration.json` holds `{a, b, threshold, fit_set_sha256}`. The threshold
is not typed by hand: `scripts/fit_calibration_gold.py` fits a question-level
Platt scaling `p = sigmoid(a * top_rerank_score + b)` by ridge-regularised
Newton-Raphson and picks the balanced-accuracy-optimal threshold. Commit
`eval/calibration_fit_set.json` along with it; the loader validates the hash's
shape at boot, and `tests/test_refusal.py` recomputes it from the fit set.

Current fit, `gold-human-v1`: the 30 hand-authored gold questions
(24 answerable, 6 unanswerable), one observation each, real
`Xenova/ms-marco-MiniLM-L-6-v2`:

```
a = 0.337571
b = 0.677505
threshold = 0.636077
fit_set_sha256 = f1a55cddcad2154aa67df9265f412d9e27b44e21fd42aac292158cd1cc580479
in-sample balanced accuracy = 0.833 (TPR 1.000, TNR 0.667)
held-out odd→even = 0.833, even→odd = 0.833 (3 negatives per fold)
```

The committed fit is **in-sample**: the fit set is the same 30 questions the
eval runs, so the 0.833 in-sample figure is optimistic and run (c) is not a
clean held-out test. The odd/even id-parity folds are the out-of-sample
evidence, and `eval/calibration.json` records both directions under `held_out`.
The equivalent raw-score cut is ≈ −0.353. A threshold fitted to the
cross-encoder's logit scale is not meaningful for `dummy:lexical-overlap`; do
not run the answer path with the dummy reranker and this artifact. The
provisional synthetic fitter, `scripts/fit_calibration.py`, is kept for
comparison but overwrites this artifact if run.

The loader refuses to start when the file is missing:

```
REFUSED: no calibration file at ...; run scripts/fit_calibration_gold.py or pass
--uncalibrated (recorded in every response)
```

`docqa serve --uncalibrated` boots without a score threshold (citation
validation still applies) and sets `"uncalibrated": true` plus
`"calibration": {"threshold": null, ...}` in every response.

## Commands added in R3

```sh
bash scripts/live.sh probe --json   # real request; records the mode
bash scripts/live.sh ask --question "…"
bash scripts/live.sh warm           # six demo answers → data/demo-answers.json
bash scripts/live.sh eval           # 30 questions; 710.3 s measured; 20 s between calls
bash scripts/live.sh eval --force-generation  # diagnostic: bypass low_score, exit 0

uv run docqa serve --port 8788            # POST /ask, GET /health, GET /ready, GET /eval/report
uv run docqa serve --uncalibrated         # explicit, recorded in responses
uv run python scripts/fit_calibration_gold.py   # refit on the human gold set (local DB)
uv run python scripts/fit_calibration.py        # provisional synthetic fit; overwrites the artifact
```

## k6 concurrency budget on this host

`scripts/k6-ask.js` gates full-response p95 at 5000 ms by default (CI keeps that number).
On this CPU-only host, with ONNX inference behind `Semaphore(2)`, four concurrent askers
measure **p95 7.92 s** (avg 6.02 s, max 8.14 s, 18 iterations, 4 VUs, 30 s), so a local
verify must state its own budget:

```sh
DOCQA_WITH_K6=1 DOCQA_K6_P95_MS=9000 bash scripts/verify.sh
```

The k6 step is **opt-in** (`DOCQA_WITH_K6=1`) because a loopback latency gate inherits
whatever else the machine is doing: two runs on this host measured p95 7.92 s and 9.22 s
for the same code, the second while a Playwright suite was running. A gate that fails
because another test was running is not a gate. The default `verify` is deterministic
(lint, types, 147 tests); the load number is measured on demand and quoted with its budget.

That is a measured CPU limitation, not a passing grade: the 5 s gate is crossed locally,
and it stays crossed until inference has a GPU or a smaller model.

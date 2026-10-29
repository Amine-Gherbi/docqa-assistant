# Proof

Every row maps a README claim to the command that reproduces it and the output
to expect. Numbers are repeated from `docs/EVAL.md` and `docs/ENV.md`; nothing
here is hand-tuned or estimated. Rows marked **unverified** were not run and say
why.

Build machine: Windows 11 Pro 10.0.26200 (Git Bash), Python 3.12.15,
PostgreSQL 16 + pgvector 0.8.7 on port 5433, ONNX Runtime on CPU.

## README claims

| # | Claim | Command | Expected output | Status |
|---|---|---|---|---|
| 1 | Every claim carries an in-context citation id and a verbatim quote; otherwise it is dropped | `uv run pytest tests/test_citations.py` | all tests pass; `validate_claims` at `src/docqa/answer.py:637` | verified |
| 2 | A chunk over the 448-token cap fails ingest instead of storing a truncated vector | `uv run pytest tests/test_chunking.py` | all tests pass; `HARD_CAP = 448` at `src/docqa/ingest/chunking.py:24` | verified |
| 3 | Ingest refuses a PDF whose SHA-256 differs from `corpus.lock.json` | `uv run docqa ingest --check` | five `OK <file>.pdf` lines, exit 0; `verify_file` at `src/docqa/ingest/lock.py:82` | verified |
| 4 | The refusal threshold is loaded from `eval/calibration.json` at boot and never hand-tuned | `uv run pytest tests/test_refusal.py`; read `eval/calibration.json` | tests pass; threshold `0.6361`, fit `gold-human-v1`; `load_calibration` at `src/docqa/answer.py:276` | verified |
| 5 | Every rate carries a Wilson 95% interval and a denominator; `stage_failed` is recorded per question | `uv run pytest tests/test_eval.py` | tests pass; `wilson_interval` at `src/docqa/eval.py:119`, stage counts at `:668` | verified |
| 6 | The judge is pinned to a different exact model id from the generator, is never substituted by the generator, and its reliability is unverified | `bash scripts/live.sh eval` (its judge probe) | judge `gemini-3.8-flash` against generator `gemini-flash-lite-latest`; `provenance.judge.status: judge_model_unavailable`, attempts `[429, 429, 503]`, `model_judge_used: false`; artifact `eval/judge-status.json` | verified (judge unavailable; κ not computed) |
| 7 | ONNX work runs in a worker thread behind a process-wide semaphore | `uv run pytest tests/test_embedder.py` | tests pass; `src/docqa/onnx.py:35`; `DOCQA_ONNX_CONCURRENCY=1` on the hosted free instance | verified |
| 8 | Production retrieval recall@8 is 18/24 (75.0%) | `bash scripts/live.sh eval` | `retrieval_recall_at_8` rate `0.750`, n=24, Wilson `[0.551, 0.880]`; artifact `eval-report.json` | verified (current artifact) |
| 9 | Production answer rate is 15/24 (62.5%) and the gate passes | `bash scripts/live.sh eval` | `answer_rate_answerable` rate `0.625`, n=24, Wilson `[0.427, 0.788]`, gate row `PASS` | verified (current artifact) |
| 10 | Unanswerable refusal is 6/6 (100.0%) | `bash scripts/live.sh eval` | `unanswerable_refusal_rate` rate `1.000`, n=6, Wilson `[0.610, 1.000]` | verified (current artifact) |
| 11 | Citation precision and faithfulness are 13/15 (86.7%) and 12/15 (80.0%), and are rule-based, not a model judge | `bash scripts/live.sh eval` | rates `0.867` and `0.800`, n=15, Wilson `[0.621, 0.963]` and `[0.548, 0.930]`; `provenance.judge.model_judge_used: false` | verified (current artifact) |
| 12 | In the historical forced-generation diagnostic, bypassing the threshold did not raise the answer rate | `bash scripts/live.sh eval --force-generation` | `run.gate_run: false`; answer rate 8/24 (33.3%) `[18.0%, 53.3%]`; exit 0 by design; artifact `eval-report-forced.json` | historical run, not re-measured in the keyed migration |
| 13 | A deliberately weakened vector-only retriever fails the recall gate | `uv run docqa eval --ablate vector-only` | recall@3 15/24 (62.5%) `[42.7%, 78.8%]` FAIL; exit 6; artifact `eval-report-ablation.json` | verified (committed artifact) |
| 14 | The production eval fits the keyed free tier's rate budget | `bash scripts/live.sh eval` | `run.wall_clock_s: 710.3`, 30 questions, `model_calls {requests: 26, retries: 0, rate_limited: 0, server_errors: 0, timeouts: 0}`; `docs/ENV.md` | verified (current artifact) |
| 15 | The gateway honours the forced `submit_answer` call | `bash scripts/live.sh probe --json` | `mode: "tool"`, `tool_call_returned: true`, `wall_clock_s: 14.531`; artifact `eval/structured-output.json` | verified |
| 16 | The generator is the keyed Gemini free tier, `gemini-flash-lite-latest`, with one request at a time and backoff on 429/503/timeouts | read `docs/ENV.md`; `bash scripts/live.sh eval` | `provenance.generator.endpoint_host: generativelanguage.googleapis.com`, `requested_models` starts `gemini-flash-lite-latest`, per-question `model_answered` recorded; `scripts/live.sh` refuses without the key | verified |
| 17 | The UI demo uses the committed Sentence-BERT paper and recorded quotes, not a live call; every citation scrolls, highlights and resolves | `node ui/scripts/verify_demo_paper.mjs`; `npm --prefix ui run e2e` | 9/9 recorded claims resolve against `ui/public/papers/sentence-bert.pdf`; 4 Playwright tests pass with no render miss | verified |
| 18 | The submitted screenshots are four 1440x900 CSS-px shots at 2x DPR in WebP, with seeded data | `uv run python scripts/capture-assets` | `[assets-mirror] valid: 4 WebP at 2880x1800`; manifest has `previewCommand` and no `demo` block | verified |
| 19 | Lint, strict types and tests pass; k6 is opt-in | `bash scripts/verify.sh` | ruff, `mypy --strict`, 147 tests pass; prints `SKIP k6`; final line `verify: PASS` | verified (k6 skipped) |
| 20 | The UI builds | `npm --prefix ui run build` | `tsc -b && vite build` exits 0 | verified |
| 21 | The hosted UI and API are live, and the hosted database holds the locked corpus | `curl -sS https://marginalia-api-4pvl.onrender.com/ready`; `curl -sS -o /dev/null -w '%{http_code}\n' https://marginalia-ui.onrender.com/` | `/ready` 200 with `documents: 5`, `chunks: 260`, `expected_documents: 5`, migration `0001_initial`; UI 200 | verified |
| 22 | The k6 concurrency scenario is green | `DOCQA_WITH_K6=1 bash scripts/verify.sh` | k6 p95 under the configured budget | **unverified** — k6 is not installed on this machine, so `verify.sh` prints `SKIP k6` |
| 23 | Generation is non-streaming: the API returns one validated JSON response and has no SSE route | `grep -rn "StreamingResponse\|text/event-stream" src/docqa` | no matches; `src/docqa/api.py:4` states the response is non-streaming | verified |

## Eval gates, current keyed run

**Generator:** `openai-compatible:gemini-flash-lite-latest` on the keyed Gemini
free tier, with `gemini-3.8-flash` as the documented retry.
**Embedder:** `fastembed:BAAI/bge-small-en-v1.5` (384-d, ONNX CPU).
**Reranker:** `fastembed:Xenova/ms-marco-MiniLM-L-6-v2` (ONNX CPU).
**Run:** 30 fixed questions (24 answerable, 6 unanswerable), 710.3 s, 26 model
requests, 0 retries, 0 rate limits, 0 server errors. Artifact: `eval-report.json`.

| gate | n | measured | Wilson 95% | gate |
|---|---:|---|---:|---|
| retrieval recall@8 | 24 | 18/24 (75.0%) | [55.1%, 88.0%] | PASS |
| answer rate | 24 | 15/24 (62.5%) | [42.7%, 78.8%] | PASS |
| unanswerable refusal | 6 | 6/6 (100.0%) | [61.0%, 100.0%] | PASS |
| citation precision* | 15 | 13/15 (86.7%) | [62.1%, 96.3%] | PASS |
| faithfulness* | 15 | 12/15 (80.0%) | [54.8%, 93.0%] | PASS |

\* rule-based, not a model judge. Rule: a citation is correct when it names the
gold chunk; a claim is faithful when every citation names the gold chunk and its
verbatim quote contains, or is contained in, the gold quote.

All five gates pass, so the command exits **0**. The refusal threshold was not
changed. Stage attribution over all 30 questions is `retrieval=5`, `fusion=0`,
`rerank=1`, `generation=0`, `schema=0`, `validation=4`; the four validation
failures are quote mismatches caused by the corpus's two-column extraction
order, not by the structured-output mode (see `docs/EVAL.md`). The
forced-generation diagnostic is **not** a gate run: it bypasses only the
calibrated low-score refusal, keeps the citation validator, records
`gate_run: false`, and exits 0 by design.

## Hosted pair

| item | value |
|---|---|
| API | <https://marginalia-api-4pvl.onrender.com> |
| UI | <https://marginalia-ui.onrender.com> |
| Render API service | `marginalia-api` (`srv-dbho7udg1s2s73aj3kd0`), live deploy `dep-dbhoumvlk1mc73dah260` at `df50ee85` |
| Render UI service | `marginalia-ui` (`srv-dbho81dg1s2s73aj3rlg`), live deploy `dep-dbhoun0n74is73elb3n0` at `df50ee85` |
| Deployed branch | `w2-b-hosting` |
| Plan | Render free: one 512 MB, 0.1-CPU API container, static UI, scale-to-zero after 15 minutes idle |
| Database | Neon PostgreSQL 16 + pgvector, database `docqa` |
| Corpus | 5 documents / 260 chunks; migration `0001_initial` |

The corpus PDFs are fetched and SHA-256-checked at image build
(`Dockerfile`: `RUN uv run python scripts/fetch-corpus && uv run docqa ingest --check`).
The hosted Neon database was populated by an **operator ingest** from a machine
holding `DOCQA_DATABASE_URL`:

```sh
# DOCQA_DATABASE_URL from P:/portfolio/.secrets/host.env
uv run docqa ingest data/corpus/*.pdf        # applies alembic upgrade head, then ingests
```

The free plan runs no pre-deploy command, so neither the migration nor the
ingest is part of the deploy; `/ready` reports both and returns 503 until they
match `corpus.lock.json`.

### Live checks

```sh
curl -sS https://marginalia-api-4pvl.onrender.com/health
# 200; "status":"ok","structured_output":"tool","uncalibrated":false

curl -sS https://marginalia-api-4pvl.onrender.com/ready
# 200; {"status":"ok","database":{"status":"ok","documents":5,"chunks":260},
#       "migration":{"current":"0001_initial","head":"0001_initial","ok":true},
#       "corpus":{"documents":5,"chunks":260,"expected_documents":5,"ok":true}}

curl -sS -o /dev/null -w '%{http_code}\n' https://marginalia-ui.onrender.com/
# 200

curl -sS https://marginalia-ui.onrender.com/ | grep -o '/assets/[^"]*\.js'
# /assets/index-Dxcar6HP.js
curl -sS https://marginalia-ui.onrender.com/assets/index-Dxcar6HP.js \
  | grep -o 'https://marginalia-api-4pvl.onrender.com/'
# https://marginalia-api-4pvl.onrender.com/
```

The deploy ids come from the Render API, with `RENDER_API_KEY` from the same
operator secrets file:

```sh
curl -sS -H "Authorization: Bearer $RENDER_API_KEY" \
  "https://api.render.com/v1/services/srv-dbho7udg1s2s73aj3kd0/deploys?limit=1"
# dep-dbhoumvlk1mc73dah260 | live | commit df50ee85

curl -sS -H "Authorization: Bearer $RENDER_API_KEY" \
  "https://api.render.com/v1/services/srv-dbho81dg1s2s73aj3rlg/deploys?limit=1"
# dep-dbhoun0n74is73elb3n0 | live | commit df50ee85
```

### `/ask` transcript

| field | value |
|---|---|
| question | `Why is SBERT more efficient than a BERT cross-encoder?` |
| HTTP | 200 |
| wall clock | 27.5 s |
| `cached` | `false` |
| claims | 1 |
| citations | 1 |
| `chunk_id` | 128 |
| `doc_id` | `sentence-bert` |
| `page_no` | 8 |
| `char_start`–`char_end` | 2613–4041 |
| `quote` | `SBERT is computationally efficient. On a GPU,` |
| quote check | normalised substring of chunk 128: NFKC maps the stored `ﬁ` ligature to `fi`; `normalise_text` at `src/docqa/answer.py:611` |
| `model_requested` / `model_answered` | `gemini-flash-lite-latest` / `gemini-flash-lite-latest` |

Reproduce the request:

```sh
curl -sS -X POST https://marginalia-api-4pvl.onrender.com/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"Why is SBERT more efficient than a BERT cross-encoder?"}'
```

The transcript above is the recorded measurement. The generator is
non-deterministic, so the same request exercises the same contract without
guaranteeing a byte-identical answer: a reproduction during the documentation
pass returned one validated claim citing chunk 127 on page 8 at 25.0 s, and two
other calls returned validated refusals (`validation_failed`,
`insufficient_context`). That is the citation contract working, not a different
build.

Verify the quoted span directly in the hosted database:

```sh
# DOCQA_DATABASE_URL from P:/portfolio/.secrets/host.env.
# This machine exports PGPORT=5433 for the local database; the Neon URL omits a
# port, so point psql at 5432 and put the options before the dbname.
PGPORT=5432 psql -Atc \
  "select id, page_no, char_start, char_end from chunks where id = 128" \
  "$DOCQA_DATABASE_URL"
# 128|8|2613|4041

PGPORT=5432 psql -Atc \
  "select strpos(normalize(text, NFKC), 'SBERT is computationally efficient. On a GPU,') > 0 \
   from chunks where id = 128" \
  "$DOCQA_DATABASE_URL"
# t
```

### 512 MB fix, measured

The release image and the Neon corpus were run under `--memory=512m`; the
measurements are recorded in commit `df50ee8` and can be printed with
`git show df50ee8 --format=%B -s`:

| boundary | RSS |
|---|---:|
| Python imports | 123 MB |
| embedder loaded | 291 MB |
| reranker loaded | 354 MB |
| retrieval | 358 MB |
| idle instance | 380 MB |
| rerank 20 passages in one ONNX batch | 796 MB (+439 MB) |
| rerank one pair at a time, CPU arena off | +31 MB, bit-identical scores |

The fix: load the reranker on the first retrieval instead of at boot; score one
pair at a time (`DOCQA_RERANK_BATCH_SIZE=1`); disable the ONNX CPU memory arena
(`DOCQA_ONNX_MEMORY_ARENA=0`); cap ONNX threads (`DOCQA_ONNX_THREADS=1`); allow
one ONNX call at a time (`DOCQA_ONNX_CONCURRENCY=1`); bound the asyncpg pool to
2 with 2 overflow and a 300 s recycle; report the corpus in `/ready`. Two
concurrent cross-encoder calls still crossed 512 MB, which is why the ONNX
limiter is 1 on the free instance. A warm `/ask` now returns in 17–25 s;
`/health` returns 200 and `/ready` reports the corpus.

## What is deliberately not claimed

- The eval set is 30 questions (24 answerable, 6 unanswerable), not the planned
  100. The intervals are wide and the ordering between vector-only, RRF and
  rerank is not statistically established.
- Citation precision and faithfulness are rule-based. Their 86.7% and 80.0%
  point estimates rest on n=15; the faithfulness interval reaches down to
  54.8% and the point estimate sits exactly on the 80% threshold.
- The committed calibration is fitted in-sample on the same 30 gold questions
  the eval runs. The held-out id-parity folds each contain only three
  unanswerable questions, so 0.833 is a direction, not a tight estimate.
- The answer-quality numbers are a snapshot of the keyed Gemini free tier,
  which has no SLA and returned HTTP 429/503 during the judge probe.
- The UI screenshots are seeded/offline: the demo answers are recorded and the
  cites resolve against the committed Sentence-BERT PDF. No live model call was
  made while capturing them.
- The hosted free instance scales to zero; a cold start waits about a minute,
  and the k6 concurrency scenario is unverified because k6 is not installed on
  this machine.
- The corpus PDFs are not committed. `scripts/fetch-corpus` materialises them
  from `corpus.lock.json` before any ingest claim can be reproduced.

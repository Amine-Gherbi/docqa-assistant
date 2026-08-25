# Local environment (P4)

Recorded 2026-10-23 on the build machine. No Docker and no hosted service.

| component | value |
|---|---|
| OS | Windows 11 Pro (10.0.26200), Git Bash |
| Python | CPython 3.12.15 (pinned in `.python-version`), managed by `uv` 0.13.5 |
| Package manager | `uv` only; `uv.lock` committed; no pip |
| Database | PostgreSQL 16.13 at `localhost:5433`, database `docqa` |
| Extension | pgvector 0.8.7, native Windows build (`CREATE EXTENSION vector`) |
| DSN | `postgresql+asyncpg://postgres@localhost:5433/docqa` |
| Schema | Alembic revision `0001_initial` (`uv run docqa db upgrade`) |
| PDF parser | PyMuPDF **2.0** (pinned exactly in `pyproject.toml`) |
| Embedding | fastembed `BAAI/bge-small-en-v1.5`, 384-d, ONNX Runtime, CPU |
| Reranking | fastembed `Xenova/ms-marco-MiniLM-L-6-v2` (Apache-2.0), ONNX Runtime, CPU |
| Judge/generator | not part of R1/R2 |

## Embedding path used: real model

**This run used the real bge-small-en-v1.5 weights, not dummy embeddings.**
Hugging Face was reachable, fastembed downloaded the ONNX artifacts on
2026-10-23, and `uv run docqa ingest` printed:

```
Embeddings: fastembed:BAAI/bge-small-en-v1.5 (384-d, ONNX CPU)
```

The model cache lives under the fastembed default
(`C:\Users\interview\AppData\Local\Temp\fastembed_cache` on this machine); set
`DOCQA_HF_CACHE` to relocate it. Inference runs in
`anyio.to_thread.run_sync` behind a process-wide `Semaphore(2)`, per invariant 7.
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
`fastembed.rerank.cross_encoder.TextCrossEncoder`; the weights downloaded on
2026-10-23 and the harness printed:

```
Reranker: fastembed:Xenova/ms-marco-MiniLM-L-6-v2 (ONNX CPU)
```

The model cache is the same fastembed cache as the embedder (`DOCQA_HF_CACHE`
relocates it). Reranking runs in `anyio.to_thread.run_sync` behind the same
process-wide `Semaphore(2)` as embedding (`src/docqa/onnx.py`, invariant 7).

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
uv run docqa db upgrade
uv run docqa ingest data/corpus/*.pdf
uv run docqa verify
uv run pytest
uv run ruff check src tests scripts
uv run mypy --strict src scripts/ablation.py
uv run python scripts/ablation.py
```

## Answer generation and structured output (R3)

The answer path reads its endpoint only from the environment. No key is
committed; `.env` is gitignored.

| variable | default | meaning |
|---|---|---|
| `DOCQA_MODEL_BASE_URL` | unset | OpenAI-compatible base URL (for example `https://gateway.example/v1`); unset ⇒ extractive fallback |
| `DOCQA_MODEL_API_KEY` | unset | bearer token; optional for keyless local gateways |
| `DOCQA_MODEL_NAME` | `claude-sonnet-5-5` | model id sent in the request body |
| `DOCQA_MODEL_TIMEOUT_S` | `60` | per-request HTTP timeout |
| `DOCQA_STRUCTURED_OUTPUT` | `tool` | `tool` = forced `submit_answer`; `json` = JSON-only prompt + `json-repair` |
| `DOCQA_CALIBRATION` | `eval/calibration.json` | refusal calibration artifact |

**This build used the extractive fallback.** No model endpoint was configured,
so `build_model` returned `extractive fallback (no model key)`. It selects
verbatim sentences that overlap the question, emits the normal answer payload,
and passes the same server-side citation validation. Every response labels it
(`model`, `structured_output: "extractive"`). It is deterministic, not a
language model, and is not evidence about generation quality.

### `tool_choice` probe verdict

`uv run docqa probe-tool-choice --json` was run on 2026-10-23. No hosted gateway
or key exists on this machine, so the probe started its own loopback
OpenAI-compatible stub and made a real HTTP request to it:

```json
{
  "detail": "client transmitted tool_choice and parsed a submit_answer call over real HTTP; no hosted gateway is configured, so the gateway itself is unverified",
  "endpoint": "local loopback stub",
  "honoured": true,
  "made_request": true,
  "tool_call_returned": true,
  "tool_choice_sent": true
}
```

Interpretation: the client puts `tool_choice={"type":"function","function":
{"name":"submit_answer"}}` on the wire and parses the returned tool call. **The
hosted gateway's `tool_choice` behaviour is unverified** — no endpoint was
contacted, and the "honoured: true" above describes the stub, not a proxy. If a
gateway is configured later, rerun the probe; exit code `5` means the gateway
rewrote the request and the deployment must use
`DOCQA_STRUCTURED_OUTPUT=json`. See `docs/adr/0004-structured-output.md`.

## Refusal calibration (R3)

`eval/calibration.json` holds `{a, b, threshold, fit_set_sha256}`. The threshold
is not typed by hand: `scripts/fit_calibration.py` fits a Platt scaling
`p = sigmoid(a * score + b)` by ridge-regularised Newton-Raphson and picks the
balanced-accuracy-optimal threshold. Commit `eval/calibration_fit_set.json` along
with it; the loader checks only the hash's shape at boot, and
`tests/test_refusal.py` recomputes the hash from the fit set.

Current fit (2026-10-23): 300 pairs (30 positive, 270 negative) from 30
machine-generated queries over the local 260-chunk corpus, hard negatives chosen
by reranker score, real `Xenova/ms-marco-MiniLM-L-6-v2`:

```
a = 0.573579
b = -4.849528
threshold = 0.255246
fit_set_sha256 = 52278766f27bd7bc78726da5b4a1afb6e2858e8a159abb9f0fca63dfe0275a21
mean balanced accuracy = 0.920
```

This is **provisional self-retrieval calibration**, recorded in the artifact's
`provenance` field. R5 must refit on the 100-question human-labelled gold set
(`uv run python scripts/fit_calibration.py` will need a gold-pair mode).
A score threshold fitted to the cross-encoder's logit scale is not meaningful
for `dummy:lexical-overlap`; do not run the R3 answer path with the dummy
reranker and this artifact.

The loader refuses to start when the file is missing:

```
REFUSED: no calibration file at ...; run scripts/fit_calibration.py or pass
--uncalibrated (recorded in every response)
```

`docqa serve --uncalibrated` boots without a score threshold (citation
validation still applies) and sets `"uncalibrated": true` plus
`"calibration": {"threshold": null, ...}` in every response.

## Commands added in R3

```sh
uv run docqa serve --port 8100          # POST /ask and GET /health
uv run docqa serve --uncalibrated       # explicit, recorded in responses
uv run docqa probe-tool-choice --json   # real request; gateway verdict
uv run python scripts/fit_calibration.py
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
(lint, types, 92 tests); the load number is measured on demand and quoted with its date.

That is a measured CPU limitation, not a passing grade: the 5 s gate is crossed locally,
and it stays crossed until inference has a GPU or a smaller model.

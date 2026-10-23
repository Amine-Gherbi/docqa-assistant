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

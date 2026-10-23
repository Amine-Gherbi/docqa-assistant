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
| Judge/generator | not part of R1 |

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
uv run ruff check
uv run mypy --strict src
```

# docqa-assistant

Ask questions about a PDF corpus and get answers whose claims point at highlighted spans.

Phase R1: ingest PDFs into PostgreSQL + pgvector as pinned, hash-checked chunks with an
inspectable retrieval-ready schema.

Phase R2: one hybrid retrieval function with vector and lexical candidates, RRF fusion,
cross-encoder reranking, and a committed ablation baseline.

See `docs/CORPUS.md` for the corpus provenance, `docs/ENV.md` for the local environment,
and `docs/ABLATION.md` for the retrieval baseline.

## Retrieval

`docqa.retrieval.retrieve(query, k=8, ...)` is the only retrieval entry point. Callers
depend on three small interfaces — `Embedder`, `CandidateSource`, and `Reranker` — so the
unit tests drive retrieval with a fake embedder and an in-memory source, with no network
and no database. `PgCandidateSource` is the only place that issues retrieval SQL.

The production pipeline:

1. vector top-20 by cosine distance (`pgvector` HNSW), and lexical top-20;
2. reciprocal rank fusion with `k=60`, deduplicated by `chunk_id` while keeping both
   ranks;
3. rerank the top-24 fused chunks with `Xenova/ms-marco-MiniLM-L-6-v2`;
4. return the top 8 with vector rank, lexical rank, RRF score, rerank score, and final
   rank attached.

The lexical half uses PostgreSQL `ts_rank_cd` over a generated `tsvector` — term
frequency plus cover density. **It is not BM25**: there is no `idf` term, no BM25
saturation, and no document-length normalisation.

Reproduce the ten-query ablation (vector-only vs hybrid+RRF vs +rerank) with:

```sh
uv run python scripts/ablation.py
```

## Commands

```sh
uv run docqa ingest data/corpus/*.pdf
uv run docqa verify
uv run pytest
uv run ruff check src tests scripts
uv run mypy --strict src
uv run python scripts/ablation.py
```

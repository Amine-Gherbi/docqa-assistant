# Retrieval ablation (R2)

Recorded on the build machine (Windows 11, 12 logical CPUs, ONNX CPU).

Corpus: the five pinned arXiv PDFs, **260 chunks**, real `BAAI/bge-small-en-v1.5`
vectors (384-d) in PostgreSQL 16 + pgvector 0.8, HNSW cosine index plus a GIN
index on the generated `tsvector`. Embedder and reranker banners printed by the
harness:

```
Embeddings: fastembed:BAAI/bge-small-en-v1.5 (384-d, ONNX CPU)
Reranker:  fastembed:Xenova/ms-marco-MiniLM-L-6-v2 (ONNX CPU)
```

Reproduce with:

```sh
uv run python scripts/ablation.py
```

## Configurations

| short name | what runs |
|---|---|
| vector-only | query embedding → vector top-8 by cosine distance. No lexical path, no fusion, no rerank. |
| hybrid+RRF | vector top-20 ⊕ lexical top-20 → RRF `k=60`, dedupe by `chunk_id` keeping both ranks → top-8. |
| hybrid+RRF+rerank | hybrid → top-24 by RRF → `Xenova/ms-marco-MiniLM-L-6-v2` cross-encoder (ONNX, CPU) → top-8. Production path. |

The lexical half uses PostgreSQL `ts_rank_cd(tsv, plainto_tsquery('english', q))`
— term frequency plus cover density. It is **not BM25**: there is no document
frequency / `idf` component and no BM25 saturation or length normalisation.

RRF contribution per list is `1 / (60 + rank)` with 1-based ranks. A chunk found
by both paths keeps `vector_rank`, `lexical_rank`, and the sum of both
contributions.

## Metric and labels

- **Hit@8** — the single gold chunk is among the eight returned chunks.
- **MRR@8** — mean reciprocal rank of the gold chunk when it is in the top 8
  (`0` otherwise).
- **n = 10** fixed queries, one hand-labelled gold chunk each.
- Hit rates are printed with a **Wilson 95% confidence interval** and the
  denominator. MRR is a mean, so no binomial interval is claimed for it.

Gold chunks are identified by `(doc_id, chunk_index)`. `chunk_index` is
deterministic for a given PDF and chunker, so the labels survive re-ingest even
though row ids do not. These are coarse single-relevance labels authored for
this ablation, not a graded relevance set; R5's labelled eval remains the
authoritative quality measurement. Two candidate queries were discarded while
authoring because PDF hyphenation split the answer across two adjacent chunks
(`default` became `de-` / `fault`), which a single-gold label cannot represent.

## Per-query rank of the gold chunk

`miss` = gold chunk outside the top 8.

| # | query | gold (doc #chunk) | vector-only | hybrid+RRF | +rerank |
|---|---|---|---:|---:|---:|
| 1 | Which optimizer and learning-rate warmup schedule trained the Transformer? | attention-is-all-you-need #16 | 1 | 1 | 1 |
| 2 | What sinusoidal positional encodings does the Transformer use? | attention-is-all-you-need #12 | 1 | 2 | 1 |
| 3 | How many parallel attention heads does multi-head attention use? | attention-is-all-you-need #8 | 1 | 1 | 1 |
| 4 | What percentage of WordPiece tokens does BERT mask during pre-training? | bert #12 | 6 | 1 | 3 |
| 5 | What is the next sentence prediction objective used for? | bert #4 | 1 | 1 | 2 |
| 6 | How does DPR compute similarity between a question and a passage? | dense-passage-retrieval #8 | 1 | 1 | 1 |
| 7 | What are in-batch negatives used for during training? | dense-passage-retrieval #12 | 1 | 1 | 1 |
| 8 | How does Sentence-BERT build sentence embeddings at inference time? | sentence-bert #7 | miss | miss | 3 |
| 9 | Which pre-trained retriever initialises RAG's document index? | retrieval-augmented-generation #9 | miss | miss | 7 |
| 10 | What does RAG use as its non-parametric memory? | retrieval-augmented-generation #4 | 8 | miss | 5 |

## Aggregate

| stage | Hit@8 | 95% Wilson CI | MRR@8 | n |
|---|---:|---|---:|---:|
| vector-only | 8/10 (80%) | [49%, 94%] | 0.629 | 10 |
| hybrid+RRF | 7/10 (70%) | [40%, 89%] | 0.650 | 10 |
| hybrid+RRF+rerank | 10/10 (100%) | [72%, 100%] | 0.651 | 10 |

## Reading the result

- The three CIs overlap heavily, so this table does **not** establish a
  statistically significant ordering. Treat it as a sanity check and a
  reproducible baseline for R5.
- Reranking changes Hit@8 from 7/10 to 10/10 by recovering two semantic matches
  that fusion placed outside the top 8 (queries 8 and 9). Gold ranks of 3 and 7
  are still not strong.
- RRF is not monotonically better at n=10. Query 10 is ranked 8th by vector
  alone but falls out of the top 8 after fusion, because the lexical list pulls
  in chunks that appear in both lists; the cross-encoder restores it to 5th.
  This is a genuine precision/recall trade-off, not a bug.
- Most queries are easy: nine gold chunks are in the top 8 under at least one
  stage and five are rank 1 in every stage.

## Latency

Production path, 20 warm calls after one warm-up, same query and corpus:

| statistic | latency |
|---|---:|
| p50 | 2394.6 ms |
| p95 | **2825.9 ms** |
| max | 3236.6 ms |

Command: `uv run python scripts/ablation.py`.

Stage breakdown on a warm call (measured separately): query embedding ≈ 26 ms,
vector SQL ≈ 8 ms, lexical SQL ≈ 2 ms, **rerank of 20–24 chunks ≈ 2.1–2.7 s**.
The cross-encoder dominates. `Xenova/ms-marco-MiniLM-L-6-v2` is run through
`onnxruntime` on CPU; explicit thread counts (1/4/8/12) on this machine did not
beat the default by more than noise, so no thread count is pinned. The latency
number is therefore a property of this Windows/CPU build, not of the retrieval
logic; ONNX work is still bounded by the shared `Semaphore(2)`.

## What is not measured here

- Graded relevance, answer correctness, citation validity, and refusal
  calibration — all R5 work; no such labels exist yet.
- Effect of the dummy hash embedder or the dummy lexical-overlap reranker. Both
  paths exist, but this table uses the real weights for both models.

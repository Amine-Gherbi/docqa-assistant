# Corpus provenance (P4)

The corpus is five real arXiv papers about retrieval, embeddings, and language
models. The PDFs are **not committed**. `corpus.lock.json` is the contract: it
pins each file's byte length and SHA-256, and both `scripts/fetch-corpus` and
`uv run docqa ingest` refuse anything that does not match.

- Lock: [`corpus.lock.json`](../corpus.lock.json)
- Materialise and verify: `scripts/fetch-corpus` (or `uv run docqa verify`)
- Destination: `data/corpus/` (git-ignored)
- Retrieved from `arxiv.org`; every file is pinned in `corpus.lock.json`.
- Extraction: PyMuPDF **2.0**, `page.get_text("text", sort=True)`; every page's
  exact string is hashed into `chunks.page_text_sha256`.
- Licence evidence: each arXiv abstract page was checked at retrieval; the
  licence column below is what that page declared.

## Files

| doc_id | source | licence | licence URL | bytes | pages | sha256 |
|---|---|---|---|---:|---:|---|
| attention-is-all-you-need | arXiv:1706.03762v7 — Attention Is All You Need | arXiv.org perpetual, non-exclusive distribution licence 1.0 | https://arxiv.org/licenses/nonexclusive-distrib/1.0/ | 2215244 | 15 | `bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697` |
| bert | arXiv:1810.04805v2 — BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | arXiv.org perpetual, non-exclusive distribution licence 1.0 | https://arxiv.org/licenses/nonexclusive-distrib/1.0/ | 775166 | 16 | `5692a5514787a8c6727b4ff3b726a3385798bc68e12138d1d4af83947e2acf6e` |
| sentence-bert | arXiv:1908.10084v1 — Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks | CC BY-SA 4.0 | https://creativecommons.org/licenses/by-sa/4.0/ | 548961 | 11 | `92366d22fc0521350d6cbe0ad72bcee754f90e88181d7b4a8e2adb1e7e594fec` |
| dense-passage-retrieval | arXiv:2004.04906v3 — Dense Passage Retrieval for Open-Domain Question Answering | arXiv.org perpetual, non-exclusive distribution licence 1.0 | https://arxiv.org/licenses/nonexclusive-distrib/1.0/ | 383508 | 13 | `3e67fc1a9977715acf722d85f0b7d124b03715e975f6310a4d6c12094190912f` |
| retrieval-augmented-generation | arXiv:2005.11401v4 — Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | arXiv.org perpetual, non-exclusive distribution licence 1.0 | https://arxiv.org/licenses/nonexclusive-distrib/1.0/ | 885323 | 19 | `23e3249e9a1e75418d82efecab0ea8c4d033b89c93742f63208d47ce01f21233` |

No synthetic fixtures were needed: all five PDFs and both network paths
(arXiv and Hugging Face) were reachable on this machine.

## Ingest result (real embeddings)

`uv run docqa ingest data/corpus/*.pdf`:

| file | pages | chunks | p50 tok | p95 tok | max tok | clipped vectors |
|---|---:|---:|---:|---:|---:|---:|
| 1706.03762v7.pdf | 15 | 35 | 330 | 376 | 430 | 0 |
| 1810.04805v2.pdf | 16 | 64 | 345 | 376 | 376 | 0 |
| 1908.10084v1.pdf | 11 | 41 | 367 | 375 | 376 | 0 |
| 2004.04906v3.pdf | 13 | 53 | 359 | 376 | 376 | 0 |
| 2005.11401v4.pdf | 19 | 67 | 330 | 376 | 376 | 0 |
| **total** | **74** | **260** | | | | **0** |

Chunks are page-scoped, 320–420 tokens where the page arithmetic allows, and
never above the 448-token hard cap. Short chunks occur only on short pages and
page tails; making every chunk 320+ would require crossing page boundaries,
which the citation contract forbids. Re-ingesting the same files produced the
same 260 chunk slots with no duplicate `(document_id, chunk_index)` rows.

# Gold set for the P4 evaluation harness

`gold.jsonl` is the fixed, hand-authored question set used by `uv run docqa eval`.
It is committed; the corpus PDFs are not (they are pinned by `corpus.lock.json`).

## Size, and why it is 30 rather than 100

**30 questions: 24 answerable and 6 unanswerable** (2 near-miss, 2 out-of-corpus,
1 under-specified, 1 ambiguous).

A 100-question set was not labelled in this session. Each answerable label
required locating a fact in the actual ingested text and recording the exact
quoting span from that chunk; doing that 100 times without inventing labels was
not possible honestly here. The 24 answerable questions span all five documents
(Transformer 5, BERT 5, Sentence-BERT 4, DPR 5, RAG 5) and the unanswerable
questions cover the four required failure modes.

`src/docqa/eval.py` refuses to run if the file stops matching this shape:
24 answerable plus exactly `{near_miss: 2, out_of_corpus: 2, under_specified: 1,
ambiguous: 1}`. A future reduction must be documented here first.

## Label provenance

Every answerable label was authored by reading the ingested chunk text and
copying a **verbatim span** out of it. The labels are not model-generated and no
gold span was placed from memory. Before any measurement, `resolve_gold` checks
each label against the live database:

- the named `(doc_id, chunk_index)` must exist,
- `page_no` must match the stored chunk,
- the quote must be a normalised substring of the stored chunk text.

The harness aborts if any of those checks fail. The BERT, Sentence-BERT and DPR
papers are two-column PDFs whose extracted line order interleaves the columns, so
several gold quotes are deliberately short fragments that remain contiguous in
the stored text. They still contain the answer; they do not pretend to be a
whole sentence when the extractor did not preserve one.

Unanswerable labels carry `"gold": null` and a `note` explaining the label.
The two near-miss labels were checked by searching every chunk of the relevant
document for the missing attribute on 2026-10-23 (`latency` / `per sentence` for
the Transformer; `learning rate` / `dropout` for RAG).

## Judge provenance

No judge model is reachable in this build: `DOCQA_JUDGE_BASE_URL` is unset and
`eval/judge.yaml` does not exist. Citation precision and faithfulness are
therefore computed by a deterministic rule-based checker and are labelled
`rule-based, not a model judge` in both the console output and
`eval-report.json`. They are not presented as model-judge scores.

## Row format

```json
{"id": "T1", "question": "...", "kind": "answerable",
 "gold": {"doc_id": "...", "page_no": 3, "chunk_index": 5, "quote": "..."}}
{"id": "U1", "question": "...", "kind": "near_miss", "gold": null, "note": "..."}
```

`kind` is one of `answerable`, `near_miss`, `out_of_corpus`, `under_specified`,
`ambiguous`.

## Running

```sh
uv run docqa eval
uv run docqa eval --ablate vector-only
```

The first writes `eval-report.json` and exits non-zero if a gate fails. The
second writes `eval-report-ablation.json` and is the deliberately weakened
vector-only/top-3 retriever required to miss a gate.

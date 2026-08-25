# Evaluation (R5)

**Question set:** 30 fixed questions — 24 answerable and 6 unanswerable (2
near-miss, 2 out-of-corpus, 1 under-specified, 1 ambiguous). **Label
provenance:** every answerable gold span was hand-copied from the ingested chunk
text (5 papers, 260 chunks) and is re-verified as a normalised substring of the
named `(doc_id, chunk_index, page_no)` before measurement; no label is
model-generated. **Judge provenance:** no judge model is reachable in this
build, so citation precision and faithfulness come from a deterministic
rule-based checker and are labelled **rule-based, not a model judge**. The set
is 30 rather than 100 because 100 spans could not be labelled honestly in one
session; see `eval/README.md` for the reduction and the guard that enforces it.

Generated 2026-10-23 on the build machine (Windows 11, ONNX CPU) against the
real corpus: `fastembed:BAAI/bge-small-en-v1.5` embeddings and
`fastembed:Xenova/ms-marco-MiniLM-L-6-v2` reranker. Generator: the labelled
extractive fallback (`no model key`). Calibration:
`eval/calibration.json`, threshold `0.2552`, fit-set
`52278766…5a21`, provenance `provisional-self-retrieval-v1` (not refitted on
this 30-question gold set — see the limitation below).

Reproduce:

```sh
uv run docqa eval                      # production path → eval-report.json
uv run docqa eval --ablate vector-only  # weakened path → eval-report-ablation.json
```

The command exits **6** when any gate fails. It currently exits 6: three
production gates fail, and the report says so rather than hiding it.

## Metrics

All rates carry a Wilson 95% interval and their denominator `n`.

| metric | definition |
|---|---|
| retrieval recall | the gold chunk is in the returned top-k for an answerable question |
| answer rate | an answerable question returns ≥1 validated claim |
| unanswerable refusal rate | an unanswerable question returns `insufficient_evidence=true` |
| citation precision (rule-based) | a citation is correct when it names the gold chunk |
| faithfulness (rule-based) | every citation on a claim names the gold chunk and its verbatim quote contains, or is contained in, the gold quote |
| `stage_failed` | earliest stage that lost the span/answer: `retrieval`, `fusion`, `rerank`, `generation`, `schema` |

Rules are enforced per metric: retrieval recall ≥ 75%, answer rate ≥ 50%,
unanswerable refusal ≥ 80%, citation precision ≥ 80%, faithfulness ≥ 80%. An
empty denominator fails a gate; a gate never passes vacuously.

## Ablation table

Retrieval rows use the same 24 answerable questions; the full answer row adds
the calibrated refusal and the extractive generator. `n` is the denominator of
the rate in that cell, not the number of configurations.

| configuration | n | metric | result | 95% Wilson CI |
|---|---:|---|---:|---|
| vector-only top-8 | 24 | recall@8 | 17/24 (70.8%) | [50.8%, 85.1%] |
| + lexical / RRF top-8 | 24 | recall@8 | 17/24 (70.8%) | [50.8%, 85.1%] |
| + rerank top-8 | 24 | recall@8 | 18/24 (75.0%) | [55.1%, 88.0%] |
| + calibrated refusal (full path) | 24 | answer rate | 5/24 (20.8%) | [9.2%, 40.5%] |
| + calibrated refusal (full path) | 6 | unanswerable refusal | 6/6 (100%) | [61.0%, 100%] |
| + calibrated refusal (full path) | 15 | citation precision* | 3/15 (20.0%) | [7.0%, 45.2%] |
| + calibrated refusal (full path) | 15 | faithfulness* | 2/15 (13.3%) | [3.7%, 37.9%] |

\* rule-based, not a model judge.

Stage attribution over all 30 questions: `retrieval=5`, `fusion=0`, `rerank=1`,
`generation=13`, `schema=0`. Live answerable misses were S1, S2, S4, D4, D5
(not retrieved by either candidate path) and R4 (reached fusion, dropped by
rerank). The remaining 13 answerable failures are `generation`: the calibrated
refusal rejected them before the extractive model ran.

## The weakened retriever must fail a gate

`--ablate vector-only` lowers the retriever to vector-only, top-3. It measures
the same 24 answerable questions:

| configuration | metric | result | 95% Wilson CI | gate |
|---|---|---:|---|---|
| production (hybrid+RRF+rerank, top-8) | recall@8 | 18/24 (75.0%) | [55.1%, 88.0%] | **PASS** (≥75%) |
| weakened (vector-only, top-3) | recall@3 | 15/24 (62.5%) | [42.7%, 78.8%] | **FAIL** (<75%) |

The ablation also fails the unanswerable-refusal gate: vector retrieval carries
no rerank score, so the calibrated threshold cannot fire and only 1/6
unanswerable questions is refused ([3.0%, 56.4%] vs the 80% gate). Its answer
rate rises to 24/24 ([86.2%, 100%]) and citation precision falls to 15/85
(17.6%, [11.0%, 27.1%]) — answering more often while grounding less. Four gates
fail under ablation versus three in production; the retrieval gate is the
controlled comparison because it changes for that reason alone.

## Reading the production result honestly

- **Retrieval passes at the threshold, barely.** 18/24 = 75.0% is exactly the
  gate. The interval reaches down to 55.1%, and the vector-only and hybrid rows
  are identical at n=24, so this set does not establish that RRF or rerank is
  reliably better. Rerank recovers one question (R4); fusion recovers none.
- **The refusal gate passes.** All six unanswerable questions are refused
  (`low_score`), including both near-misses and both out-of-corpus questions.
- **The answer-rate, citation-precision and faithfulness gates fail.** Only
  5/24 answerable questions are answered. Of 15 claims that reach the user, 3
  cite the gold chunk and 2 quote a span overlapping the gold span. This is the
  measured quality of the extractive fallback under the current calibration,
  not a model-judge score.
- **Measured limitation — calibration refit is still owed.** The threshold
  `0.2552` was fitted on machine-generated self-retrieval sentences, not on
  these human questions. On real question phrasing the top reranker score maps
  to `p ≈ 0.02–0.10`, below the threshold, so 19/24 answerable questions are
  refused at the `low_score` gate before generation. The threshold was **not**
  hand-tuned to make the gate pass; invariant 4 forbids that. R5's remaining
  action is to refit `eval/calibration.json` on a larger human gold set (this
  build's 30 questions are enough to expose the gap, not to fit a stable
  threshold) and then re-run this harness unchanged.
- **Ordering is not statistically established.** CIs overlap; n=24 is a sanity
  check. The harness reports the numbers it can defend and refuses to claim
  more.

## Files

| file | contents |
|---|---|
| `eval/gold.jsonl` | 30 fixed questions; span-level gold for the 24 answerable ones |
| `eval/README.md` | size rationale, label provenance, judge provenance, row format |
| `eval-report.json` | production per-question rows, aggregate metrics, retrieval ablation, gates |
| `eval-report-ablation.json` | vector-only/top-3 weakened run, same schema |
| `src/docqa/eval.py` | loader, gold/corpus verification, Wilson intervals, stage attribution, gates, runner |
| `tests/test_eval.py` | golden tests for intervals, gold shape, stage attribution, and gate failure |

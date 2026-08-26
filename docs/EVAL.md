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
`fastembed:Xenova/ms-marco-MiniLM-L-6-v2` reranker. Generator: the live, keyless
`openai-compatible:openai-fast` endpoint
(`https://text.pollinations.ai/openai`), forced `submit_answer` tool call; the
probe verdict is in `docs/ENV.md`. Calibration: `eval/calibration.json`,
`gold-human-v1` (fitted on all 30 gold questions — see the in-sample caveat
below), threshold `0.6361` on the probability scale (≈ −0.353 on the raw
reranker score), fit-set `f1a55cdd…0479`.

Reproduce:

```sh
bash scripts/live.sh eval                      # production path → eval-report.json
bash scripts/live.sh eval --force-generation   # diagnostic only → eval-report-forced.json
bash scripts/live.sh eval --ablate vector-only  # weakened path → eval-report-ablation.json
```

`scripts/live.sh` exports only the public endpoint and model id; it contains no
secret. Without `DOCQA_MODEL_BASE_URL` the same harness uses the labelled
extractive fallback (that is how the committed `eval-report-ablation.json` was
produced in R2).

The command exits **6** when any gate fails. It currently exits 6: **three of
the five production gates fail** (answer rate, citation precision,
faithfulness). The final production run took **778.6 s (12.98 min)** wall clock,
measured 30 questions, and made **26 model requests with 0 retries and 0
rate-limit failures**. The numbers are reported, not tuned away.

## Three runs on 2026-10-23

All three use the same 30 questions, corpus, embedder and reranker. The endpoint
allows about one request per 30 s, so the runs were serialised.

| run | UTC window | threshold | purpose | wall clock | model requests | answer rate (24) | unanswerable refusal (6) | citation precision* | faithfulness* | gates | exit |
|---|---|---|---:|---|---:|---:|---|---|---|---|---:|
| (a) baseline | 17:43:50–17:46:19 | `0.2552` (provisional) | gate run before the refit | 148.1 s | 5 | 2/24 (8.3%) [2.3%, 25.8%] | 6/6 (100%) [61.0%, 100%] | 2/2 (100%) [34.2%, 100%] | 2/2 (100%) [34.2%, 100%] | 4 PASS / 1 FAIL | 6 |
| (b) forced diagnostic | 17:52:55–18:08:12 | `0.6361` bypassed | measure the model's own answer rate | 916.4 s | 30 | 8/24 (33.3%) [18.0%, 53.3%] | 5/6 (83.3%) [43.6%, 97.0%] | 7/10 (70.0%) [39.7%, 89.2%] | 7/10 (70.0%) [39.7%, 89.2%] | 3 FAIL (diagnostic only) | 0 by design |
| (c) final production | 18:09:38–18:22:37 | `0.6361` | gate run after the refit | 778.6 s | 26 | 8/24 (33.3%) [18.0%, 53.3%] | 6/6 (100%) [61.0%, 100%] | 7/9 (77.8%) [45.3%, 93.7%] | 7/9 (77.8%) [45.3%, 93.7%] | 2 PASS / 3 FAIL | 6 |

\* rule-based, not a model judge. Run (b) is **not a gate run**: it bypasses
only the calibrated low-score refusal (`DOCQA_FORCE_GENERATION=1` /
`--force-generation`), the citation validator still runs, and the report records
`run.gate_run: false`. Its report is `eval-report-forced.json`; run (a) is
`eval-report-baseline.json`; run (c) is `eval-report.json`.

## What the forced diagnostic showed

Run (b) answers the question this card existed to ask: **is the free model
unable to answer, or is the refusal threshold doing the refusing?** Both were
true, in different proportions.

- With the threshold bypassed, the model still produced a contract-valid answer
  for only **8/24 answerable questions (33.3%, [18.0%, 53.3%])**. The threshold
  is no longer the binding constraint after the refit: it admits all 24
  answerable questions (in-sample), and run (c) reaches the same 8/24.
- Of the 16 answerable questions that did not answer, **13 model replies did make
  a claim, but the quote was not a verbatim substring of the cited chunk**, so
  the validator correctly dropped it (B1–B5, S3, S4, D1–D3, D5, R2, R5). This
  is the dominant failure, and it is a model/protocol failure, not a retrieval
  failure: 11 of those 13 had the gold chunk in the top-8.
- Two replies (S2, D4) declined with `insufficient_context`; one call (S1)
  returned no `submit_answer` tool call.
- Five questions (S1, S2, S4, D4, D5) had already lost the gold chunk at
  retrieval, so they could not have been answered from the context.

The unanswerable side moved the other way. With the new threshold, run (c)
admits two unanswerable questions (U1, U6) to the model instead of refusing
them at `low_score`, but the model or validator then refuses both — U1 with
`insufficient_context`, U6 with `validation_failed` — so the refusal rate stays
6/6. The in-sample calibration alone refuses 4/6 (U2–U5); the other two are
refused downstream. This is stated as measured rather than presented as a
calibration win.

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

Retrieval rows use the same 24 answerable questions. The answer rows come from
run (c), the final production run; the forced row is a labelled diagnostic and
is not a production result.

| configuration | n | metric | result | 95% Wilson CI |
|---|---:|---|---:|---|
| vector-only top-8 | 24 | recall@8 | 17/24 (70.8%) | [50.8%, 85.1%] |
| + lexical / RRF top-8 | 24 | recall@8 | 17/24 (70.8%) | [50.8%, 85.1%] |
| + rerank top-8 | 24 | recall@8 | 18/24 (75.0%) | [55.1%, 88.0%] |
| + calibrated refusal + live generator (full path) | 24 | answer rate | 8/24 (33.3%) | [18.0%, 53.3%] |
| + calibrated refusal + live generator (full path) | 6 | unanswerable refusal | 6/6 (100%) | [61.0%, 100.0%] |
| + calibrated refusal + live generator (full path) | 9 | citation precision* | 7/9 (77.8%) | [45.3%, 93.7%] |
| + calibrated refusal + live generator (full path) | 9 | faithfulness* | 7/9 (77.8%) | [45.3%, 93.7%] |
| forced generation (diagnostic, not a gate run) | 24 | answer rate | 8/24 (33.3%) | [18.0%, 53.3%] |

\* rule-based, not a model judge.

Stage attribution over all 30 questions in run (c): `retrieval=5`, `fusion=0`,
`rerank=1`, `generation=0`, `schema=11`. The five retrieval failures are S1, S2,
S4, D4 and D5 (the gold chunk is absent from the reranked top-8); R4 is the
single rerank loss — the gold chunk was not in context, yet the model answered
with two citations to other chunks, which is why precision is 7/9 rather than
8/8. The eleven schema failures are B1–B5, S3, D1–D3, R2 and R5: the model ran,
but every claim it returned failed the citation contract. The `generation=0`
count differs from run (a)'s `generation=13` because the refitted threshold no
longer refuses answerable questions before generation; the refusals that remain
in run (c) are four unanswerable questions at `low_score`.

## The weakened retriever must fail a gate

`--ablate vector-only` lowers the retriever to vector-only, top-3. It measures
the same 24 answerable questions:

| configuration | metric | result | 95% Wilson CI | gate |
|---|---|---:|---|---|
| production (hybrid+RRF+rerank, top-8) | recall@8 | 18/24 (75.0%) | [55.1%, 88.0%] | **PASS** (≥75%) |
| weakened (vector-only, top-3) | recall@3 | 15/24 (62.5%) | [42.7%, 78.8%] | **FAIL** (<75%) |

The ablation also fails the unanswerable-refusal gate: vector retrieval carries
no rerank score, so the calibrated threshold cannot fire and only 1/6
unanswerable questions is refused ([3.0%, 56.4%] vs the 80% gate). In the R2
extractive run its answer rate rose to 24/24 ([86.2%, 100%]) and citation
precision fell to 15/85 (17.6%, [11.0%, 27.1%]) — answering more often while
grounding less. Those answer-path rows are labelled with their generator in
`eval-report-ablation.json`; the retrieval gate is the controlled comparison
because it is generator-independent and changes for that reason alone. The
ablation artifact predates the gold refit, so its `refusal_threshold` field
still records `0.2552`; no retrieval number depends on that field.

## Reading the production result honestly

- **Retrieval passes at the threshold, barely.** 18/24 = 75.0% is exactly the
  gate. The interval reaches down to 55.1%, and the vector-only and hybrid rows
  are identical at n=24, so this set does not establish that RRF or rerank is
  reliably better. Rerank recovers one question (R4); fusion recovers none.
- **The answer-rate gate fails, and the model — not the threshold — is now the
  binding constraint.** The refitted calibration admits all 24 answerable
  questions, and 8/24 (33.3%) still return a validated claim. The forced
  diagnostic shows the same 8/24 without the threshold at all. The remaining
  16 failures are 13 dropped citation-contract violations, 2 model-declared
  `insufficient_context` replies, and 1 missing tool call, five of which had
  already lost the gold chunk at retrieval.
- **Citation precision and faithfulness now fail at 7/9 (77.8%).** The larger
  answered set exposed a real defect the n=2 baseline could not see: R4's two
  claims cite non-gold chunks. They pass the verbatim-quote contract (the quote
  is a substring of the chunk it names) but do not answer the question from the
  gold passage. The gate stays failed at 80%; it is not lowered. The interval is
  wide ([45.3%, 93.7%]) because n=9.
- **The refusal gate passes, but not because the calibration is perfect.** All
  six unanswerable questions are refused: U2–U5 at `low_score`, U1 with
  `insufficient_context`, U6 with `validation_failed`. The in-sample calibration
  itself refuses only 4/6; U1 and U6 reach the model and are refused there. The
  held-out folds refused 2/3 unanswerables each.
- **The calibration's reported fit is in-sample.** The final artifact is fitted
  on all 30 gold questions — the same questions run (c) evaluates. Its
  balanced-accuracy figure is therefore optimistic, and run (c) is not a clean
  held-out test. The honest estimate is the id-parity split: fit on odd ids and
  evaluate even ids gives BA 0.833 (TPR 1.0, TNR 0.667; 10/10 answerable
  admitted, 2/3 unanswerable refused), and the reverse split gives the same BA
  (14/14, 2/3). Each fold has only three unanswerable questions, so those
  numbers are a direction, not a tight estimate. Details are in
  `eval/calibration.json` and below.
- **Ordering is not statistically established.** CIs overlap; n=24 is a sanity
  check. The harness reports the numbers it can defend and refuses to claim
  more.

## Calibration provenance and the in-sample caveat

`scripts/fit_calibration_gold.py` fits a question-level Platt scaling on the
top reranked score of each of the 30 hand-authored gold questions
(`label = 1` for the 24 answerable, `0` for the 6 unanswerable), then chooses
the threshold that maximises balanced accuracy (mean of TPR and TNR). Fitting
on the decision statistic the refusal gate actually uses is deliberate: the
gate asks "should this question be answered?", not "is this chunk relevant?".

| fit | n | a | b | threshold | TPR | TNR | balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| odd ids → even ids | 17 fit / 13 eval | 0.2627 | 0.8584 | 0.6921 | 1.000 | 0.667 | 0.833 |
| even ids → odd ids | 13 fit / 17 eval | 0.2659 | 0.8185 | 0.6461 | 1.000 | 0.667 | 0.833 |
| **all 30 (committed artifact)** | **30** | **0.3376** | **0.6775** | **0.6361** | **1.000** | **0.667** | **0.833** |

The committed row is **in-sample**: `eval/calibration_fit_set.json` is the same
30 questions as `eval/gold.jsonl`, and the threshold was refitted on all of
them after the held-out check. The held-out rows are the out-of-sample
evidence; both directions agree, which is why the final threshold is not
visibly overfitted, but with three negatives per fold the agreement is weak
evidence. The score cut implied by the final threshold (p = 0.6361) is ≈ −0.353
on the reranker's raw logit scale; note that `dummy:lexical-overlap` has a
different scale, so this artifact must not be used with the dummy reranker.

## Run cost and rate limiting

The endpoint is public, keyless, and answers about once per 30 s per IP. The
client holds one request in flight per endpoint, spaces request **starts** by at
least 31 s, and retries a 429 twice (5 s, then 10 s) before raising
`RateLimitedError`.

| run | requests | retries | rate-limited | minimum spacing floor | measured wall clock |
|---|---:|---:|---:|---:|---:|
| (a) baseline | 5 | 0 | 0 | 4 × 31 = 124 s | 148.1 s |
| (b) forced diagnostic | 30 | 0 | 0 | 29 × 31 = 899 s | 916.4 s |
| (c) final production | 26 | 0 | 0 | 25 × 31 = 775 s | 778.6 s |

Run (c)'s wall clock is almost entirely the rate-limit floor; local ONNX and
retrieval work adds ~4 s across 30 questions because it is dominated by the
31 s waits. The report records the budget in `provenance.run`:
`model_calls {requests, retries, rate_limited}`, `model_min_interval_s`,
`wall_clock_s`, and `blocked` (set after three consecutive 429s). There were 0
retries and 0 rate-limit failures in all three runs, and `blocked` is `null` in
every report.

## Files

| file | contents |
|---|---|
| `eval/gold.jsonl` | 30 fixed questions; span-level gold for the 24 answerable ones |
| `eval/README.md` | size rationale, label provenance, judge provenance, row format |
| `eval/calibration.json` | coefficients, threshold, fit-set hash, held-out history, in-sample caveat |
| `eval/calibration_fit_set.json` | 30 question-level fit rows (`question_id`, `query`, `chunk_id`, `label`, `rerank_score`, `split`) |
| `scripts/fit_calibration_gold.py` | human-gold fitter (production calibration) |
| `scripts/fit_calibration.py` | provisional synthetic fitter, kept for comparison; overwrites the production artifact |
| `eval-report.json` | final production run (c): per-question rows, aggregates, retrieval ablation, gates |
| `eval-report-baseline.json` | run (a): gate run with the provisional `0.2552` threshold |
| `eval-report-forced.json` | run (b): forced-generation diagnostic, `gate_run: false` |
| `eval-report-ablation.json` | vector-only/top-3 weakened run (R2 extractive generator), same schema |
| `src/docqa/eval.py` | loader, gold/corpus verification, Wilson intervals, stage attribution, gates, runner |
| `src/docqa/calibration_fit.py` | shared Platt fit, balanced accuracy, threshold choice |
| `tests/test_eval.py` | golden tests for intervals, gold shape, stage attribution, and gate failure |

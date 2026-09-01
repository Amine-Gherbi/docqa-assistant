# W2-a2 — schema-stage failures: result

Branch `w2-a2-schema-failures`. No deployment (W2-b). The refusal threshold in
`eval/calibration.json` is unchanged (`git diff eval/calibration.json` empty);
no gate was removed or weakened, no gold label was edited, and no judge was
invented.

## Before / after

| gate | W2-a (committed report) | W2-a2 (final keyed run) |
|---|---|---|
| retrieval recall@8 (n=24) | 18/24 (75.0%) [55.1%, 88.0%] PASS | 18/24 (75.0%) [55.1%, 88.0%] PASS |
| answer rate (n=24) | 11/24 (45.8%) [27.9%, 65.0%] **FAIL** | 15/24 (62.5%) [42.7%, 78.8%] **PASS** |
| unanswerable refusal (n=6) | 6/6 (100.0%) [61.0%, 100.0%] PASS | 6/6 (100.0%) [61.0%, 100.0%] PASS |
| citation precision* | 10/11 (90.9%) [62.3%, 98.4%] PASS | 13/15 (86.7%) [62.1%, 96.3%] PASS |
| faithfulness* | 9/11 (81.8%) [52.3%, 94.9%] PASS | 12/15 (80.0%) [54.8%, 93.0%] PASS |
| exit | 6 (answer rate red) | 0 (all five green) |

\* rule-based, not a model judge. W2-a's precision/faithfulness denominator was
11 because 11 answerable questions returned a validated claim; the final run's
denominator is 15. `judge_model_unavailable` in both runs; the generator was not
substituted as judge.

Final run: 710.3 s (11.84 min), 30 questions, 26 model requests, 0 retries,
0 rate-limit failures, 0 server errors. Artifact: `eval-report.json`.

**The lift is +4 answerable questions (45.8% → 62.5%).** It comes from the
generation prompt, not from the JSON fallback: `structured_mode_counts` is
`{"tool": 26}` and the fallback fired **zero** times. The answer-rate interval's
lower bound (42.7%) is below the 50% threshold, so the point estimate passes,
not the whole interval.

## What the 7 "schema" failures actually were

The eval discarded the reason (`error: null`), so a diagnostic run re-ran the
same questions with wire-level records. All seven W2-a `schema` rows came back
HTTP 200, `finish_reason: "tool_calls"`, `structured_mode: "tool"`, with zero
parse errors. The forced `submit_answer` call was honoured and the JSON parsed;
the claims died in the citation contract.

| id | W2-a stage | measured cause | evidence | final |
|---|---|---|---|---|
| B1 | schema | `quote_mismatch` | quote crosses an interleaved column line | `validation` |
| B3 | schema | `quote_mismatch` | quote spans a hyphenated/line break | `validation` |
| B4 | schema | `quote_mismatch` | quote reconstructed across the column gap | **answered** |
| B5 | schema | `quote_mismatch` | `Label = IsNext` interleaved inside the quote | `validation` |
| D1 | schema | `quote_mismatch` | quote skips the interleaved right column | **answered** |
| D2 | schema | `quote_mismatch` | quote crosses the interleaved column | `validation` |
| R2 | schema | `bad_quote` | verbatim quote exceeded the 240-character cap | **answered** |

The instrumented diagnostic run found 9 schema-stage rows, not 7, because two
more questions (S3, D3) failed the same contract in that sample. Cause counts
for its 9: `quote_mismatch` 8, `bad_quote` 1; `schema_parse_error` 0,
`model_call_error` 0, prose-instead-of-tool-call 0, `finish_reason: length` 0,
genuine refusal 0. The diagnostic run's answer rate was 9/24 (37.5%), so its
sample was harsher than W2-a; the final run is the gate run.

The root cause is the corpus, not the model protocol: the chunks are raw
PyMuPDF extraction of two-column pages. A left-column line and a right-column
line are stored on the same extracted line, and words are hyphenated across
line breaks. A sentence quoted in natural reading order is therefore not a
contiguous substring of the chunk. For example, the stored chunk contains
`Training of BERTBASE was performed on 4` / `Label = IsNext` / `Cloud TPUs ...`
and `setting (sec-` / `70` / `tion 3.2)`.

## What changed

- `src/docqa/answer.py`: wire-level `ModelFailure`/`FailureDetail` (HTTP status,
  finish reason, structured mode, parse errors, 200-character body and raw-text
  excerpts, claim counters); `structured_mode` on every reply; a one-shot
  `structured_output` override on `ModelRequest`; exactly one JSON-only retry
  after a schema parse failure, recorded as `structured_mode: "json-fallback"`;
  the prompt now states that passages are interleaved two-column extraction and
  that a quote must not contain a newline or cross the column gap.
- `src/docqa/eval.py`: `schema` and `validation` are separate stages;
  every failing row carries the concrete `failure` object;
  `metrics.stage_failure_reasons` and `metrics.structured_mode_counts` are
  aggregated and printed.
- `docs/adr/0004-structured-output.md`: the fallback is documented as one
  automatic retry after a schema failure, not a deployment switch; citation
  drops get no retry.
- Tests: a fallback-aware fake asserts `[None, "json"]`, `json-fallback`,
  one repair turn, same validation; a quote mismatch asserts **no** retry and
  `structured_mode: "tool"`; a second schema failure refuses once; reply
  diagnostics and the JSON-only body are covered. `122 passed`.
- `docs/EVAL.md` and `docs/PROOF.md`: current numbers, failure-field names and
  the remaining cause.

## What is still not fixed

- Four answerable questions (B1, B3, B5, D2) still fail on the same
  `quote_mismatch`: the model ignores the no-newline rule and returns a quote
  that spans the interleaved columns. The prompt mitigates the cause; it does
  not remove it.
- The true fix is column-aware extraction at ingest. That re-ingests the corpus,
  changes chunk text and embeddings (retrieval), and can invalidate the gold
  chunk ids; this card forbids touching retrieval or re-labelling gold, so it is
  left as a separate ingest task.
- Faithfulness is exactly 80.0% (12/15), on the gate; the Wilson interval
  reaches 54.8%. Recall is exactly 75.0% (18/24). Both are point-estimate
  passes.
- The judge remains `judge_model_unavailable` (HTTP 429/429/503); precision and
  faithfulness stay rule-based.
- No deployment was performed; W2-b owns that.

# Invariants

Rules that must never break. Each cites the enforcing code and the test that
fails if it regresses. Invariant 6's reliability half is **unmeasured**: the
judge id is pinned and differs from the generator, but no human labels exist
for Cohen's kappa, so the rule-based metric is the reported one.

| # | Invariant | Code | Test |
|---|---|---|---|
| 1 | `len(tokenizer.encode(chunk).ids) <= 448`; a batch over the cap fails instead of storing a truncated vector | `src/docqa/ingest/chunking.py:24` (`HARD_CAP = 448`), `:32` (`TokenCapViolation`), assertion at `:182` | `tests/test_chunking.py` |
| 2 | Every claim carries ≥1 citation id in the context, and every quote is a normalised substring of the cited chunk; otherwise the claim is dropped | `src/docqa/answer.py:637` (`validate_claims`), `:611` (`normalise_text`) | `tests/test_citations.py` |
| 3 | `corpus.lock.json` is enforced by SHA-256; ingest refuses a mismatched or unlisted file | `src/docqa/ingest/lock.py:82` (`verify_file`), `:92` (digest check) | `tests/test_lock.py`, `tests/test_cli_and_corpus_script.py` |
| 4 | The refusal threshold comes from `eval/calibration.json` (`{a,b,threshold,fit_set_sha256}`), loaded at boot, never hand-tuned | `src/docqa/answer.py:276` (`load_calibration`); artifact `eval/calibration.json` | `tests/test_refusal.py` |
| 5 | Every rate prints a Wilson 95% CI and its denominator; `stage_failed` is recorded per question | `src/docqa/eval.py:119` (`wilson_interval`), `:383` (`stage_failed`), `:668` (stage counts) | `tests/test_eval.py` |
| 6 | The judge model differs from the generator and is pinned by exact id; it is never replaced by the generator when unavailable | `src/docqa/judge.py` (`probe_judge`), `src/docqa/cli.py` (`JUDGE_REQUIRED`), `eval/judge.yaml`, `eval/judge-status.json` | `tests/test_judge_check.py`, `tests/test_judge_probe.py`; **κ not computed** — 30 question labels, 0 human judge labels |
| 7 | ONNX work runs in `anyio.to_thread.run_sync` behind a process-wide semaphore (`Semaphore(2)` by default; `1` on the hosted free instance via `DOCQA_ONNX_CONCURRENCY`) | `src/docqa/onnx.py:20` (`DEFAULT_CONCURRENCY`), `:35` (`onnx_semaphore`), `:39` (`Semaphore`) | `tests/test_embedder.py` |

The production calibration is fitted on the human gold questions by
`scripts/fit_calibration_gold.py`; its held-out split and in-sample caveat are
recorded in `eval/calibration.json` and `docs/EVAL.md`. The
`DOCQA_FORCE_GENERATION` / `--force-generation` diagnostic bypasses **only**
invariant 4's score gate for measurement; invariant 2's citation contract still
runs, and a forced eval records `gate_run: false`.

Also deliberate and tested by absence: generation is **not** streamed. The API
returns one validated JSON response and has no SSE or `StreamingResponse` route
(`src/docqa/api.py`), so an uncited claim cannot reach the client;
`test_citations.py` covers the rejection path. `citation_render_miss` is counted
client-side in `ui/src/App.tsx`; the UI resolves `quote` first and `rects[]`
second (`ui/src/components/PdfPane.tsx`).

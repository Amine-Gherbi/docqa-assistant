# Invariants

Rules that must never break. Each cites the enforcing code and the test that
fails if it regresses. Invariant 6's reliability half is **unverified**.

| # | Invariant | Code | Test |
|---|---|---|---|
| 1 | `len(tokenizer.encode(chunk).ids) <= 448`; a batch over the cap fails instead of storing a truncated vector | `src/docqa/ingest/chunking.py:24` (`HARD_CAP = 448`), `:32` (`TokenCapViolation`), assertion at `:182` | `tests/test_chunking.py` |
| 2 | Every claim carries ≥1 citation id in the context, and every quote is a normalised substring of the cited chunk; otherwise the claim is dropped | `src/docqa/answer.py:512` (`validate_claims`) | `tests/test_citations.py` |
| 3 | `corpus.lock.json` is enforced by SHA-256; ingest refuses a mismatched or unlisted file | `src/docqa/ingest/lock.py:82` (`verify_file`), `:92` (digest check) | `tests/test_lock.py`, `tests/test_cli_and_corpus_script.py` |
| 4 | The refusal threshold comes from `eval/calibration.json` (`{a,b,threshold,fit_set_sha256}`), loaded at boot, never hand-tuned | `src/docqa/answer.py:216` (`load_calibration`); artifact `eval/calibration.json` | `tests/test_refusal.py` |
| 5 | Every rate prints a Wilson 95% CI and its denominator; `stage_failed` is recorded per question | `src/docqa/eval.py:101` (`wilson_interval`), `:365` (`stage_failed`), `:585` (stage counts) | `tests/test_eval.py` |
| 6 | The judge model differs from the generator and is pinned by exact id; reliability (Cohen's κ) must be measured before any judge score is claimed | `src/docqa/cli.py:31` (`JUDGE_REQUIRED`), `:254` (`_run_judge_check`), `eval/judge.yaml` | `tests/test_judge_check.py`; **κ unverified** — no endpoint or human anchor set |
| 7 | ONNX work runs in `anyio.to_thread.run_sync` behind `Semaphore(2)` | `src/docqa/onnx.py:12`, `:15` (`onnx_semaphore`), `:19` (`Semaphore(2)`) | `tests/test_embedder.py` |

Also deliberate and tested by absence: generation is **not** streamed, and SSE
streams validated claims only (`src/docqa/api.py`); `test_citations.py` covers
the rejection path. `citation_render_miss` is counted client-side in
`ui/src/App.tsx`; the UI resolves `quote` first and `rects[]` second
(`ui/src/components/PdfPane.tsx`).

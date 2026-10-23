# Interview drill

Twenty hard questions, each answered in at most six lines with a `file:line` or
a runnable command. Anything that cannot be backed that way is labelled
**unverified**.

1. **Why is generation non-streaming?** The API returns one validated JSON response; there is no SSE or `StreamingResponse` route, so an uncited claim cannot reach the user before the citation check. `src/docqa/api.py:4`; test: `uv run pytest tests/test_citations.py`.
2. **What exactly makes a citation valid?** The id must be in the retrieved context and the normalised quote must be a substring of the cited chunk; otherwise the claim is dropped. `src/docqa/answer.py:542` (`validate_claims`, quote check at `:579`).
3. **Why store `page_text_sha256`?** PyMuPDF offsets drift across extractor versions; the hash makes an upgrade fail loudly instead of silently moving highlights. `src/docqa/answer.py:319`; `docs/INVARIANTS.md`.
4. **How does the UI find the highlight?** Rebuild the page text from PDF.js, resolve the verbatim quote to DOM ranges, paint with the CSS Custom Highlight API, and fall back to `rects[]`. `ui/src/components/PdfPane.tsx:320`, `:338`.
5. **What is `citation_render_miss`?** A client-side count of quotes that could not be mapped to a DOM range; it is shown in the trace drawer's final context. `ui/src/App.tsx:250`; `ui/src/components/TraceDrawer.tsx:307`.
6. **Why `ts_rank_cd` and not BM25?** The lexical half is PostgreSQL-native; `ts_rank_cd` is term frequency plus cover density with no IDF or length normalisation. `src/docqa/retrieval.py:8`, SQL at `:70`; `docs/ABLATION.md`.
7. **Why RRF instead of score blending?** Vector distance and `ts_rank_cd` are not comparable scales; RRF uses only ranks and adds `1/(60+rank)` per list. `src/docqa/retrieval.py:12`, `:309`.
8. **What does the ablation prove?** Rerank moves recall@8 from 17/24 to 18/24, one question; fusion moves none, and the intervals overlap, so no significant ordering is claimed. `docs/EVAL.md`; artifact `eval-report.json`.
9. **Why 30 questions, not 100?** 100 span labels could not be authored honestly in one session; the loader refuses a different shape. `src/docqa/eval.py`; `eval/README.md`.
10. **What is the biggest eval failure?** Answer rate 8/24 (33.3%, [18.0%, 53.3%]). With the threshold bypassed it is still 8/24; 13 replies made claims whose quotes were not verbatim, so the contract dropped them. `eval-report.json`; `eval-report-forced.json`.
11. **Is the refusal threshold doing the refusing?** Not any more: it admits all 24 answerable questions in-sample, and the model is the binding constraint. U1 and U6 reach the model and are refused downstream, so the gate stays 6/6. `docs/EVAL.md`.
12. **Can you tune the threshold now?** No: invariant 4 forbids hand-tuning. It is fitted by `scripts/fit_calibration_gold.py`; the committed fit is in-sample and the odd/even held-out folds are recorded in `eval/calibration.json`. `src/docqa/answer.py:245`.
13. **How is the judge kept independent?** `eval/judge.yaml` pins `claude-opus-5-5`, which differs from the generator `openai-fast`, plus a prompt hash and temperature 0; reliability κ is **unverified** (no endpoint or anchor set). `src/docqa/cli.py:511`.
14. **What happens if the gateway rewrites `tool_choice`?** `docqa probe` sends a real forced call and records the verdict; the documented fallback is JSON-only plus `json-repair`. `src/docqa/cli.py`; `docs/adr/0004-structured-output.md`.
15. **Why ONNX and not PyTorch?** Python 3.12 is the supported ML path here and fastembed's ONNX artifacts run on CPU without torch; `docs/ENV.md` records the exact versions.
16. **How is the event loop protected?** ONNX calls run in `anyio.to_thread.run_sync` behind a process-wide `Semaphore(2)`. `src/docqa/onnx.py:15`, `:19`; test: `uv run pytest tests/test_embedder.py`.
17. **Is the corpus committed?** No; `corpus.lock.json` pins SHA-256 and `scripts/fetch-corpus` materialises it before ingest. `src/docqa/ingest/lock.py:82`; test: `uv run pytest tests/test_lock.py`.
18. **What is synthetic in the demo?** The answers are recorded extractive outputs, not live calls; the PDF is the real Sentence-BERT paper, and the eval corpus and embeddings are real. `ui/src/fixtures/real-paper.json`; `node ui/scripts/verify_demo_paper.mjs`.
19. **Is there a live deployment?** No — local only: UI `127.0.0.1:5173`, API `127.0.0.1:8788`. A hosted URL is **unverified** because none exists.
20. **Is the k6 concurrency test green?** **Unverified**: k6 is not installed, so `bash scripts/verify.sh` prints `SKIP k6`; the scenario is `scripts/k6-ask.js` and `docs/ENV.md` records the local p95.

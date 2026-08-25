# Interview drill

Twenty hard questions, each answered in at most six lines with a `file:line` or
a runnable test. Anything that cannot be backed that way is labelled
**unverified**.

1. **Why is generation non-streaming?** Streaming would let an uncited claim reach the user before validation. `src/docqa/api.py` returns the validated response; the SSE contract streams claims, not tokens. Test: `tests/test_citations.py`.
2. **What exactly makes a citation valid?** The id must be in the context and the normalised quote must be a substring of the cited chunk. `src/docqa/answer.py:512`.
3. **Why store `page_text_sha256`?** PyMuPDF offsets drift across versions; the hash makes an extractor upgrade fail loudly instead of silently moving highlights. `src/docqa/ingest/extract.py`.
4. **How does the UI find the highlight?** Rebuild the page text from PDF.js, resolve the verbatim quote, paint with the CSS Custom Highlight API, fall back to `rects[]`. `ui/src/components/PdfPane.tsx`.
5. **What is `citation_render_miss`?** A client-side count of quotes that could not be mapped to a DOM range; it is shown in the trace drawer. `ui/src/App.tsx`.
6. **Why `ts_rank_cd` and not BM25?** The lexical half is PostgreSQL native; `ts_rank_cd` is TF plus cover density with no IDF or length normalisation. `src/docqa/eval.py` / `docs/ABLATION.md`.
7. **Why RRF instead of score blending?** Vector distance and `ts_rank_cd` are not comparable; RRF uses ranks only. `src/docqa/retrieval.py`.
8. **What does the ablation prove?** Rerank recovers two of ten gold chunks; CIs overlap, so no significant ordering is claimed. `docs/ABLATION.md`.
9. **Why 30 questions, not 100?** 100 span labels could not be authored honestly in one session; the loader refuses a different shape. `src/docqa/eval.py`, `eval/README.md`.
10. **What is the biggest eval failure?** Answer rate 5/24, citation precision 3/15, faithfulness 2/15 because there is no model key and the extractive fallback is weak. `eval-report.json`.
11. **Why does the refusal gate pass while answers fail?** The calibration threshold refuses 19/24 answerable questions before generation; it is honest but over-conservative. `docs/EVAL.md`.
12. **Can you tune the threshold now?** No: invariant 4 forbids hand-tuning; refit on a larger human set and re-run. `src/docqa/answer.py:216`.
13. **How is the judge kept independent?** `eval/judge.yaml` pins a different model id and a prompt hash; the blinded prompt is `eval/judge-prompt.txt`. Reliability κ is **unverified** — no endpoint or anchor set. `src/docqa/cli.py:254`.
14. **What happens if the gateway rewrites `tool_choice`?** `judge-check --probe-tool-choice` reports it; the documented fallback is JSON-only plus `json-repair`. `docs/adr/0004-structured-output.md`.
15. **Why ONNX and not PyTorch?** Python 3.14 has no ML wheels; 3.12 + fastembed ONNX is the supported path. `docs/ENV.md`.
16. **How is the event loop protected?** ONNX calls run in `anyio.to_thread.run_sync` behind a process-wide `Semaphore(2)`. `src/docqa/onnx.py:15`.
17. **Is the corpus committed?** No; `corpus.lock.json` pins SHA-256 and `scripts/fetch-corpus` materialises it. `src/docqa/ingest/lock.py:82`.
18. **What is synthetic in the demo?** The UI fixture PDF and offline answer; the eval corpus and embeddings are real. `README.md`, `ui/src/fixtures/demo-answer.json`.
19. **Is there a live deployment?** No — local only (`127.0.0.1:5173`, API `:8788`). Hosted URL: **unverified**.
20. **Is the k6 concurrency test green?** **Unverified**: k6 is not installed; `scripts/verify.sh` prints `SKIP k6`. The scenario is `scripts/k6-ask.js`.

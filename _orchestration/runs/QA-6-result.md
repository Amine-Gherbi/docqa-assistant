# QA-6 — headers, bundle, docs

Branch: `qa-6-headers-perf` (from `master`). Not merged and not deployed.

## Findings → fix → proof

| # | Finding | Fix | Proof |
|---|---|---|---|
| 1 | The API sends none of the six security headers. | `SecurityHeadersMiddleware` adds HSTS, `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, `Permissions-Policy`, and `Content-Security-Policy: default-src 'none'; frame-ancestors 'none'` at `http.response.start`, for every response including 404s and CORS preflights. CORS still allows only the configured UI origin (`HEAD` added to the allowed methods). | Transcript below; `tests/test_live_endpoint.py::test_security_headers_are_on_health_ask_and_404`, `::test_cors_still_admits_only_the_configured_ui_origin`. |
| 2 | `/docs` and `/openapi.json` are published. | `FastAPI(docs_url=None, redoc_url=None, openapi_url=None)`. | `GET /docs` → 404, `GET /openapi.json` → 404; test `::test_interactive_docs_and_openapi_are_not_published`. |
| 3 | `422` responses echo the request body (120 KB in, 120 KB out). | A `RequestValidationError` handler returns only `{path, type, reason}` per error. Pydantic's `input`/`ctx` are dropped. | A 100,000-byte question gives a **142-byte** 422 with path `body.question`, reason `String should have at most 2000 characters`, no `input`; malformed JSON also answers without the sentinel. Test `::test_invalid_body_reports_the_path_without_echoing_the_submitted_bytes`. |
| 4 | `/health` discloses an absolute container path. | Removed `demo_cache.path`; the route reports `entries`, `model`, `generated_at` only. | `demo_cache = {"entries": 0, "model": null, "generated_at": null}`; test `::test_health_does_not_disclose_the_cache_path`. |
| 5 | Every GET route answers `405` to `HEAD`. | Chosen behaviour: **HEAD is answered like GET with an empty body** on routes that support GET (`HeadAsGetMiddleware` rewrites the method and suppresses the body). Routes that do not support GET keep Starlette's `405` with its `Allow` header. This is stated in the middleware docstring. | `HEAD /health` → 200, 0 body bytes, `Content-Length: 412` (same as GET); `HEAD /ask` → 405, `Allow: POST`, 0 body bytes; `HEAD /missing` → 404. Test `::test_head_is_get_without_a_body_and_keeps_allow_on_post_only_routes`. |
| 6 | `src/docqa/api.py` interpolates an upstream model exception into a 502 body. | The response carries the stable code `model_unavailable` and a fixed message; the exception is logged server-side at ERROR. | Test `::test_model_failure_returns_a_stable_code_and_logs_the_detail` asserts the upstream host/token never appears in the body and does appear in the captured log call. |
| 7 | One 216 KB Vite entry bundle; the `/eval` view is evaluated on first paint. | `EvalPage` is a `React.lazy` route chunk. Additionally, PDF.js — the largest dependency, already loaded asynchronously — is dynamically imported inside `PdfPane`, so it is no longer part of the entry chunk either. URLs and rendered markup are unchanged; no dependency added. | Bundle table below; Playwright citation test passes; `/eval` still renders the gallery. |
| 8 | Lighthouse best-practices 96 from missing `/favicon.ico`; `valid-source-maps` fails. | Added a real 32×32 ICO at `ui/public/favicon.ico` (PNG-in-ICO), linked from `index.html`; enabled Vite source maps. | `GET /favicon.ico` → 200 `image/x-icon`, 202 bytes; best-practices 96 → **100**; `valid-source-maps` 0 → **1**. |
| 9 | `docs/EVAL.md:107` calls an old 8/24 run "final production". | Relabelled it a historical pre-keyed gate run (also the two other "final production" labels). The text now says its artifact was superseded by the current keyed run in `eval-report.json`. | `grep "final production" docs/` → no matches. |
| 10 | `data/demo-answers.json` claims `openai-compatible:openai-fast` 13 times, a removed model. | Did **not** fabricate a model id and did not regenerate (that needs the keyed endpoint). Removed the per-entry and top-level model claims and set each cached response's `model` to `null`; `CachedAnswer.model` is now optional. `docs/PROOF.md` now says the cache was recorded 2026-10-23, predates the keyed run, and is not a product of the current model. `docs/ENV.md` was corrected to match. | `grep "openai-fast" data/ docs/` → no matches outside a test fixture; `/health` reports `demo_cache.model: null`; cache parsing still accepts old entries. |

## Header / protocol transcript

Captured with `fastapi.testclient.TestClient(create_app(...))` (injected calibration and model):

```
GET /health        200
  Strict-Transport-Security: max-age=31536000; includeSubDomains
  X-Content-Type-Options: nosniff
  X-Frame-Options: DENY
  Referrer-Policy: no-referrer
  Permissions-Policy: accelerometer=(), camera=(), geolocation=(), gyroscope=(), magnetometer=(), microphone=(), payment=(), usb=()
  Content-Security-Policy: default-src 'none'; frame-ancestors 'none'
  demo_cache: {'entries': 0, 'model': None, 'generated_at': None}
POST /ask (valid)  200  CSP present
GET /missing       404  X-Frame-Options: DENY
POST /ask (100 KB) 422  142 bytes, paths ['body.question'], "input" absent
HEAD /health       200  0 body bytes, Content-Length 412; GET is also 412
HEAD /ask          405  Allow: POST, 0 body bytes
GET /docs          404
GET /openapi.json  404
OPTIONS /ask, Origin: https://evil.example -> 400, no Access-Control-Allow-Origin
```

`/health` and `/ready` remain reachable (both 200 in the suite; Render probes them).

## UI bundle and Lighthouse

Local production build, Vite 8.3.5:

| artifact | before | after |
|---|---:|---:|
| entry JS | 712.59 KB raw / 213.92 KB gzip | **271.54 KB raw / 81.88 KB gzip** |
| `/eval` chunk | in the entry | **13.55 KB raw / 4.58 KB gzip** |
| PDF.js chunk | in the entry | **430.21 KB raw / 129.32 KB gzip**, loaded after mount |
| CSS | 23.83 KB / 4.84 KB gzip | unchanged |
| favicon | missing (404) | 202 B, 200 |

Lighthouse 13.6.0, home view, default mobile preset, warm runs (`--only-categories=performance,best-practices,accessibility`; headless Chrome, local `vite preview`):

- External audit baseline: **perf 84**, TBT 490 ms.
- Local baseline build (5 runs; runs 2–5 warm): **perf 89.5 median, TBT 437 ms median**.
- Final build (5 runs; runs 13–15 warm): **perf 89 median, TBT 434 ms median** (all five: 83/84/89/90/88, TBT 690/644/434/421/482 ms).

The score and TBT are unchanged within this machine's run-to-run noise. The measured win is structural: the entry chunk drops from 213.92 to 81.88 KB gzip because the eval view and PDF.js no longer execute on first paint. Lighthouse still reports the deferred PDF.js chunk as partly unused (107 KB) — the home view renders a PDF, so that code is required by the product, just no longer by the entry chunk.

`/eval` was checked in Playwright: the failure gallery renders. The existing "honest empty report" assertion in that spec failed locally because a `docqa serve` instance on port 8788 answered through the Vite proxy with the committed report; that is an environment assumption in the test, not a regression from the split.

## Verification

- `bash scripts/verify.sh` → **exit 0**: ruff clean, `mypy --strict` clean (27 files), **208 pytest tests passed**; k6 prints `SKIP` because it is not installed (unchanged, documented).
- `npm --prefix ui run verify` → lint clean, typecheck clean, **19/19 UI tests passed**.
- `npm --prefix ui run e2e -- --grep "all three citations"` → **1 passed** after the PDF.js dynamic import.
- `npm --prefix ui run build` → exit 0, demo guard passes.

## Anything left

- The deployed services were not touched (no merge, no deploy), so the hosted Lighthouse and header checks were not re-run. The proofs above are from the same `create_app` and the same production build.
- The API is still served over HTTP locally; the HSTS header is only meaningful behind the Render TLS proxy.
- `data/demo-answers.json` remains historical. Regenerating the six answers against the keyed endpoint is a separate operation and was deliberately not faked.
- The PDF.js chunk (129.32 KB gzip) is still downloaded on the home view because the page shows a PDF. Removing it would change the product, not just the bundle.
- `ui.html` remains an untracked scratch file at the repository root and is not part of this change.

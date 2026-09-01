# Visitor PDF Upload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A visitor drops a PDF on the hosted page and gets a cited answer about their own document, with hard limits, quotas, a 24-hour TTL, and the free-tier disclosure stated where the decision is made.

**Architecture:** Uploads are stored as `bytea` in Postgres and enqueued to an in-process worker that shares the API's already-loaded embedder. Each upload owns a `collection_id`; retrieval filters on it, and `/ask` verifies the caller's signed cookie owns a ready collection before using it. A scheduler sweeps expired uploads, their pages, documents and chunks.

**Tech Stack:** FastAPI, SQLAlchemy async, PostgreSQL 16 + pgvector, PyMuPDF 2.0, React/Vite, PDF.js.

**Spec:** run card W2-c (`P:/portfolio/_orchestration/LIVE-PLAN.md` §2) plus `CLAUDE.md` invariants.

## Global Constraints

- Limits on bytes, never filenames: 413 > 25 MB, 415 unless offset 0 is `%PDF-`, 422 > 80 pages / encrypted / unparseable.
- Never parse in the request loop; the worker has a 90 s budget.
- One ingest per client (409 with the running id), 3 uploads/client/day, 20 uploads/hour host-wide (429 with reset).
- `expires_at = now() + interval '24 hours'`; sweep deletes upload, pages, document, chunks and vectors.
- The disclosure sentence is above the drop zone at button type size, in `README.md` and `docs/PROOF.md`.
- No `secure`, `private` or `encrypted` privacy claim in `ui/src` or `README.md`.
- No PDF bytes or page text to a non-owning client; PDF.js `isEvalSupported: false`; model key never in the browser.
- Branch `w2-c-pdf-upload` from `master`; `bash scripts/verify.sh` exit 0; no merge, no force-push, no `portfolio-site/`.

## Review Focus

- A forged or absent cookie must never create or read an upload (401/404, not a new owner).
- A second client must not learn that another client's document exists (404 for both unknown and foreign).
- A timeout or exception mid-ingest must not leave chunks retrievable under the collection.
- The demo collection must remain the default for `/ask`; an upload query must never retrieve demo chunks.
- The sweep must be idempotent and must remove vectors (chunks) before the upload row.

---

### Task 1: Schema and models

**Files:** `alembic/versions/0002_uploads.py`, `src/docqa/db.py`, `src/docqa/config.py`
**Produces:** `Document.collection_id`, `Chunk.collection_id`, `Upload`, `UploadPage`; settings fields for limits/secret.

- [ ] Add migration 0002: `collection_id` on documents/chunks default `demo` + indexes; `uploads`; `upload_pages`; FKs with cascade; checks on status.
- [ ] Add ORM models and settings fields (`cookie_secret`, `upload_max_bytes`, `upload_max_pages`, `upload_ttl_hours`, `upload_per_client_day`, `upload_per_host_hour`, `upload_timeout_s`, `upload_sweep_interval_s`).
- [ ] `uv run alembic upgrade head` on the local database; test migration round-trip via existing idempotency test.
- [ ] Commit.

### Task 2: Limits and session signing (pure functions, no DB)

**Files:** `src/docqa/uploads/__init__.py`, `limits.py`, `session.py`; `tests/test_upload_limits.py`, `tests/test_upload_session.py`
**Produces:** `check_upload_bytes(raw, max_bytes, max_pages) -> PdfInspection`, `UploadLimitError(status, code, detail)`; `SessionSigner.new_token()/verify()`.

- [ ] Failing tests: prefix, size, `/Count 81`, `/Encrypt`, no `%%EOF`/`startxref`, oversized vector, tampered cookie, expired-shape token.
- [ ] Implement the trailer scan (last 256 KiB): magic, `%%EOF`, `startxref`, `/Encrypt N N R`, max `/Count`; fall back to `/Type /Page` count.
- [ ] Implement HMAC-SHA256 `uuid4hex.digest` cookie signing with `compare_digest`.
- [ ] Commit.

### Task 3: Extract bytes + upload store

**Files:** `src/docqa/ingest/extract.py`, `src/docqa/uploads/store.py`; `tests/test_upload_store.py`
**Produces:** `extract_pdf_bytes(raw, label) -> ExtractedPdf`; quota/ownership/sweep functions.

- [ ] Refactor `extract_pdf` to call `extract_pdf_bytes`; keep pinned-version check.
- [ ] Add store functions: recent client/host counts, active upload, create, owned get, ready collection, page text, sweep, recover interrupted.
- [ ] Integration tests (skip if PostgreSQL unreachable): quotas, ownership, sweep removes upload + pages + documents + chunks, demo chunks untouched.
- [ ] Commit.

### Task 4: Upload worker and service

**Files:** `src/docqa/uploads/worker.py`, `src/docqa/uploads/service.py`; `tests/test_upload_worker.py`
**Produces:** `UploadService.submit/status/file/page/collection_for/sweep/session`, `start/stop/drain`.

- [ ] Worker: set parsing, extract in a thread with `abandon_on_cancel=True` under `asyncio.timeout(90)`, reject encrypted/page-limit, create document, store pages, chunk+embed per page batch, update pages_done, mark ready; on failure delete partial rows and mark failed with code.
- [ ] Service: quota checks under `pg_advisory_xact_lock`, insert queued row, schedule worker task, 60 s sweep loop + boot sweep/recovery.
- [ ] Tests: one-at-a-time 409, per-client day 429, host hour 429, ready → chunks in its collection only, timeout/exception cleanup, sweep.
- [ ] Commit.

### Task 5: Retrieval collection filter and API endpoints

**Files:** `src/docqa/retrieval.py`, `src/docqa/api.py`; `tests/test_upload_api.py`
**Produces:** `PgCandidateSource(session, collection)`, `retrieve_with_trace(..., collection)`, `GET /session`, `POST /documents`, `GET /documents/{id}`, `GET /documents/{id}/file`, `GET /documents/{id}/pages/{n}`, `POST /ask {collection}`.

- [ ] Add a mandatory `d.collection_id = :collection` predicate to both SQL paths; default `demo`.
- [ ] `AskPayload.collection`; `/ask` bypasses the demo cache for uploads, requires the owning cookie and a ready upload.
- [ ] Session cookie: `HttpOnly`, `Secure` on https, `SameSite=None` on https (`Lax` locally), 30 days; `/session` returns the client id.
- [ ] Upload endpoints map limit/quota/conflict errors to 413/415/422/429/409; foreign ids return 404; `Origin` (when present) must be allowed.
- [ ] CORS `allow_credentials=True`; PDF file response `Cache-Control: no-store`, `application/pdf`.
- [ ] Tests with a fake backend for endpoint mapping and with the real service against the local DB for isolation.
- [ ] Commit.

### Task 6: UI

**Files:** `ui/src/api.ts`, `ui/src/types.ts`, `ui/src/components/UploadPanel.tsx`, `ui/src/App.tsx`, `ui/src/components/PdfPane.tsx`, `ui/src/styles.css`, `ui/src/App.test.tsx`
**Produces:** drag-and-drop, stage progress, uploaded-document reader, demo button.

- [ ] API client: `ensureSession`, `uploadDocument`, `uploadStatus`, `uploadedPdfUrl`, `askQuestion(question, collection?)` with `credentials: 'include'`.
- [ ] `UploadPanel`: disclosure (11 px, same as `.primary`) immediately above the drop zone, "deleted after 24 hours" before it, file input + drag/drop, stage text `queued → parsing (n/m) → embedding → ready`, error copy, limits.
- [ ] App state: demo vs upload document, dynamic title/source label/citation rendering, `?document=` restore, "Demo paper" reset.
- [ ] PdfPane: `isEvalSupported: false`, `withCredentials: true`.
- [ ] Tests: disclosure text present, upload POST + poll + ask includes `collection`, demo unchanged.
- [ ] `npm --prefix ui run verify`; commit.

### Task 7: Docs and gate

**Files:** `README.md`, `docs/PROOF.md`, `docs/ENV.md`, `docs/INVARIANTS.md`
- [ ] README upload paragraph with the exact disclosure and limits; PROOF section (to be filled with hosted numbers); ENV variables; invariant row for collection isolation and TTL.
- [ ] `grep -rniE "secure|private|encrypted" ui/src README.md` reviewed and justified.
- [ ] `bash scripts/verify.sh` exit 0; report the test count.
- [ ] Commit.

### Task 8: Hosted proof

- [ ] Push branch; apply migration 0002 to Neon from the build machine.
- [ ] Set `DOCQA_COOKIE_SECRET` on the API service; point both services at `w2-c-pdf-upload`; deploy; warm `/health`.
- [ ] Hosted: GET `/session`; upload a 5-page PDF; poll to ready; ask; verify quote resolves on the supplied page; second client gets 404.
- [ ] Hosted: prove 413/415/422; 409 concurrent; 429 daily/hourly.
- [ ] Measure parse/embed budget for 80 pages; lower `DOCQA_UPLOAD_MAX_PAGES` + UI copy to the measured limit if needed, redeploy.
- [ ] Set an upload's `expires_at` in the past via Neon; wait for the in-app sweep; show before/after counts and 404.
- [ ] Write `_orchestration/runs/W2-c-result.md` with transcripts, limits table, TTL sweep, budget and what is red.

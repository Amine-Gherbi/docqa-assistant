# Marginalia UI

Ask a question about the commit-pinned PDF fixture and get an answer whose
claims point at highlighted spans, with the retrieval trace and refusal number
next to it. This package is the R4 UI; the backend lives in the repository root.

## Ports (read this before starting anything)

| Service | Port | Why |
| --- | --- | --- |
| API | **8788** | Port 8100 (the value in the root `CLAUDE.md`) falls inside Windows' excluded port range **8001–8300** on this machine, so the API is documented and proxied at 8788. |
| UI dev server | **5173** | Vite, `strictPort: true`. |
| UI preview | 5174 | `npm run preview`. |

`vite.config.ts` proxies `/api` to `http://127.0.0.1:8788` and strips the
prefix, so the browser calls `/api/ask` and the server receives `/ask`. Start
the API on 8788; the UI itself never needs an absolute API URL. Override with
`VITE_API_BASE` if the proxy is not in play.

## Commands

```bash
npm install
npm run dev          # UI on http://localhost:5173
npm run build        # tsc -b && vite build
npx tsc --noEmit     # typecheck without emit
npm run lint
npm test             # vitest: fixture resolution checks
npm run e2e          # Playwright highlight flow (chromium)
```

Playwright uses the locally installed Chromium. If it is missing on another
machine, run `npx playwright install chromium`.

## What the Ask page does

- **Question box** posts `{ "question": "…" }` to `POST /api/ask`.
- **Claims** are rendered from the validated response. Every claim is a button;
  each citation chip (`p.3 · chunk 112`) selects one citation. The server has
  already dropped uncited claims and quote mismatches before they arrive.
- **PDF pane** loads `public/fixtures/fixture-corpus.pdf` with PDF.js, renders
  the cited page's canvas and text layer, and highlights the citation's verbatim
  `quote`.
  - Resolution is quote-first: the page string is rebuilt from PDF.js text
    items (`hasEOL` → newline), normalised exactly like the server (NFKC,
    typographic quotes, collapsed whitespace, lower case), and searched. A map
    from normalised offsets back to raw offsets yields the text-item slices.
  - The slices become a DOM `Range` over the PDF.js text layer and are applied
    with the **CSS Custom Highlight API** (`::highlight(docqa-citation)`).
  - If the quote cannot be mapped (or the browser lacks the Highlight API), the
    citation's stored **`rects[]`** are painted as a scaled overlay.
  - A genuine resolution miss increments the client-side
    `citation_render_miss`; the count and details are in the trace drawer.
- **Fixture self-check**: on load, the app resolves all 20 committed citations
  against the text PDF.js actually extracts. The drawer shows `20/20 resolve`
  and names any drift. This is the runtime twin of
  `scripts/verify_fixtures.mjs`.
- **Offline demo**: "Load offline demo" renders the committed demo answer (and
  the retrieval trace) without calling the API. Ask the demo question with no
  API running and the same fixture is used automatically.
- **Trace drawer**: five sections from `/ask`:
  1. vector hits — rank, chunk, page, cosine distance;
  2. lexical hits — rank, chunk, page, `ts_rank_cd`;
  3. RRF fusion — score, vector/lexical ranks, rerank rank, rank delta;
  4. reranked candidates — score, calibrated probability, kept/dropped;
  5. final context — token count, chunk ids, refusal decision, top
     probability, threshold, `citation_render_miss`.

Documents other than the committed fixture have no PDF served by this build;
the pane says so instead of pretending to resolve a span in the wrong file.

## Eval page

`/eval` reads a report from, in order:

1. `/api/eval-report.json` (proxied to the API's `/eval-report.json`),
2. `/eval-report.json` (static, if one is placed in `public/`),
3. `/api/eval/report`.

If none exists it says so and tells you to run `uv run docqa eval`; it never
prints invented rates. A recognised rate table is shown with its denominator
and Wilson 95% CI when present; the raw JSON is always available in a
disclosure.

The **three-case failure gallery** is implemented below the report. It renders
the committed `src/fixtures/failure-gallery.json`: calibrated refusal with
competing passages, a claim dropped for `quote_mismatch`, and vector-only
retrieval losing to hybrid on the rare term.

## Highlights and tests

- `src/fixtures.test.ts` verifies 20/20 committed citations resolve against
  the stored PDF.js page strings (zero misses), that the offline demo resolves,
  and that an absent quote is detected as a miss.
- `e2e/highlight.spec.ts` drives the real browser: load the demo, click the
  `quokka` claim, assert page 3 renders, assert `CSS.highlights` contains a
  range whose text contains the quote, and assert the drawer shows
  `citation_render_miss = 0` and `20/20 resolve`. It also exercises every demo
  claim and checks `/eval` renders three gallery cards with an honest empty
  report state.

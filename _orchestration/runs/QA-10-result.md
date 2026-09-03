# QA-10 — static entry for the `/eval` route

Branch: `qa-10-eval-route` (from `master`). Not merged and not deployed; the
supervisor rebuilds the static site.

## Root cause

The app is a single-page Vite build published as a Render **static site** with
no rewrite/fallback rule. `App.tsx` selects the evaluation view by reading
`window.location.pathname` (`isEvalRoute()`), and the header links to `/eval`
with a plain `<a href="/eval">`. A link click that triggers a full page load (or
a bookmark, or a shared URL) therefore asks the static host for `/eval`, which
has no file behind it and returns `404`. The home page works only because `/`
maps to `dist/index.html`.

## Fix

`ui/scripts/emit-routes.mjs`, appended to the `ui` build after the existing
steps:

```
tsc -b && vite build && node scripts/check-demo-guard.mjs && node scripts/emit-routes.mjs
```

The script:

1. **Derives the routes from `src/App.tsx`**, not from a hard-coded list. It
   reads two independent representations:
   - the router predicate `isEvalRoute()` (`path === '/eval'`);
   - the primary `<nav>` links (`href="/eval"`).
2. **Fails the build when they disagree.** A link with no router branch, or a
   router branch with no link, throws `emit-routes: ... disagree`.
3. **Copies `dist/index.html` byte-for-byte** (a `Buffer` write) to
   `dist/<route>/index.html`, creating the directory. The document therefore
   keeps the same hashed asset URLs and module preload.

The built `index.html` references its entry chunk at `/assets/index-*.js` (Vite
base `/`). The copy is under `/eval/` but the asset URL is still resolved from
the root, so no rebase is needed. The script asserts that the entry script is
root-absolute and refuses to emit if a future `base` change makes it relative.

No router, route, design, copy, or dependency changed. Only the build script,
the vitest `include` glob, and the new script/test files differ.

## Proof

Clean build (`rm -rf dist && npm ci && npm run build`) ends with:

```
demo guard: no offline fixture in the production bundle
emit-routes: /eval → dist\eval\index.html
```

- `dist/eval/index.html` exists and `cmp dist/index.html dist/eval/index.html`
  reports **byte-identical**.
- Mismatch guard, run against an isolated copy of the script and a synthetic
  `App.tsx` whose nav links `/eval` and `/report` while the router only serves
  `/eval`: **exit 1** with
  `emit-routes: the navigation links [/eval, /report] and the router [/eval] disagree`.
- `vite preview` on a fresh port (SPA fallback is active there — `/nope` also
  returns 200, so preview alone is not the mechanism proof):

  ```
  200 /
  200 /eval
  200 /eval/
  200 /assets/index-DIo2Ciy8.js
  200 /nope        (preview's own SPA fallback)
  ```

- A plain static server with **no fallback**, which is the Render situation
  (`python -m http.server --directory dist`):

  ```
  200 /
  301 /eval        (SimpleHTTPRequestHandler redirects to the directory)
  200 /eval/
  200 /assets/index-DIo2Ciy8.js
  404 /nope
  ```

  The `/eval/` body is byte-identical to `dist/index.html`.

- Chromium (project Playwright 1.64, against `vite preview`):

  ```json
  {
    "eval": { "h1": "Measure the answer, not the demo.", "evalPage": 1, "askView": 0 },
    "home": { "h1": "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
              "evalPage": 0, "askView": 1 }
  }
  ```

  `/eval` renders the evaluation view, not the home view; `/` still renders the
  home view.

Unit coverage in `ui/scripts/emit-routes.test.mjs` (6 tests): derivation from
the real `App.tsx`, both mismatch directions, a missing router predicate,
byte-identical emission, and refusal of a relative entry path.

## Verification

- `bash scripts/verify.sh` → **exit 0**: ruff clean, `mypy --strict` clean
  (27 source files), **208 pytest tests passed**, 1 pre-existing Starlette
  deprecation warning; k6 prints `SKIP` as documented.
- `npm --prefix ui run verify` → lint clean, typecheck clean, **25/25 UI tests
  passed in 5 files** (19 existing + 6 new).
- `npm --prefix ui run build` → exit 0 after a clean `npm ci`.

## Anything left

- The deployed static site was not rebuilt or touched; the supervisor rebuilds
  it. Once published, the `/eval` entry will exist as a file.
- Render's static service maps the directory index itself; the local no-fallback
  server redirects `/eval` to `/eval/` (301) before serving the document. Either
  behaviour is answered from the emitted file; no rewrite rule is involved.
- The derivation deliberately requires a route to be both in the router
  predicate and in the primary nav. A future route reachable only by URL, with
  no nav link, will fail the build instead of silently shipping without an
  entry; that is the intended disagreement guard.

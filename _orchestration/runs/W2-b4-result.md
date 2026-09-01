# W2-b4 — documentation truth: result

Branch `w2-b-hosting`. No code path, calibration file, corpus or
`portfolio-site/` file was touched. `eval/calibration.json` is unchanged
(`git diff eval/calibration.json` empty). No visibility change, no force-push,
no merge to `master`.

## What changed

| file | change |
|---|---|
| `README.md` | Leads with both live URLs; states the keyed generator (`gemini-flash-lite-latest`, retry `gemini-3.8-flash`), the pinned judge (`gemini-3.8-flash`), the judge's unavailability and the rule-based fallback; records the free instance's 512 MB / 0.1 CPU / scale-to-zero / ~1 min cold start and serial 20 s spacing; replaces the stale two-of-five-gates status with the five passing gates from `eval-report.json`. The keyless-endpoint paragraph and every local-only claim are gone. |
| `docs/PROOF.md` | Row 21 is verified against the hosted UI and API; the "unverified by design" row is gone. New **Hosted pair** section records both URLs, service and deploy ids, the `/ready` counts, the `/ask` transcript (200, 27.5 s, one claim, one citation, chunk 128, page 8, chars 2613–4041, quote), the deploy commands, the direct database quote check, and the 512 MB measurements with their reproduction command. Row 19 now reports 147 tests. Stale `file:line` references corrected. |
| `docs/ENV.md` | New **Hosted services** section lists every variable the hosted services read (`DOCQA_DATABASE_URL`, `DOCQA_MODEL_*`, `DOCQA_CORS_ORIGINS`, `DOCQA_ONNX_THREADS`, `DOCQA_RERANK_BATCH_SIZE`, `DOCQA_ONNX_MEMORY_ARENA`, `DOCQA_ONNX_CONCURRENCY`, `DOCQA_HF_CACHE`, `PORT`), the fixed pool settings, and what the free instance imposes. It states that the free plan ignores `preDeployCommand` and that migration/ingest are operator steps. Corrects the non-existent `docqa db upgrade` command and the stale 778.6 s cost figure; the demo cache is described without the retired model id. |
| `docs/EVAL.md` | Historical runs are labelled historical; the retired keyless endpoint is described without naming it; the current keyed cost (710.3 s, 26 requests, 20 s spacing) is stated. The gate table is unchanged. |
| `docs/INTERVIEW.md` | #10/#11 updated to the current 15/24 answer rate and 6/6 refusal; #13 records the judge's `[429, 429, 503]`; #19 now points at the live pair; #16 notes `DOCQA_ONNX_CONCURRENCY=1` on the free instance. |
| `docs/INVARIANTS.md` | Invariant 7 notes the hosted concurrency of 1; the stale "SSE streams validated claims" sentence is replaced with the true non-streaming contract; line references corrected. |
| `docs/ABLATION.md`, `docs/CORPUS.md` | Dates removed (required by the card); no measurement changed. |
| `render.yaml` | Keeps `VITE_API_BASE=https://marginalia-api-4pvl.onrender.com/`. `branch` corrected from `master` to `w2-b-hosting`, which is the branch both live services actually track. No `preDeployCommand` or other field the free plan ignores. |

All `YYYY-MM-DD` patterns were removed from `README.md` and `docs/*.md`. The
required greps are clean:

```sh
grep -rn "pollinations\|openai-fast\|runs locally" README.md docs/*.md   # no output
grep -rnE "20[0-9]{2}-[0-9]{2}-[0-9]{2}" README.md docs/*.md             # no output
grep -rn "unverified by design" README.md docs/*.md                       # no output
```

## Note on the eval figures

The card's context quoted faithfulness `9/11`. That is the **W2-a** run, not the
final W2-a2 gate run. The committed `eval-report.json` records **12/15 (80.0%)**
faithfulness and **13/15 (86.7%)** citation precision, and the docs use the
committed artifact. The gate table is otherwise exactly as supplied: recall 18/24,
answer 15/24, refusal 6/6.

## Reproducing every hosted number

`DOCQA_DATABASE_URL` and `RENDER_API_KEY` are read from
`P:/portfolio/.secrets/host.env`; no value is printed here.

### Health, readiness and corpus

```sh
curl -sS https://marginalia-api-4pvl.onrender.com/health
# 200; "status":"ok","structured_output":"tool","uncalibrated":false

curl -sS https://marginalia-api-4pvl.onrender.com/ready
# 200; documents 5, chunks 260, expected_documents 5,
#      migration current=head=0001_initial, corpus ok

curl -sS -o /dev/null -w '%{http_code}\n' https://marginalia-ui.onrender.com/
# 200
```

### UI bundle carries the API base

```sh
curl -sS https://marginalia-ui.onrender.com/ -o /tmp/ui.html
grep -o '/assets/[^"]*\.js' /tmp/ui.html
# /assets/index-Dxcar6HP.js
curl -sS https://marginalia-ui.onrender.com/assets/index-Dxcar6HP.js \
  | grep -o 'https://marginalia-api-4pvl.onrender.com/'
# https://marginalia-api-4pvl.onrender.com/
```

### Deploy ids

```sh
curl -sS -H "Authorization: Bearer $RENDER_API_KEY" \
  "https://api.render.com/v1/services/srv-dbho7udg1s2s73aj3kd0/deploys?limit=1"
# dep-dbhoumvlk1mc73dah260 | live | commit df50ee85
curl -sS -H "Authorization: Bearer $RENDER_API_KEY" \
  "https://api.render.com/v1/services/srv-dbho81dg1s2s73aj3rlg/deploys?limit=1"
# dep-dbhoun0n74is73elb3n0 | live | commit df50ee85
```

### `/ask` transcript

```sh
curl -sS -m 180 -X POST https://marginalia-api-4pvl.onrender.com/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"Why is SBERT more efficient than a BERT cross-encoder?"}'
```

Recorded run: HTTP 200 in 27.5 s, `cached: false`, one claim, one citation,
`chunk_id 128`, `sentence-bert` page 8, chars 2613–4041, quote
`SBERT is computationally efficient. On a GPU,`.

Reproduction variance: the generator is non-deterministic. A repeat during this
pass returned one validated claim citing chunk 127 at 25.0 s, and two other calls
returned validated refusals (`validation_failed`, `insufficient_context`). The
citation contract is the invariant; the exact citation is not.

### The cited span in the hosted database

```sh
PGPORT=5432 psql -Atc \
  "select id, page_no, char_start, char_end from chunks where id = 128" \
  "$DOCQA_DATABASE_URL"
# 128|8|2613|4041

PGPORT=5432 psql -Atc \
  "select strpos(normalize(text, NFKC), 'SBERT is computationally efficient. On a GPU,') > 0 \
   from chunks where id = 128" \
  "$DOCQA_DATABASE_URL"
# t
```

(`PGPORT=5432` is required because this machine exports `PGPORT=5433` for its
local database and the Neon URL omits a port.)

### 512 MB measurements

```sh
git show df50ee8 --format=%B -s
# imports 123 MB, embedder 291 MB, reranker 354 MB, retrieval 358 MB,
# idle 380 MB; rerank of 20 passages in one batch 796 MB (+439 MB);
# one pair at a time, CPU arena off: +31 MB, bit-identical scores.
```

The measurements were taken with the release image and the Neon corpus under
`--memory=512m`. The change set is in `df50ee8`: lazy reranker load,
`DOCQA_RERANK_BATCH_SIZE=1`, `DOCQA_ONNX_MEMORY_ARENA=0`,
`DOCQA_ONNX_THREADS=1`, `DOCQA_ONNX_CONCURRENCY=1`, a 2 + 2 asyncpg pool with a
300 s recycle, and `/ready` corpus counts.

### Eval gates

```sh
python -c "import json; r=json.load(open('eval-report.json')); [print(g['name'], g['successes'], g['n'], round(g['rate'],3), g['passed']) for g in r['gates']]"
# retrieval_recall_at_8 18 24 0.75 True
# answer_rate_answerable 15 24 0.625 True
# unanswerable_refusal_rate 6 6 1.0 True
# citation_precision 13 15 0.867 True
# faithfulness 12 15 0.8 True
```

The gate run itself: `bash scripts/live.sh eval` → exit 0, `run.wall_clock_s`
710.3, `model_calls.requests` 26, `retries` 0, `rate_limited` 0.

### Verification

```sh
bash scripts/verify.sh
# ruff: All checks passed
# mypy --strict: Success: no issues found in 21 source files
# pytest: 147 passed, 1 warning
# SKIP k6: ...
# verify: PASS (lint, types, tests; k6 skipped and documented)
# exit 0
```

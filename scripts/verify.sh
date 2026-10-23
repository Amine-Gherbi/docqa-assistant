#!/usr/bin/env bash
# One local gate: lint, types, unit/integration tests, then the concurrency
# scenario when k6 is installed. The k6 step is never reported as passing when
# the binary is absent; it prints SKIP and PROOF.md labels it unverified.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "== ruff =="
uv run ruff check src tests scripts

echo "== mypy --strict =="
uv run mypy --strict src

echo "== pytest =="
uv run pytest

if [ "${DOCQA_WITH_K6:-0}" = "1" ] && command -v k6 >/dev/null 2>&1; then
  echo "== k6: 4 concurrent askers =="
  uv run docqa serve --port 8788 > "${TMPDIR:-/tmp}/docqa-verify-api.log" 2>&1 &
  API_PID=$!
  cleanup() {
    kill "$API_PID" 2>/dev/null || true
  }
  trap cleanup EXIT
  ready=0
  for _ in $(seq 1 120); do
    if uv run python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8788/health', timeout=1)" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 1
  done
  if [ "$ready" -ne 1 ]; then
    echo "FAIL: API did not become healthy on 127.0.0.1:8788" >&2
    exit 1
  fi
  # CPU-only hosts: DOCQA_K6_P95_MS=9000 documents the measured local budget.
  K6_P95_MS="${DOCQA_K6_P95_MS:-5000}" k6 run scripts/k6-ask.js
  echo "verify: PASS (including k6)"
else
  echo "SKIP k6: loopback latency gate is opt-in (DOCQA_WITH_K6=1); measured numbers live in docs/ENV.md"
  echo "verify: PASS (lint, types, tests; k6 skipped and documented)"
fi

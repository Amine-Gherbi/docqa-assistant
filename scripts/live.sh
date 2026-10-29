#!/usr/bin/env bash
# Run a docqa command against the keyed Gemini free tier.
#
# The key comes from P:/portfolio/.secrets/model.env (outside every repository)
# or from the environment. The file is parsed, never sourced, and no value is
# ever echoed. The free tier answers one request at a time: the client holds
# one call in flight and spaces request starts by DOCQA_MODEL_MIN_INTERVAL_S.
# HTTP 429/503 and timeouts are retried with exponential backoff; the first
# retry uses the documented fallback id, then the run fails loudly.
#
# The keyless public endpoint is gone. It was a stand-in, not a fallback.
#
# Usage:
#   bash scripts/live.sh probe --json
#   bash scripts/live.sh ask --question "How many identical layers ...?"
#   bash scripts/live.sh warm
#   bash scripts/live.sh eval
#   bash scripts/live.sh eval --force-generation   # diagnostic, not a gate run
set -euo pipefail

SECRETS_FILE="${DOCQA_SECRETS_FILE:-P:/portfolio/.secrets/model.env}"

if [ -f "$SECRETS_FILE" ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%$'\r'}"
    case "$line" in
      ''|'#'*) continue ;;
    esac
    name="${line%%=*}"
    value="${line#*=}"
    case "$value" in
      \"*\") value="${value:1:${#value}-2}" ;;
      \'*\') value="${value:1:${#value}-2}" ;;
    esac
    case "$name" in
      DOCQA_MODEL_API_KEY|DOCQA_JUDGE_API_KEY)
        if [ -n "$value" ]; then
          export "$name=$value"
        fi
        ;;
    esac
  done < "$SECRETS_FILE"
fi

if [ -z "${DOCQA_MODEL_API_KEY:-}" ]; then
  echo "REFUSED: DOCQA_MODEL_API_KEY is not set." >&2
  echo "Put it in $SECRETS_FILE as DOCQA_MODEL_API_KEY=... (or export it)." >&2
  echo "The keyless public endpoint was removed; there is no anonymous fallback." >&2
  exit 2
fi

export DOCQA_MODEL_BASE_URL="${DOCQA_MODEL_BASE_URL:-https://generativelanguage.googleapis.com/v1beta/openai/}"
export DOCQA_MODEL_NAME="${DOCQA_MODEL_NAME:-gemini-flash-lite-latest}"
export DOCQA_MODEL_FALLBACK_NAMES="${DOCQA_MODEL_FALLBACK_NAMES:-gemini-3.8-flash}"
export DOCQA_MODEL_TIMEOUT_S="${DOCQA_MODEL_TIMEOUT_S:-180}"
export DOCQA_MODEL_MIN_INTERVAL_S="${DOCQA_MODEL_MIN_INTERVAL_S:-20}"
export DOCQA_MODEL_MAX_RETRIES="${DOCQA_MODEL_MAX_RETRIES:-2}"
export DOCQA_MODEL_RETRY_BACKOFF_S="${DOCQA_MODEL_RETRY_BACKOFF_S:-5}"
export DOCQA_JUDGE_API_KEY="${DOCQA_JUDGE_API_KEY:-$DOCQA_MODEL_API_KEY}"

cd "$(dirname "$0")/.."
exec uv run docqa "$@"

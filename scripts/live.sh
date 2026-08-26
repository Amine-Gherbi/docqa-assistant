#!/usr/bin/env bash
# Run a docqa command against the free keyless model endpoint.
#
# The endpoint is public and rate-limited to about one answer per 30 s per IP.
# The client serialises calls and spaces request starts by 31 s
# (DOCQA_MODEL_MIN_INTERVAL_S). No secret is used or stored.
#
# Usage:
#   bash scripts/live.sh probe --json
#   bash scripts/live.sh ask --question "How many identical layers ...?"
#   bash scripts/live.sh warm
#   bash scripts/live.sh eval
#   bash scripts/live.sh eval --force-generation   # diagnostic, not a gate run
set -euo pipefail

export DOCQA_MODEL_BASE_URL="${DOCQA_MODEL_BASE_URL:-https://text.pollinations.ai/openai}"
export DOCQA_MODEL_NAME="${DOCQA_MODEL_NAME:-openai-fast}"
export DOCQA_MODEL_API_KEY=""
export DOCQA_MODEL_TIMEOUT_S="${DOCQA_MODEL_TIMEOUT_S:-90}"

cd "$(dirname "$0")/.."
exec uv run docqa "$@"

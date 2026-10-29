# ADR 0004 — Structured output: forced `submit_answer`, with a JSON fallback

- **Status:** Accepted
- **Date:** 2026-10-23
- **Supersedes:** none
- **Related:** R3 (answer + citations), `docs/ENV.md`, `src/docqa/answer.py`

## Context

The citation guarantee depends on the answer arriving in one machine-checkable
shape: `{claims:[{text, citation_ids, quote}], insufficient_evidence, reason?}`.
The planned mechanism is one forced chat-completions tool call named
`submit_answer`. A model gateway sits between this service and the model, and
gateways sometimes ignore or rewrite `tool_choice` — some downgrade the request
to an ordinary completion. If that happened silently, answers would lose their
structure and the citation gate would have nothing to validate.

The project has no model credential and must not commit one. The endpoint is
therefore read from `DOCQA_MODEL_BASE_URL`, `DOCQA_MODEL_API_KEY`, and
`DOCQA_MODEL_NAME`. The measured demo uses a public, keyless OpenAI-compatible
endpoint (`scripts/live.sh`, no secret); when no endpoint is configured, the
answer path runs a deterministic extractive fallback that still emits the same
payload shape, so the citation gate is exercised end to end without a model.

## Decision

1. **Primary path: one forced tool call.** `OpenAICompatibleModel` sends
   `tools=[{function: submit_answer, parameters: <answer schema>}]` and
   `tool_choice={"type":"function","function":{"name":"submit_answer"}}`.
   Generation is not streamed. The response must contain a `submit_answer` call
   whose arguments parse as JSON; `json-repair` is applied before giving up.
2. **Probe before trusting the gateway.** `uv run docqa probe` sends a real HTTP
   request that forces `submit_answer` and reports whether a tool call came
   back. It exits `0` when the call was honoured, `5` when the gateway returned
   content instead, and `1` on a transport failure. The verdict is written to
   `eval/structured-output.json` and recorded in `docs/ENV.md`. A CI job must
   run it before any eval that assumes tool mode.
3. **Documented fallback: JSON-only prompt + repair.** Setting
   `DOCQA_STRUCTURED_OUTPUT=json` (or `docqa serve --structured-output json`)
   removes `tools` and `tool_choice`, appends an instruction to return one JSON
   object, and parses `message.content` with `json-repair`. Both modes feed the
   *same* server-side validation, repair turn, and refusal path.
4. **The fallback is a deployment switch, not an automatic downgrade.** The
   service does not silently switch mid-request if a tool call is missing: the
   missing call is a schema failure, which triggers the single repair turn and
   then the refusal path. Silent mode changes would make the probe verdict
   meaningless.

## Verification status

The probe was first exercised against a local loopback stub, then against the
keyed Gemini free tier (`https://generativelanguage.googleapis.com/v1beta/openai/`,
model `gemini-flash-lite-latest`). That request carried
`tool_choice={"type":"function","function":{"name":"submit_answer"}}` and the
gateway returned a `submit_answer` tool call; the verdict is
`eval/structured-output.json` (`mode: "tool"`, `tool_call_returned: true`,
14.531 s). The retired keyless stand-in is recorded in `docs/EVAL.md`. The
recorded verdict is therefore:

- client serialisation, forced `tool_choice`, and response parsing:
  **verified** against the keyed gateway;
- JSON fallback: **verified against the loopback stub and unit tests**, not
  needed against this gateway (tool mode was honoured, so JSON mode was not
  probed);
- a gateway that rewrites `tool_choice`: **not observed here**. The probe is the
  control for that case; if it exits `5`, deploy with
  `DOCQA_STRUCTURED_OUTPUT=json`, and update `docs/ENV.md` and this ADR.

## Consequences

- A gateway that rewrites `tool_choice` is detected by an explicit, recorded
  check rather than discovered through malformed answers.
- The JSON fallback costs one extra parse/repair step but shares every
  validation guarantee with the tool path.
- The answer schema is duplicated in the prompt and in `_ANSWER_SCHEMA`; both
  must change together.
- When `DOCQA_MODEL_BASE_URL` is unset, answers come from the labelled
  deterministic extractive fallback. It is not a language model and must not be
  presented as one; every response says
  `"model": "extractive fallback (no model key)"` and
  `"structured_output": "extractive"`. When a live endpoint is configured,
  there is no silent fallback to it: a 429 after two retries surfaces as
  `RateLimitedError` (HTTP 429), and other failures as `model_unavailable`
  (HTTP 502).

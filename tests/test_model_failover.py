"""Keyed-endpoint behaviour: retries, the answering model id, and failover.

No test here contacts the network. Every call goes through
``httpx.MockTransport``; the free tier's serial gate is exercised with
``min_interval_s=0`` so the suite stays fast.
"""

from __future__ import annotations

import json

import httpx
import pytest

from docqa.answer import (
    FailoverModel,
    ModelCallError,
    ModelRequest,
    OpenAICompatibleModel,
)


def _request() -> ModelRequest:
    return ModelRequest(question="q", system_prompt="s", user_prompt="u", chunks=())


def _tool_body(model: str = "gemini-flash-lite-latest") -> dict[str, object]:
    return {
        "model": model,
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "submit_answer",
                                "arguments": (
                                    '{"claims": [], "insufficient_evidence": true}'
                                ),
                            }
                        }
                    ]
                }
            }
        ],
    }


async def test_503_is_retried_then_succeeds() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(503, text="high demand")
        return httpx.Response(200, json=_tool_body())

    model = OpenAICompatibleModel(
        "http://busy.invalid/v1",
        "gemini-flash-lite-latest",
        transport=httpx.MockTransport(handler),
        min_interval_s=0,
        max_retries=2,
        retry_backoff_s=0.001,
    )
    reply = await model.complete(_request())

    assert calls == 3
    assert reply.error is None
    assert model.stats.retries == 2


async def test_timeout_is_retried_then_reported_as_a_model_error() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("too slow", request=request)

    model = OpenAICompatibleModel(
        "http://slow.invalid/v1",
        "gemini-flash-lite-latest",
        transport=httpx.MockTransport(handler),
        min_interval_s=0,
        max_retries=1,
        retry_backoff_s=0.001,
    )
    reply = await model.complete(_request())

    assert calls == 2
    assert reply.payload is None
    assert reply.error is not None
    assert "Timeout" in reply.error


async def test_reply_records_the_answering_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=_tool_body(model="gemini-3.8-flash")
        )

    model = OpenAICompatibleModel(
        "http://answer.invalid/v1",
        "gemini-flash-lite-latest",
        transport=httpx.MockTransport(handler),
        min_interval_s=0,
    )
    reply = await model.complete(_request())

    assert reply.model == "gemini-3.8-flash"


async def test_failover_uses_the_second_model_and_records_it() -> None:
    def first(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["model"] == "gemini-flash-lite-latest"
        return httpx.Response(404, text="model not found")

    def second(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["model"] == "gemini-3.8-flash"
        return httpx.Response(200, json=_tool_body(model="gemini-3.8-flash"))

    model = FailoverModel(
        [
            OpenAICompatibleModel(
                "http://chain.invalid/v1",
                "gemini-flash-lite-latest",
                transport=httpx.MockTransport(first),
                min_interval_s=0,
                max_retries=0,
            ),
            OpenAICompatibleModel(
                "http://chain.invalid/v1",
                "gemini-3.8-flash",
                transport=httpx.MockTransport(second),
                min_interval_s=0,
                max_retries=0,
            ),
        ]
    )
    reply = await model.complete(_request())

    assert reply.error is None
    assert reply.model == "gemini-3.8-flash"
    assert model.requested_model == "gemini-flash-lite-latest"
    assert model.stats.requests == 2


async def test_failover_fails_loudly_when_every_model_fails() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="broken")

    def client(name: str, transport: httpx.MockTransport) -> OpenAICompatibleModel:
        return OpenAICompatibleModel(
            "http://chain.invalid/v1",
            name,
            transport=transport,
            min_interval_s=0,
            max_retries=0,
        )

    model = FailoverModel(
        [
            client("gemini-flash-lite-latest", httpx.MockTransport(handler)),
            client("gemini-3.8-flash", httpx.MockTransport(handler)),
        ]
    )
    with pytest.raises(ModelCallError) as raised:
        await model.complete(_request())

    message = str(raised.value)
    assert "gemini-flash-lite-latest" in message
    assert "gemini-3.8-flash" in message


async def test_reply_records_finish_reason_mode_and_excerpt() -> None:
    body = _tool_body()
    choices = body["choices"]
    assert isinstance(choices, list)
    choice = choices[0]
    assert isinstance(choice, dict)
    choice["finish_reason"] = "length"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    model = OpenAICompatibleModel(
        "http://answer.invalid/v1",
        "gemini-flash-lite-latest",
        transport=httpx.MockTransport(handler),
        min_interval_s=0,
    )
    reply = await model.complete(_request())

    assert reply.finish_reason == "length"
    assert reply.structured_mode == "tool"
    assert reply.http_status == 200
    assert reply.response_excerpt
    assert len(reply.response_excerpt) <= 200


async def test_request_override_sends_a_json_only_body() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": (
                                '{"claims": [], "insufficient_evidence": true}'
                            )
                        },
                    }
                ]
            },
        )

    model = OpenAICompatibleModel(
        "http://fallback.invalid/v1",
        "gemini-flash-lite-latest",
        transport=httpx.MockTransport(handler),
        min_interval_s=0,
    )
    reply = await model.complete(
        ModelRequest(
            question="q",
            system_prompt="s",
            user_prompt="u",
            chunks=(),
            structured_output="json",
        )
    )

    assert "tools" not in seen
    assert "tool_choice" not in seen
    assert reply.structured_mode == "json"
    assert reply.finish_reason == "stop"
    assert reply.payload == {"claims": [], "insufficient_evidence": True}

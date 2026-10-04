"""LLM adapters without network: request/response shapes for Bedrock and Anthropic.

Bedrock requests are checked by botocore's own ``Stubber``, which validates
parameters and responses against the installed service model, so a wrong
shape fails here, offline. The credentials below are dummy strings that
botocore requires to build a client; no request leaves the process.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import boto3
import pytest
from botocore.stub import ANY, Stubber

from yield_triage.agent.llm import (
    AnthropicLLM,
    BedrockLLM,
    ToolSpec,
    require_model_id,
    to_anthropic_messages,
    to_bedrock_messages,
)

TOOLS = [ToolSpec("get_window_summary", "summary", {"type": "object", "properties": {}})]
HISTORY: list[dict[str, Any]] = [
    {"role": "user", "text": "investigate"},
    {
        "role": "assistant",
        "text": "",
        "tool_calls": [{"id": "t1", "name": "get_window_summary", "arguments": {"start": "a"}}],
    },
    {
        "role": "tool",
        "tool_call_id": "t1",
        "name": "get_window_summary",
        "content": "{}",
        "is_error": False,
    },
    {"role": "user", "text": "now summarise"},
]


def test_bedrock_messages_alternate_and_merge_tool_results() -> None:
    out = to_bedrock_messages(HISTORY)
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert out[2]["content"][0]["toolResult"]["status"] == "success"
    assert out[2]["content"][1] == {"text": "now summarise"}


def test_bedrock_converse_request_validated_by_botocore() -> None:
    client = boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id="dummy",
        aws_secret_access_key="dummy",  # noqa: S106  # pragma: allowlist secret
    )
    response = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {"text": "checking"},
                    {
                        "toolUse": {
                            "toolUseId": "t2",
                            "name": "get_window_summary",
                            "input": {"start": "x"},
                        }
                    },
                ],
            }
        },
        "stopReason": "tool_use",
        "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
        "metrics": {"latencyMs": 1},
    }
    with Stubber(client) as stub:
        stub.add_response(
            "converse",
            response,
            {
                "modelId": "test-model",
                "system": ANY,
                "messages": ANY,
                "inferenceConfig": ANY,
                "toolConfig": ANY,
            },
        )
        result = BedrockLLM(model_id="test-model", client=client).complete("sys", HISTORY, TOOLS)
        stub.assert_no_pending_responses()
    assert result.text == "checking"
    assert result.tool_calls[0].name == "get_window_summary"
    assert result.tool_calls[0].arguments == {"start": "x"}


def test_anthropic_messages_and_response_parsing() -> None:
    out = to_anthropic_messages(HISTORY)
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert out[2]["content"][0]["type"] == "tool_result"

    captured: dict[str, Any] = {}

    def create(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="ok"),
                SimpleNamespace(
                    type="tool_use", id="t3", name="get_window_summary", input={"a": 1}
                ),
            ]
        )

    fake = SimpleNamespace(messages=SimpleNamespace(create=create))
    result = AnthropicLLM(model_id="test-model", client=fake).complete("sys", HISTORY, TOOLS)
    assert (
        captured["model"] == "test-model" and captured["tools"][0]["name"] == "get_window_summary"
    )
    assert result.text == "ok" and result.tool_calls[0].id == "t3"


@pytest.mark.parametrize(
    "env", [{}, {"YIELD_TRIAGE_MODEL_ID": ""}, {"YIELD_TRIAGE_MODEL_ID": "CHANGE_ME"}]
)
def test_model_id_has_no_default(env: dict[str, str]) -> None:
    with pytest.raises(RuntimeError, match="YIELD_TRIAGE_MODEL_ID"):
        require_model_id(env)

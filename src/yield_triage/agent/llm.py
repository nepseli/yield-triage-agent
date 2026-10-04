"""LLM access behind one small protocol, with three implementations.

What: a provider-neutral message format (plain dicts), the ``LLMClient``
protocol (``complete(system, messages, tools) -> LLMResponse``), and:
  * ``ScriptedLLM``  - deterministic steps, used by tests, smoke and eval;
  * ``BedrockLLM``   - AWS Bedrock Converse API (default for live runs);
  * ``AnthropicLLM`` - Anthropic Messages API (optional extra).

Why: the agent graph must not care which model answers, and tests must not
need a network or credentials. Credentials come only from the standard
provider chains (AWS profile/SSO/env; ``ANTHROPIC_API_KEY``). The model ID
comes from ``YIELD_TRIAGE_MODEL_ID`` with no default, so a stale ID is never
baked in.

Connects to: ``agent/graph.py`` calls ``complete``; ``agent/scenarios.py``
builds ScriptedLLM scripts. Request and response shapes were checked against
the installed botocore service model and anthropic SDK types.

Neutral message format (all JSON-serialisable, so LangGraph can checkpoint it):
  {"role": "user", "text": str}
  {"role": "assistant", "text": str, "tool_calls": [{"id", "name", "arguments"}]}
  {"role": "tool", "tool_call_id": str, "name": str, "content": str, "is_error": bool}
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

Message = dict[str, Any]
MODEL_ENV = "YIELD_TRIAGE_MODEL_ID"
MAX_TOKENS = 2000


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)

    def as_message(self) -> Message:
        calls = [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in self.tool_calls]
        return {"role": "assistant", "text": self.text, "tool_calls": calls}


class LLMClient(Protocol):
    def complete(
        self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> LLMResponse: ...


def require_model_id(env: dict[str, str] | None = None) -> str:
    env = dict(os.environ) if env is None else env
    model_id = env.get(MODEL_ENV, "").strip()
    if not model_id or model_id == "CHANGE_ME":
        raise RuntimeError(f"set {MODEL_ENV} to a model ID available in your account")
    return model_id


# ---------------------------------------------------------------- scripted

Step = Callable[[Sequence[Message], Sequence[ToolSpec]], LLMResponse]


class ScriptedLLM:
    """Returns pre-written responses in order; a step may inspect the history.

    It plays the role of the model in tests. It can be scripted to behave
    well, or to obey injected instructions, which is how the tests prove the
    policy layer holds even when the model is fooled.
    """

    def __init__(self, steps: Sequence[Step]) -> None:
        self._steps = list(steps)
        self.calls = 0

    def complete(
        self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> LLMResponse:
        if self.calls >= len(self._steps):
            return LLMResponse(text="(script finished)")
        step = self._steps[self.calls]
        self.calls += 1
        return step(messages, tools)


# ---------------------------------------------------------------- shared helpers


def _merge_same_role(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Both APIs require user/assistant alternation; merge adjacent turns."""
    merged: list[dict[str, Any]] = []
    for msg in messages:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"].extend(msg["content"])
        else:
            merged.append({"role": msg["role"], "content": list(msg["content"])})
    return merged


# ---------------------------------------------------------------- Bedrock


class BedrockLLM:
    """AWS Bedrock Converse API. Region and credentials: standard boto3 chain."""

    def __init__(self, model_id: str | None = None, client: Any = None) -> None:
        self.model_id = model_id or require_model_id()
        if client is None:
            import boto3

            client = boto3.client("bedrock-runtime")
        self.client = client

    def complete(
        self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> LLMResponse:
        request: dict[str, Any] = {
            "modelId": self.model_id,
            "system": [{"text": system}],
            "messages": to_bedrock_messages(messages),
            "inferenceConfig": {"maxTokens": MAX_TOKENS, "temperature": 0.0},
        }
        if tools:
            request["toolConfig"] = {
                "tools": [
                    {
                        "toolSpec": {
                            "name": t.name,
                            "description": t.description,
                            "inputSchema": {"json": t.input_schema},
                        }
                    }
                    for t in tools
                ]
            }
        return from_bedrock_response(self.client.converse(**request))


def to_bedrock_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m["role"] == "user":
            out.append({"role": "user", "content": [{"text": m["text"]}]})
        elif m["role"] == "assistant":
            content: list[dict[str, Any]] = [{"text": m["text"]}] if m.get("text") else []
            content += [
                {"toolUse": {"toolUseId": c["id"], "name": c["name"], "input": c["arguments"]}}
                for c in m.get("tool_calls", [])
            ]
            out.append({"role": "assistant", "content": content or [{"text": "."}]})
        else:
            block = {
                "toolUseId": m["tool_call_id"],
                "content": [{"text": m["content"]}],
                "status": "error" if m.get("is_error") else "success",
            }
            out.append({"role": "user", "content": [{"toolResult": block}]})
    return _merge_same_role(out)


def from_bedrock_response(response: dict[str, Any]) -> LLMResponse:
    blocks = response["output"]["message"]["content"]
    text = "".join(b["text"] for b in blocks if "text" in b)
    calls = [
        ToolCall(b["toolUse"]["toolUseId"], b["toolUse"]["name"], dict(b["toolUse"]["input"]))
        for b in blocks
        if "toolUse" in b
    ]
    return LLMResponse(text, calls)


# ---------------------------------------------------------------- Anthropic


class AnthropicLLM:
    """Anthropic Messages API. Key from ``ANTHROPIC_API_KEY`` via the SDK."""

    def __init__(self, model_id: str | None = None, client: Any = None) -> None:
        self.model_id = model_id or require_model_id()
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client

    def complete(
        self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> LLMResponse:
        response = self.client.messages.create(
            model=self.model_id,
            max_tokens=MAX_TOKENS,
            temperature=0.0,
            system=system,
            messages=to_anthropic_messages(messages),
            tools=[
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in tools
            ],
        )
        text = "".join(b.text for b in response.content if b.type == "text")
        calls = [
            ToolCall(b.id, b.name, dict(b.input)) for b in response.content if b.type == "tool_use"
        ]
        return LLMResponse(text, calls)


def to_anthropic_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m["role"] == "user":
            out.append({"role": "user", "content": [{"type": "text", "text": m["text"]}]})
        elif m["role"] == "assistant":
            content: list[dict[str, Any]] = (
                [{"type": "text", "text": m["text"]}] if m.get("text") else []
            )
            content += [
                {"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["arguments"]}
                for c in m.get("tool_calls", [])
            ]
            out.append({"role": "assistant", "content": content or [{"type": "text", "text": "."}]})
        else:
            out.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": m["tool_call_id"],
                            "content": m["content"],
                            "is_error": bool(m.get("is_error")),
                        }
                    ],
                }
            )
    return _merge_same_role(out)


def make_llm(kind: str) -> LLMClient:
    """Live clients only; scripted clients are built from ``agent/scenarios.py``."""
    if kind == "bedrock":
        return BedrockLLM()
    if kind == "anthropic":
        return AnthropicLLM()
    raise ValueError(f"unknown LLM kind {kind!r}")

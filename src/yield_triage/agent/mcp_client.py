"""MCP client side: start the server over stdio and expose it as a ``ToolPort``.

What: ``ToolPort`` is the tiny interface the graph needs (tool specs plus an
async ``call``). ``MCPToolPort`` implements it with the official MCP SDK
``Client``. ``server_params`` builds the subprocess launch with an explicit,
minimal environment.

Why: the agent talks to tools only through MCP, so every call crosses the
policy gateway. The launch environment contains only non-secret settings;
the SDK adds a fixed safe set (PATH and similar) and nothing else, so the
approval key never reaches the server (THREAT_MODEL.md T8).

Connects to: ``agent/graph.py`` (consumer), ``cli.py`` (launches it),
``server/app.py`` (the other end).
"""

from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Protocol

from mcp import Client
from mcp.client.stdio import StdioServerParameters

from yield_triage.agent.llm import ToolSpec
from yield_triage.config import ServerSettings


class ToolPort(Protocol):
    tool_specs: list[ToolSpec]

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


class MCPToolPort:
    def __init__(self, client: Client) -> None:
        self.client = client
        self.tool_specs: list[ToolSpec] = []

    async def refresh(self) -> None:
        listed = await self.client.list_tools()
        self.tool_specs = [
            ToolSpec(t.name, t.description or "", dict(t.input_schema)) for t in listed.tools
        ]

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = await self.client.call_tool(name, arguments)
        if result.structured_content is not None:
            return dict(result.structured_content)
        text = "".join(getattr(c, "text", "") for c in result.content)
        parsed: dict[str, Any] = json.loads(text)
        return parsed


@asynccontextmanager
async def connect(target: Any) -> AsyncIterator[MCPToolPort]:
    """``target`` is ``StdioServerParameters`` (real runs) or an in-process server (tests)."""
    async with Client(target) as client:
        port = MCPToolPort(client)
        await port.refresh()
        yield port


def server_params(settings: ServerSettings) -> StdioServerParameters:
    # SECURITY: T8 - explicit allowlist of variables for the server process.
    env = {
        "YIELD_TRIAGE_DATA_DIR": str(settings.data_dir),
        "YIELD_TRIAGE_STATE_DIR": str(settings.state_dir),
        "YIELD_TRIAGE_POLICY": str(settings.policy_path),
        "YIELD_TRIAGE_PROFILE": settings.profile,
        "YIELD_TRIAGE_RUN_ID": settings.run_id,
    }
    return StdioServerParameters(
        command=sys.executable, args=["-m", "yield_triage.server"], env=env
    )

"""MCP server over stdio, built on the official SDK's low-level ``Server``.

What: ``build_gateway`` assembles dataset, policy, audit log, ticket store and
budget into a ``PolicyGateway``; ``build_server`` exposes it over MCP with two
handlers: ``tools/list`` (only tools the active profile allows, with JSON
schemas generated from the pydantic argument models) and ``tools/call``
(forwarded verbatim to the gateway). ``main`` runs it on stdio.

Why the low-level server: the SDK's high-level ``MCPServer`` validates
arguments and rejects unknown tool names *before* user code runs, so those
denials would never reach our audit log. With the low-level server every call,
including hostile ones, goes through the gateway (DECISIONS.md D14).

Connects to: ``policy/gateway.py``; started by the agent (Phase 3) as a
subprocess via the MCP stdio client, or directly with
``python -m yield_triage.server``.
"""

from __future__ import annotations

import json
from typing import Any

import anyio
import mcp_types as types
from mcp.server import Server, ServerRequestContext
from mcp.server.stdio import stdio_server

from yield_triage import __version__
from yield_triage.audit import AuditLog
from yield_triage.config import ServerSettings
from yield_triage.data import Dataset, load_dataset
from yield_triage.policy.budget import RunBudget
from yield_triage.policy.config import Policy, load_policy
from yield_triage.policy.gateway import PolicyGateway
from yield_triage.policy.models import TOOL_ARG_MODELS
from yield_triage.server.tools import DESCRIPTIONS, ToolContext, load_notes
from yield_triage.tickets import TicketStore


def build_gateway(
    settings: ServerSettings, *, dataset: Dataset | None = None, policy: Policy | None = None
) -> PolicyGateway:
    policy = policy or load_policy(settings.policy_path)
    notes, notes_sha = load_notes()
    ctx = ToolContext(
        dataset=dataset or load_dataset(settings.data_dir),
        policy=policy,
        tickets=TicketStore(settings.tickets_dir),
        run_id=settings.run_id,
        notes=notes,
        notes_sha256=notes_sha,
    )
    budget = RunBudget(policy.limits.max_tool_calls, policy.limits.run_timeout_s)
    return PolicyGateway(
        policy=policy,
        profile=settings.profile,
        ctx=ctx,
        audit=AuditLog(settings.audit_path),
        budget=budget,
    )


def build_server(gateway: PolicyGateway) -> Server[Any]:
    async def list_tools(
        ctx: ServerRequestContext[Any], params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        tools = [
            types.Tool(
                name=name,
                description=DESCRIPTIONS[name],
                input_schema=TOOL_ARG_MODELS[name].model_json_schema(),
            )
            for name in gateway.allowed_tools()
        ]
        return types.ListToolsResult(tools=tools)

    async def call_tool(
        ctx: ServerRequestContext[Any], params: types.CallToolRequestParams
    ) -> types.CallToolResult:
        envelope = await gateway.call(params.name, params.arguments or {})
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(envelope, sort_keys=True))],
            structured_content=envelope,
            is_error=envelope["decision"] == "deny",
        )

    return Server(
        "yield-triage",
        version=__version__,
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )


async def _serve(server: Server[Any]) -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    gateway = build_gateway(ServerSettings.from_env())
    anyio.run(_serve, build_server(gateway))

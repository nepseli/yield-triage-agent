"""The triage agent as a LangGraph state graph with a human approval pause.

What: nodes ``triage -> plan -> tool_loop (repeats) -> summarize ->
propose -> await_approval``. State is checkpointed to SQLite after every
step. ``await_approval`` calls LangGraph's ``interrupt``, which ends the
process's run; a later ``resume`` (a new process, after a human has run
``yield-triage approve`` and ``commit``) continues from the checkpoint and
checks whether the ticket was committed.

Why: the graph makes the control flow explicit and reviewable. The model
picks tools and writes the summary; everything else (the order of steps, the
turn limit, the pause for a human) is fixed code. The agent never commits:
commit needs a token signed with a key this process never loads.

Connects to: ``agent/llm.py`` (the model), ``agent/mcp_client.py`` (tools
via MCP), ``tickets.py`` (read-only check of commit status), ``cli.py``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt

from yield_triage.agent.llm import LLMClient, Message
from yield_triage.agent.mcp_client import ToolPort
from yield_triage.agent.prompts import (
    ACT_INSTRUCTION,
    SUMMARY_INSTRUCTION,
    SYSTEM_PROMPT,
    task_message,
)
from yield_triage.tickets import TicketStore

MAX_TURNS = 6  # agent-side cap; the server's call budget is the hard limit


class AgentState(TypedDict, total=False):
    run_id: str
    start: str
    end: str
    status: str  # running | no_data | failed | ticket_denied | awaiting_approval | committed
    error: str
    messages: list[Message]
    tool_log: list[dict[str, Any]]  # call_id, tool, decision, reason per call
    turns: int
    ticket: dict[str, Any]


@dataclass
class Deps:
    """Live objects for one process; never checkpointed."""

    tickets: TicketStore
    llm: LLMClient | None = None
    tools: ToolPort | None = None


def _llm(rt: Runtime[Deps]) -> LLMClient:
    if rt.context.llm is None:
        raise RuntimeError("this step needs an LLM")
    return rt.context.llm


def _tools(rt: Runtime[Deps]) -> ToolPort:
    if rt.context.tools is None:
        raise RuntimeError("this step needs the MCP tools")
    return rt.context.tools


async def _complete(rt: Runtime[Deps], messages: Sequence[Message]) -> Any:
    llm, tools = _llm(rt), _tools(rt)
    # boto3 / anthropic clients are blocking; keep the event loop free.
    return await asyncio.to_thread(llm.complete, SYSTEM_PROMPT, list(messages), tools.tool_specs)


def _log_entry(env: dict[str, Any]) -> dict[str, Any]:
    return {k: env.get(k) for k in ("call_id", "tool", "decision", "reason")}


# ---------------------------------------------------------------- nodes


async def triage(state: AgentState, runtime: Runtime[Deps]) -> AgentState:
    """Deterministic first look: the window summary, before any model call."""
    env = await _tools(runtime).call(
        "get_window_summary", {"start": state["start"], "end": state["end"]}
    )
    log = [_log_entry(env)]
    if env["decision"] != "allow" or not env["data"].get("n_units"):
        return {"status": "no_data", "error": env.get("reason") or "empty window", "tool_log": log}
    first = {"role": "user", "text": task_message(state["start"], state["end"], json.dumps(env))}
    return {"status": "running", "messages": [first], "tool_log": log, "turns": 0}


async def plan(state: AgentState, runtime: Runtime[Deps]) -> AgentState:
    response = await _complete(runtime, state["messages"])
    # A plan is text only; any tool calls the model sneaks in here are dropped.
    planned: Message = {"role": "assistant", "text": response.text or "(no plan)", "tool_calls": []}
    act = {"role": "user", "text": ACT_INSTRUCTION}
    return {"messages": [*state["messages"], planned, act]}


async def tool_loop(state: AgentState, runtime: Runtime[Deps]) -> AgentState:
    response = await _complete(runtime, state["messages"])
    messages = [*state["messages"], response.as_message()]
    log = list(state["tool_log"])
    for call in response.tool_calls:
        # Forwarded as-is, even unknown tools: the server decides, not us.
        env = await _tools(runtime).call(call.name, call.arguments)
        log.append(_log_entry(env))
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.name,
                "content": json.dumps(env),
                "is_error": env.get("decision") == "deny",
            }
        )
    return {"messages": messages, "tool_log": log, "turns": state["turns"] + 1}


def after_tool_loop(state: AgentState) -> Literal["tool_loop", "summarize"]:
    last = state["messages"][-1]
    out_of_budget = any(str(e.get("reason", "")).startswith("budget_") for e in state["tool_log"])
    if last["role"] == "tool" and state["turns"] < MAX_TURNS and not out_of_budget:
        return "tool_loop"
    return "summarize"


async def summarize(state: AgentState, runtime: Runtime[Deps]) -> AgentState:
    messages = [*state["messages"], {"role": "user", "text": SUMMARY_INSTRUCTION}]
    response = await _complete(runtime, messages)
    messages.append({"role": "assistant", "text": response.text, "tool_calls": []})
    draft = _extract_json(response.text)
    if draft is None:
        return {"messages": messages, "status": "failed", "error": "summary_not_json"}
    return {"messages": messages, "ticket": {"draft": draft}}


async def propose(state: AgentState, runtime: Runtime[Deps]) -> AgentState:
    draft = state["ticket"]["draft"]
    # The draft is sent exactly as the model wrote it; the gateway validates it.
    env = await _tools(runtime).call("propose_ticket", draft)
    log = [*state["tool_log"], _log_entry(env)]
    if env["decision"] != "allow":
        return {"tool_log": log, "status": "ticket_denied", "error": str(env.get("reason"))}
    return {
        "tool_log": log,
        "status": "awaiting_approval",
        "ticket": {**state["ticket"], **env["data"]},
    }


def await_approval(state: AgentState, runtime: Runtime[Deps]) -> AgentState:
    ticket_id = state["ticket"]["ticket_id"]
    # Pauses here (state saved). Resumed by `yield-triage resume <run_id>`.
    interrupt({"ticket_id": ticket_id, "next": f"yield-triage approve {ticket_id}"})
    if runtime.context.tickets.is_committed(ticket_id):
        return {"status": "committed", "ticket": {**state["ticket"], "status": "committed"}}
    return {"status": "awaiting_approval"}


# ---------------------------------------------------------------- wiring


def _route_status(ok: str) -> Any:
    def route(state: AgentState) -> str:
        return END if state.get("status") in {"no_data", "failed", "ticket_denied"} else ok

    return route


def build_graph() -> StateGraph[AgentState, Deps, AgentState, AgentState]:
    g = StateGraph(AgentState, context_schema=Deps)
    for name, fn in [
        ("triage", triage),
        ("plan", plan),
        ("tool_loop", tool_loop),
        ("summarize", summarize),
        ("propose", propose),
        ("await_approval", await_approval),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "triage")
    g.add_conditional_edges("triage", _route_status("plan"))
    g.add_edge("plan", "tool_loop")
    g.add_conditional_edges("tool_loop", after_tool_loop)
    g.add_conditional_edges("summarize", _route_status("propose"))
    g.add_conditional_edges("propose", _route_status("await_approval"))
    g.add_conditional_edges(
        "await_approval", lambda s: END if s.get("status") == "committed" else "await_approval"
    )
    return g


def _extract_json(text: str) -> dict[str, Any] | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _public(state: dict[str, Any]) -> dict[str, Any]:
    keys = ("run_id", "status", "error", "ticket", "tool_log", "turns")
    out = {k: state[k] for k in keys if k in state}
    if "ticket" in out:
        out["ticket"] = {k: v for k, v in out["ticket"].items() if k != "draft"}
    return out


async def run_agent(
    *, run_id: str, start: str, end: str, deps: Deps, checkpoint_path: Path
) -> dict[str, Any]:
    """Run until the graph ends or pauses for approval; return the public state."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    config: Any = {"configurable": {"thread_id": run_id}}
    async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
        graph = build_graph().compile(checkpointer=saver)
        state = await graph.ainvoke(
            {"run_id": run_id, "start": start, "end": end}, config, context=deps
        )
    return _public(dict(state))


async def resume_agent(*, run_id: str, deps: Deps, checkpoint_path: Path) -> dict[str, Any]:
    """Continue a paused run after the human step; needs no LLM and no tools."""
    if not checkpoint_path.exists():
        raise FileNotFoundError("no checkpoint store; run the agent first")
    config: Any = {"configurable": {"thread_id": run_id}}
    async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
        graph = build_graph().compile(checkpointer=saver)
        snapshot = await graph.aget_state(config)
        if not snapshot.next:
            return _public(dict(snapshot.values))
        state = await graph.ainvoke(Command(resume="checked"), config, context=deps)
    return _public(dict(state))

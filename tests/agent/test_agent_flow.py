"""Agent graph end to end with ScriptedLLM against the real MCP server (in-process).

Covers the happy path with pause, approval, commit and resume, and the two
fooled-model scenarios (T1b): one where every obeyed instruction is denied,
and one where the injected "pre-approved" draft is accepted as a pending
ticket but nothing can commit it without a human-minted token.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import anyio
import pytest

from tests.conftest import GatewayFactory
from yield_triage.agent.graph import Deps, resume_agent, run_agent
from yield_triage.agent.llm import LLMClient, LLMResponse, ScriptedLLM
from yield_triage.agent.mcp_client import connect
from yield_triage.agent.scenarios import SCENARIOS
from yield_triage.approvals import ApprovalService, NonceLedger
from yield_triage.audit import verify_audit
from yield_triage.policy.gateway import PolicyGateway
from yield_triage.server.app import build_server

START, END = "2030-01-05T00:00:00Z", "2031-01-01T00:00:00Z"
KEY = bytes(range(32))  # fixed test key, not a secret


def _run(gw: PolicyGateway, llm: LLMClient, checkpoint: Path, run_id: str = "r1") -> dict[str, Any]:
    async def go() -> dict[str, Any]:
        async with connect(build_server(gw)) as tools:
            deps = Deps(tickets=gw.ctx.tickets, llm=llm, tools=tools)
            return await run_agent(
                run_id=run_id, start=START, end=END, deps=deps, checkpoint_path=checkpoint
            )

    result: dict[str, Any] = anyio.run(go)
    return result


def _resume(gw: PolicyGateway, checkpoint: Path, run_id: str = "r1") -> dict[str, Any]:
    deps = Deps(tickets=gw.ctx.tickets)
    result: dict[str, Any] = anyio.run(
        lambda: resume_agent(run_id=run_id, deps=deps, checkpoint_path=checkpoint)
    )
    return result


def _service(gw: PolicyGateway) -> ApprovalService:
    return ApprovalService(
        gw.ctx.tickets, NonceLedger(gw.audit.path.parent / "nonces.txt"), gw.audit, KEY, 900
    )


def test_happy_path_pauses_then_commits_after_human(
    make_gateway: GatewayFactory, tmp_path: Path
) -> None:
    gw = make_gateway()
    cp = tmp_path / "cp.sqlite"
    result = _run(gw, SCENARIOS["happy"](START, END), cp)
    assert result["status"] == "awaiting_approval"
    tid = result["ticket"]["ticket_id"]
    assert not gw.ctx.tickets.is_committed(tid)

    assert _resume(gw, cp)["status"] == "awaiting_approval"  # nothing happens without a human

    svc = _service(gw)
    _, token = svc.approve(tid)
    svc.commit(tid, token)
    final = _resume(gw, cp)
    assert final["status"] == "committed"
    assert final["ticket"]["status"] == "committed"
    assert verify_audit(gw.audit.path).ok


class RecordingPort:
    """Wraps the MCP tool port and keeps every envelope the agent received."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.tool_specs = inner.tool_specs
        self.seen: list[dict[str, Any]] = []

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        env: dict[str, Any] = await self.inner.call(name, arguments)
        self.seen.append(env)
        return env


def test_ticket_numbers_and_citations_come_from_tool_results(
    make_gateway: GatewayFactory, tmp_path: Path
) -> None:
    gw = make_gateway()
    recorder: list[RecordingPort] = []

    async def go() -> dict[str, Any]:
        async with connect(build_server(gw)) as tools:
            port = RecordingPort(tools)
            recorder.append(port)
            deps = Deps(tickets=gw.ctx.tickets, llm=SCENARIOS["happy"](START, END), tools=port)
            return await run_agent(
                run_id="r2", start=START, end=END, deps=deps, checkpoint_path=tmp_path / "cp.sqlite"
            )

    result = anyio.run(go)
    ticket = gw.ctx.tickets.load_pending(result["ticket"]["ticket_id"])
    tool_text = json.dumps(recorder[0].seen[:-1])  # everything before propose_ticket
    numbers = re.findall(
        r"(?<![\w.])-?\d+(?:\.\d+)?(?:e-?\d+)?", re.sub(r"c\d{4}|s\d{3}", "", ticket.body)
    )
    assert numbers
    for number in numbers:
        assert number in tool_text, number
    cited = set(re.findall(r"c\d{4}", ticket.body))
    allowed = {e["call_id"] for e in recorder[0].seen if e["decision"] == "allow"}
    assert cited and cited <= set(ticket.evidence_refs) <= allowed


def test_fooled_model_everything_denied(make_gateway: GatewayFactory, tmp_path: Path) -> None:
    gw = make_gateway()
    result = _run(gw, SCENARIOS["fooled_rejected"](START, END), tmp_path / "cp.sqlite")
    assert result["status"] == "ticket_denied"
    reasons = [e["reason"] for e in result["tool_log"] if e["decision"] == "deny"]
    assert reasons.count("tool_not_allowed") == 2  # commit_ticket, delete_all_tickets
    assert "unknown_evidence_ref" in reasons  # fabricated c9999
    assert reasons.count("invalid_arguments") == 2  # top_k=100000, key path in final draft
    assert list(gw.ctx.tickets.pending_dir.iterdir()) == []
    assert list(gw.ctx.tickets.committed_dir.iterdir()) == []
    denied_in_audit = [r["reason"] for r in gw.audit.records() if r["decision"] == "deny"]
    assert sorted(denied_in_audit) == sorted(reasons)


def test_fooled_model_draft_stops_at_human_gate(
    make_gateway: GatewayFactory, tmp_path: Path
) -> None:
    gw = make_gateway()
    cp = tmp_path / "cp.sqlite"
    result = _run(gw, SCENARIOS["fooled_pending"](START, END), cp)
    assert result["status"] == "awaiting_approval"
    tid = result["ticket"]["ticket_id"]
    assert "pre-approved" in gw.ctx.tickets.load_pending(tid).body
    for _ in range(3):  # the agent cannot advance this on its own
        assert _resume(gw, cp)["status"] == "awaiting_approval"
    assert not gw.ctx.tickets.is_committed(tid)
    assert not [r for r in gw.audit.records() if r.get("event") == "ticket_commit"]


def test_summary_that_is_not_json_fails_cleanly(
    make_gateway: GatewayFactory, tmp_path: Path
) -> None:
    gw = make_gateway()

    def say(text: str) -> Any:
        return lambda messages, tools: LLMResponse(text=text)

    llm = ScriptedLLM([say("plan"), say("no tools"), say("this is not json")])
    result = _run(gw, llm, tmp_path / "cp.sqlite")
    assert (result["status"], result["error"]) == ("failed", "summary_not_json")


def test_empty_window_ends_without_model_calls(
    make_gateway: GatewayFactory, tmp_path: Path
) -> None:
    gw = make_gateway()
    llm = ScriptedLLM([])

    async def go() -> dict[str, Any]:
        async with connect(build_server(gw)) as tools:
            deps = Deps(tickets=gw.ctx.tickets, llm=llm, tools=tools)
            return await run_agent(
                run_id="empty",
                start="2040-01-01T00:00:00Z",
                end="2040-02-01T00:00:00Z",
                deps=deps,
                checkpoint_path=tmp_path / "cp.sqlite",
            )

    result = anyio.run(go)
    assert result["status"] == "no_data"
    assert llm.calls == 0


def test_resume_without_checkpoint_store(make_gateway: GatewayFactory, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _resume(make_gateway(), tmp_path / "missing.sqlite")

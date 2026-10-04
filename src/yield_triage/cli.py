"""``yield-triage`` command line: run the agent, approve, commit, verify.

What: subcommands
  run           start the MCP server over stdio and run the agent graph until
                it pauses for approval (or ends);
  resume        continue a paused run after the human step;
  approve       HUMAN: show a pending ticket and mint a single-use token;
  commit        HUMAN: finalise a ticket with that token;
  verify-audit  check the audit hash chain;
  serve         run the MCP server on stdio (same as python -m yield_triage.server).
Machine-readable results go to stdout as JSON; human-facing text to stderr.

Why: approval is deliberately not an MCP tool. It is a separate command a
person runs, in a process that loads the signing key. ``run`` removes the key
variables from its own environment before doing anything, and the server
subprocess receives an explicit allowlist of variables.

Connects to: ``agent/graph.py``, ``agent/mcp_client.py``, ``approvals.py``,
``audit.py``, ``server/app.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

import anyio

from yield_triage.approvals import (
    KEY_ENV,
    KEY_FILE_ENV,
    ApprovalError,
    ApprovalService,
    NonceLedger,
    load_signing_key,
)
from yield_triage.audit import AuditLog, verify_audit
from yield_triage.config import ServerSettings
from yield_triage.policy.config import load_policy
from yield_triage.tickets import TicketStore


def _emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _checkpoint_path(settings: ServerSettings) -> Path:
    return settings.state_dir / "checkpoints.sqlite"


def scrub_agent_environment() -> None:
    """Remove approval-key variables from this process before the agent runs."""
    # SECURITY: T8 - the agent process must not hold the signing key.
    for name in (KEY_ENV, KEY_FILE_ENV):
        os.environ.pop(name, None)


def cmd_run(args: argparse.Namespace) -> int:
    scrub_agent_environment()
    from yield_triage.agent.graph import Deps, run_agent
    from yield_triage.agent.llm import LLMClient, make_llm
    from yield_triage.agent.mcp_client import connect, server_params
    from yield_triage.agent.scenarios import SCENARIOS

    settings = replace(ServerSettings.from_env(), run_id=args.run_id or uuid.uuid4().hex[:12])
    llm: LLMClient = (
        SCENARIOS[args.scenario](args.start, args.end)
        if args.llm == "scripted"
        else make_llm(args.llm)
    )

    async def go() -> dict[str, Any]:
        async with connect(server_params(settings)) as tools:
            deps = Deps(tickets=TicketStore(settings.tickets_dir), llm=llm, tools=tools)
            return await run_agent(
                run_id=settings.run_id,
                start=args.start,
                end=args.end,
                deps=deps,
                checkpoint_path=_checkpoint_path(settings),
            )

    result = anyio.run(go)
    _emit(result)
    if result.get("status") == "awaiting_approval":
        tid = result["ticket"]["ticket_id"]
        print(f"Paused for approval. A human runs: yield-triage approve {tid}", file=sys.stderr)
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    scrub_agent_environment()
    from yield_triage.agent.graph import Deps, resume_agent

    settings = ServerSettings.from_env()
    deps = Deps(tickets=TicketStore(settings.tickets_dir))
    result = anyio.run(
        lambda: resume_agent(
            run_id=args.run_id, deps=deps, checkpoint_path=_checkpoint_path(settings)
        )
    )
    _emit(result)
    return 0


def _service(settings: ServerSettings) -> ApprovalService:
    policy = load_policy(settings.policy_path)
    return ApprovalService(
        store=TicketStore(settings.tickets_dir),
        ledger=NonceLedger(settings.ledger_path),
        audit=AuditLog(settings.audit_path),
        key=load_signing_key(),
        ttl_s=policy.approval.token_ttl_s,
    )


def cmd_approve(args: argparse.Namespace) -> int:
    settings = ServerSettings.from_env()
    store = TicketStore(settings.tickets_dir)
    try:
        ticket = store.load_pending(args.ticket_id)
    except (ValueError, FileNotFoundError):
        print("No such pending ticket.", file=sys.stderr)
        return 2
    print(
        f"\nTicket {ticket.ticket_id} (run {ticket.run_id})\nTitle: {ticket.title}\n\n"
        f"{ticket.body}\n\nEvidence: {', '.join(ticket.evidence_refs)}\n"
        f"Content SHA-256: {ticket.content_hash}\n",
        file=sys.stderr,
    )
    if not args.yes:
        answer = input("Approve this exact content? [y/N] ")
        if answer.strip().lower() != "y":
            print("Not approved.", file=sys.stderr)
            return 1
    service = _service(settings)
    _, token = service.approve(ticket.ticket_id)
    _emit(
        {
            "ticket_id": ticket.ticket_id,
            "content_hash": ticket.content_hash,
            "token": token,
            "expires_in_s": service.ttl_s,
        }
    )
    return 0


def cmd_commit(args: argparse.Namespace) -> int:
    settings = ServerSettings.from_env()
    token = sys.stdin.readline().strip() if args.token == "-" else args.token  # noqa: S105
    try:
        ticket = _service(settings).commit(args.ticket_id, token)
    except ApprovalError as err:
        _emit({"ticket_id": args.ticket_id, "committed": False, "reason": err.reason})
        return 2
    _emit({"ticket_id": ticket.ticket_id, "committed": True, "content_hash": ticket.content_hash})
    return 0


def cmd_verify_audit(args: argparse.Namespace) -> int:
    path = Path(args.path) if args.path else ServerSettings.from_env().audit_path
    result = verify_audit(path)
    _emit(
        {
            "path": str(path),
            "ok": result.ok,
            "n_records": result.n_records,
            "first_bad_seq": result.first_bad_seq,
            "problem": result.problem,
        }
    )
    return 0 if result.ok else 1


def cmd_serve(args: argparse.Namespace) -> int:
    from yield_triage.server.app import main as serve

    serve()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="yield-triage", description="Yield triage agent (proof of concept)"
    )
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the agent until it pauses for approval")
    run.add_argument("--start", required=True, help="ISO-8601, e.g. 2030-01-12T00:00:00Z")
    run.add_argument("--end", required=True)
    run.add_argument("--llm", choices=["bedrock", "anthropic", "scripted"], default="bedrock")
    run.add_argument("--scenario", default="happy", help="scripted scenario name")
    run.add_argument("--run-id", default=None)
    run.set_defaults(func=cmd_run)

    resume = sub.add_parser("resume", help="continue a paused run after approval")
    resume.add_argument("run_id")
    resume.set_defaults(func=cmd_resume)

    approve = sub.add_parser("approve", help="HUMAN: approve a pending ticket, print a token")
    approve.add_argument("ticket_id")
    approve.add_argument("--yes", action="store_true", help="skip the interactive confirmation")
    approve.set_defaults(func=cmd_approve)

    commit = sub.add_parser("commit", help="HUMAN: commit a ticket with an approval token")
    commit.add_argument("ticket_id")
    commit.add_argument("--token", required=True, help="token, or '-' to read it from stdin")
    commit.set_defaults(func=cmd_commit)

    verify = sub.add_parser("verify-audit", help="verify the audit log hash chain")
    verify.add_argument("--path", default=None)
    verify.set_defaults(func=cmd_verify_audit)

    serve = sub.add_parser("serve", help="run the MCP server on stdio")
    serve.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code: int = args.func(args)
    return code


if __name__ == "__main__":
    sys.exit(main())

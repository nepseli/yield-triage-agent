"""T3-T8, T12: approval tokens and the commit path.

A ticket is proposed through the gateway exactly as the agent would, then the
human-side ``ApprovalService`` is attacked: no token, replay, wrong ticket,
edited ticket, expired token, forged or malformed token. Every refusal must be
audited and must leave the ticket uncommitted.
"""

from __future__ import annotations

import json
import os
import re
import stat
import sys
from pathlib import Path

import pytest

from tests.conftest import ALL_TIME, GatewayFactory, call, last_audit
from yield_triage.approvals import (
    KEY_ENV,
    KEY_FILE_ENV,
    ApprovalError,
    ApprovalService,
    NonceLedger,
    load_signing_key,
    mint_token,
)
from yield_triage.policy.gateway import PolicyGateway

KEY = bytes(range(32))  # fixed test key, obviously not a real secret


class FakeClock:
    def __init__(self, now: float = 1_900_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _propose(gw: PolicyGateway, title: str = "Sensor shift on s001") -> str:
    call(gw, "get_window_summary", ALL_TIME)
    env = call(
        gw,
        "propose_ticket",
        {"title": title, "body": "Evidence in c0001 shows a shift.", "evidence_refs": ["c0001"]},
    )
    assert env["decision"] == "allow", env
    assert env["data"]["status"] == "pending"
    return str(env["data"]["ticket_id"])


def _service(gw: PolicyGateway, clock: FakeClock, key: bytes = KEY) -> ApprovalService:
    state = gw.audit.path.parent
    return ApprovalService(
        store=gw.ctx.tickets,
        ledger=NonceLedger(state / "used_nonces.txt"),
        audit=gw.audit,
        key=key,
        ttl_s=900,
        clock=clock,
    )


def _assert_denied(svc: ApprovalService, ticket_id: str, token: str | None, reason: str) -> None:
    with pytest.raises(ApprovalError) as err:
        svc.commit(ticket_id, token)
    assert err.value.reason == reason
    rec = svc.audit.records()[-1]
    assert (rec["event"], rec["decision"], rec["reason"]) == ("ticket_commit", "deny", reason)


def test_happy_path_commits_once(make_gateway: GatewayFactory) -> None:
    gw = make_gateway()
    clock = FakeClock()
    svc = _service(gw, clock)
    tid = _propose(gw)
    _, token = svc.approve(tid)
    svc.commit(tid, token)
    assert gw.ctx.tickets.is_committed(tid)
    events = [(r.get("event"), r["decision"]) for r in gw.audit.records()]
    assert events[-2:] == [("ticket_approved", "allow"), ("ticket_commit", "allow")]


def test_propose_alone_never_commits(make_gateway: GatewayFactory) -> None:
    gw = make_gateway()
    tid = _propose(gw)
    assert not gw.ctx.tickets.is_committed(tid)


def test_commit_without_token(make_gateway: GatewayFactory) -> None:
    gw = make_gateway()
    tid = _propose(gw)
    _assert_denied(_service(gw, FakeClock()), tid, None, "token_missing")
    assert not gw.ctx.tickets.is_committed(tid)


def test_token_replay(make_gateway: GatewayFactory) -> None:
    gw = make_gateway()
    svc = _service(gw, FakeClock())
    tid = _propose(gw)
    _, token = svc.approve(tid)
    svc.commit(tid, token)
    # Recreate the same draft content under the same ID and try the old token again.
    committed = gw.ctx.tickets.committed_dir / f"{tid}.json"
    raw = json.loads(committed.read_text())
    raw.pop("approval")
    raw["status"] = "pending"
    (gw.ctx.tickets.pending_dir / f"{tid}.json").write_text(json.dumps(raw))
    _assert_denied(svc, tid, token, "token_replayed")


def test_token_for_a_different_ticket(make_gateway: GatewayFactory) -> None:
    gw = make_gateway()
    svc = _service(gw, FakeClock())
    first, second = _propose(gw), _propose(gw, "Another ticket")
    _, token_for_first = svc.approve(first)
    _assert_denied(svc, second, token_for_first, "token_wrong_ticket")
    assert not gw.ctx.tickets.is_committed(second)


def test_token_after_ticket_is_edited(make_gateway: GatewayFactory) -> None:
    gw = make_gateway()
    svc = _service(gw, FakeClock())
    tid = _propose(gw)
    _, token = svc.approve(tid)
    path = gw.ctx.tickets.pending_dir / f"{tid}.json"
    raw = json.loads(path.read_text())
    raw["body"] = "Edited after approval: also shut down the line."
    path.write_text(json.dumps(raw))
    _assert_denied(svc, tid, token, "ticket_content_changed")


def test_expired_token(make_gateway: GatewayFactory) -> None:
    gw = make_gateway()
    clock = FakeClock()
    svc = _service(gw, clock)
    tid = _propose(gw)
    _, token = svc.approve(tid)
    clock.now += 901
    _assert_denied(svc, tid, token, "token_expired")


def test_token_from_another_key_is_rejected(make_gateway: GatewayFactory) -> None:
    gw = make_gateway()
    svc = _service(gw, FakeClock())
    tid = _propose(gw)
    ticket = gw.ctx.tickets.load_pending(tid)
    forged = mint_token(b"\x01" * 32, tid, ticket.content_hash, now=FakeClock().now, ttl_s=900)
    _assert_denied(svc, tid, forged, "token_bad_signature")


@pytest.mark.parametrize(
    "token", ["v1.ZmFrZQ.ZmFrZQ", "garbage", "v2.a.b", "v1." + "a" * 2000 + ".b"]
)
def test_malformed_or_injected_tokens(make_gateway: GatewayFactory, token: str) -> None:
    gw = make_gateway()
    tid = _propose(gw)
    with pytest.raises(ApprovalError) as err:
        _service(gw, FakeClock()).commit(tid, token)
    assert err.value.reason in {"token_malformed", "token_bad_signature"}
    assert last_audit(gw)["decision"] == "deny"


def test_tampered_payload_fails_signature(make_gateway: GatewayFactory) -> None:
    gw = make_gateway()
    svc = _service(gw, FakeClock())
    tid = _propose(gw)
    _, token = svc.approve(tid)
    version, payload, sig = token.split(".")
    flipped = payload[:-1] + ("A" if payload[-1] != "A" else "B")
    _assert_denied(svc, tid, f"{version}.{flipped}.{sig}", "token_bad_signature")


def test_commit_with_traversal_ticket_id(make_gateway: GatewayFactory) -> None:
    gw = make_gateway()
    _assert_denied(_service(gw, FakeClock()), "../../etc/passwd", "v1.a.b", "ticket_not_pending")
    assert last_audit(gw)["ticket_id"] == "<invalid>"


# ------------------------------------------------------------------ key handling (T8)


def test_key_from_env_must_be_long_enough() -> None:
    with pytest.raises(ValueError, match="at least"):
        load_signing_key({KEY_ENV: "abcd"})
    assert load_signing_key({KEY_ENV: "ab" * 32}) == bytes([0xAB]) * 32


def test_key_file_is_generated_owner_only(tmp_path: Path) -> None:
    path = tmp_path / "keys" / "approval.key"
    key = load_signing_key({KEY_FILE_ENV: str(path)})
    assert len(key) == 32 and path.exists()
    assert load_signing_key({KEY_FILE_ENV: str(path)}) == key  # stable once created
    if sys.platform != "win32":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits only")
def test_group_readable_key_file_refused(tmp_path: Path) -> None:
    path = tmp_path / "approval.key"
    path.write_text("ab" * 32)
    os.chmod(path, 0o644)
    with pytest.raises(PermissionError):
        load_signing_key({KEY_FILE_ENV: str(path)})


def test_mcp_stdio_client_does_not_pass_key_to_server(monkeypatch: pytest.MonkeyPatch) -> None:
    from mcp.client.stdio import get_default_environment

    monkeypatch.setenv(KEY_ENV, "ab" * 32)
    monkeypatch.setenv(KEY_FILE_ENV, "somewhere")
    env = get_default_environment()
    assert KEY_ENV not in env and KEY_FILE_ENV not in env


def test_server_code_never_loads_the_signing_key() -> None:
    server_src = Path(__file__).parents[2] / "src" / "yield_triage"
    for sub in ("server", "policy"):
        for py in (server_src / sub).rglob("*.py"):
            text = py.read_text(encoding="utf-8")
            assert "load_signing_key" not in text, py.name
            assert not re.search(r"^\s*(from|import)\s+yield_triage\.approvals", text, re.M), (
                py.name
            )

"""T10, T11, T13: audit chain tamper detection, no raw secrets, policy file checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from tests.conftest import ALL_TIME, GatewayFactory, call
from yield_triage.audit import AuditLog, verify_audit
from yield_triage.policy.config import DEFAULT_POLICY_PATH, load_policy


def _log_with(n: int, path: Path) -> AuditLog:
    log = AuditLog(path)
    for i in range(n):
        log.append(actor="test", event="e", n=i)
    return log


def _rewrite(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n")


def test_intact_chain_verifies(tmp_path: Path) -> None:
    log = _log_with(5, tmp_path / "a.jsonl")
    assert verify_audit(log.path).ok
    assert verify_audit(log.path).n_records == 5


def test_edited_record_detected(tmp_path: Path) -> None:
    log = _log_with(5, tmp_path / "a.jsonl")
    lines = log.path.read_text().splitlines()
    rec = json.loads(lines[2])
    rec["decision"] = "allow"  # an attacker rewriting history
    lines[2] = json.dumps(rec)
    _rewrite(log.path, lines)
    result = verify_audit(log.path)
    assert not result.ok and result.first_bad_seq == 3
    assert result.problem == "record content does not match its hash"


def test_rehashed_edit_still_breaks_the_link(tmp_path: Path) -> None:
    """Recomputing the edited record's own hash is not enough: the next link breaks."""
    from yield_triage.audit import record_hash

    log = _log_with(5, tmp_path / "a.jsonl")
    lines = log.path.read_text().splitlines()
    rec = json.loads(lines[1])
    rec["n"] = 99
    rec["hash"] = record_hash(rec)
    lines[1] = json.dumps(rec)
    _rewrite(log.path, lines)
    result = verify_audit(log.path)
    assert not result.ok and result.first_bad_seq == 3


def test_deleted_record_detected(tmp_path: Path) -> None:
    log = _log_with(5, tmp_path / "a.jsonl")
    lines = log.path.read_text().splitlines()
    del lines[2]
    _rewrite(log.path, lines)
    assert verify_audit(log.path).first_bad_seq == 3


def test_reordered_records_detected(tmp_path: Path) -> None:
    log = _log_with(5, tmp_path / "a.jsonl")
    lines = log.path.read_text().splitlines()
    lines[1], lines[2] = lines[2], lines[1]
    _rewrite(log.path, lines)
    assert not verify_audit(log.path).ok


def test_garbage_line_detected(tmp_path: Path) -> None:
    log = _log_with(2, tmp_path / "a.jsonl")
    with log.path.open("a") as fh:
        fh.write("not json\n")
    assert verify_audit(log.path).problem == "unparseable line"


def test_known_limitation_tail_truncation_not_detected(tmp_path: Path) -> None:
    """Documented in THREAT_MODEL.md: dropping the last records keeps a valid chain."""
    log = _log_with(5, tmp_path / "a.jsonl")
    _rewrite(log.path, log.path.read_text().splitlines()[:3])
    assert verify_audit(log.path).ok


def test_gateway_audit_chain_verifies(make_gateway: GatewayFactory) -> None:
    gw = make_gateway()
    for _ in range(3):
        call(gw, "get_window_summary", ALL_TIME)
    call(gw, "nope", {})
    assert verify_audit(gw.audit.path).ok


def test_audit_never_contains_raw_argument_values(make_gateway: GatewayFactory) -> None:
    gw = make_gateway()
    secret_like = "pretend-secret-" + "q" * 24  # pragma: allowlist secret
    call(gw, "get_window_summary", {**ALL_TIME, "api_key": secret_like})
    call(gw, "propose_ticket", {"title": secret_like, "body": "b" * 20, "evidence_refs": ["c0001"]})
    text = gw.audit.path.read_text()
    assert secret_like not in text
    assert all("args_sha256" in r for r in gw.audit.records())


# ------------------------------------------------------------------ policy file (T13)


def _policy_with(tmp_path: Path, mutate: object) -> Path:
    raw = yaml.safe_load(DEFAULT_POLICY_PATH.read_text())
    mutate(raw)  # type: ignore[operator]
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_profile_allowing_unknown_tool_rejected(tmp_path: Path) -> None:
    path = _policy_with(
        tmp_path, lambda r: r["profiles"]["read_only"]["allowed_tools"].append("shell")
    )
    with pytest.raises(ValidationError, match="unknown tool"):
        load_policy(path)


def test_profile_missing_scope_rejected(tmp_path: Path) -> None:
    path = _policy_with(
        tmp_path, lambda r: r["profiles"]["read_only"]["allowed_tools"].append("propose_ticket")
    )
    with pytest.raises(ValidationError, match="lacks scope"):
        load_policy(path)


def test_unknown_policy_key_rejected(tmp_path: Path) -> None:
    path = _policy_with(tmp_path, lambda r: r.update({"allow_everything": True}))
    with pytest.raises(ValidationError):
        load_policy(path)


def test_unknown_profile_rejected(make_gateway: GatewayFactory) -> None:
    with pytest.raises(ValueError, match="unknown policy profile"):
        make_gateway(profile="admin")

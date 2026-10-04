"""T2: malicious or malformed tool arguments are rejected and audited.

Each case asserts three things: the call is denied with the expected reason,
the audit log records the denial, and the offending raw value appears neither
in the response (no reflection) nor in the audit log (hash only).
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.conftest import ALL_TIME, GatewayFactory, call, last_audit

T0, T1 = ALL_TIME["start"], ALL_TIME["end"]

BAD_CALLS: list[tuple[str, str, Any, str]] = [
    # (case id, tool, args, expected reason)
    (
        "traversal-sensor",
        "check_sensor_drift",
        {"start": T0, "end": T1, "sensor_id": "../../etc/passwd"},
        "invalid_arguments",
    ),
    (
        "unknown-sensor",
        "check_sensor_drift",
        {"start": T0, "end": T1, "sensor_id": "s999"},
        "invalid_arguments",
    ),
    (
        "sensor-wrong-shape",
        "check_sensor_drift",
        {"start": T0, "end": T1, "sensor_id": "S001"},
        "invalid_arguments",
    ),
    (
        "ts-month-13",
        "get_window_summary",
        {"start": "2030-13-01T00:00:00Z", "end": T1},
        "invalid_arguments",
    ),
    ("ts-words", "get_window_summary", {"start": "yesterday", "end": T1}, "invalid_arguments"),
    ("ts-date-only", "get_window_summary", {"start": "2030-01-01", "end": T1}, "invalid_arguments"),
    (
        "ts-no-offset",
        "get_window_summary",
        {"start": "2030-01-01T00:00:00", "end": T1},
        "invalid_arguments",
    ),
    (
        "ts-sql",
        "get_window_summary",
        {"start": "2030-01-01T00:00:00Z' OR 1=1--", "end": T1},
        "invalid_arguments",
    ),
    ("ts-reversed", "get_window_summary", {"start": T1, "end": T0}, "invalid_arguments"),
    ("extra-key", "get_window_summary", {**ALL_TIME, "path": "/etc/passwd"}, "invalid_arguments"),
    ("missing-key", "get_window_summary", {"start": T0}, "invalid_arguments"),
    ("top_k-string", "rank_failing_sensors", {**ALL_TIME, "top_k": "5"}, "invalid_arguments"),
    ("top_k-zero", "rank_failing_sensors", {**ALL_TIME, "top_k": 0}, "invalid_arguments"),
    ("top_k-huge", "rank_failing_sensors", {**ALL_TIME, "top_k": 51}, "invalid_arguments"),
    ("args-not-object", "get_window_summary", ["2030-01-01T00:00:00Z"], "invalid_arguments"),
    (
        "args-oversized",
        "get_window_summary",
        {"start": "A" * 5000, "end": T1},
        "arguments_too_large",
    ),
    (
        "title-oversized",
        "propose_ticket",
        {"title": "x" * 121, "body": "b" * 20, "evidence_refs": ["c0001"]},
        "invalid_arguments",
    ),
    (
        "title-windows-path",
        "propose_ticket",
        {"title": "C:\\secrets\\x", "body": "b" * 20, "evidence_refs": ["c0001"]},
        "invalid_arguments",
    ),
    (
        "body-abs-path",
        "propose_ticket",
        {"title": "ok title", "body": "read /etc/shadow now", "evidence_refs": ["c0001"]},
        "invalid_arguments",
    ),
    (
        "body-control-chars",
        "propose_ticket",
        {"title": "ok title", "body": "clear screen \x1b[2J now", "evidence_refs": ["c0001"]},
        "invalid_arguments",
    ),
    (
        "evidence-not-call-id",
        "propose_ticket",
        {"title": "ok title", "body": "b" * 20, "evidence_refs": ["../x"]},
        "invalid_arguments",
    ),
    (
        "evidence-empty",
        "propose_ticket",
        {"title": "ok title", "body": "b" * 20, "evidence_refs": []},
        "invalid_arguments",
    ),
    ("tool-name-junk", "../../bin/sh", {}, "tool_not_allowed"),
]


@pytest.mark.parametrize(
    ("case", "tool", "args", "reason"), BAD_CALLS, ids=[c[0] for c in BAD_CALLS]
)
def test_bad_arguments_denied_and_audited(
    make_gateway: GatewayFactory, case: str, tool: str, args: Any, reason: str
) -> None:
    gw = make_gateway()
    call(gw, "get_window_summary", ALL_TIME)  # c0001 exists, so evidence refs are not the issue
    env = call(gw, tool, args)
    assert env["decision"] == "deny"
    assert env["reason"] == reason
    rec = last_audit(gw)
    assert (rec["decision"], rec["reason"], rec["call_id"]) == ("deny", reason, env["call_id"])
    _assert_not_reflected(gw.audit.path.read_text(), env, args)


def _assert_not_reflected(audit_text: str, env: dict[str, Any], args: Any) -> None:
    values = list(args.values()) if isinstance(args, dict) else list(args)
    for value in values:
        if isinstance(value, str) and len(value) > 3 and value not in (T0, T1):
            assert value not in str(env)
            assert value not in audit_text


def test_validation_error_names_fields_only(make_gateway: GatewayFactory) -> None:
    env = call(make_gateway(), "check_sensor_drift", {"start": T0, "end": T1, "sensor_id": "s999"})
    assert env["fields"] == ["sensor_id"]
    assert "s999" not in str(env)


def test_unknown_tool_name_is_logged_as_invalid(make_gateway: GatewayFactory) -> None:
    gw = make_gateway()
    call(gw, "rm -rf /", {})
    assert last_audit(gw)["tool"] == "<invalid>"


def test_valid_arguments_pass(make_gateway: GatewayFactory) -> None:
    gw = make_gateway()
    env = call(
        gw,
        "check_sensor_drift",
        {"start": "2030-01-05T00:00:00+08:00", "end": T1, "sensor_id": "s001"},
    )
    assert env["decision"] == "allow"

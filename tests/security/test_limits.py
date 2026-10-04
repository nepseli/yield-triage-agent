"""T9: unbounded consumption - call budget, wall clock, row cap, result size."""

from __future__ import annotations

import time
from typing import Any

import pytest

from tests.conftest import ALL_TIME, GatewayFactory, call, last_audit, with_limits
from yield_triage.audit import canonical_json
from yield_triage.policy.config import Policy
from yield_triage.server import tools


def test_call_budget_counts_allowed_and_denied(
    make_gateway: GatewayFactory, policy: Policy
) -> None:
    gw = make_gateway(policy_override=with_limits(policy, max_tool_calls=3))
    assert call(gw, "get_window_summary", ALL_TIME)["decision"] == "allow"
    assert call(gw, "nope", {})["reason"] == "tool_not_allowed"
    assert call(gw, "get_window_summary", ALL_TIME)["decision"] == "allow"
    env = call(gw, "get_window_summary", ALL_TIME)
    assert (env["decision"], env["reason"]) == ("deny", "budget_calls_exceeded")
    assert last_audit(gw)["reason"] == "budget_calls_exceeded"


def test_run_deadline_expired(make_gateway: GatewayFactory) -> None:
    gw = make_gateway()
    gw.budget.started -= gw.policy.limits.run_timeout_s + 1  # pretend the run started long ago
    env = call(gw, "get_window_summary", ALL_TIME)
    assert env["reason"] == "budget_time_exceeded"
    assert last_audit(gw)["reason"] == "budget_time_exceeded"


def test_slow_tool_is_cut_off(
    make_gateway: GatewayFactory, policy: Policy, monkeypatch: pytest.MonkeyPatch
) -> None:
    def slow(ctx: Any, args: Any) -> Any:
        time.sleep(2.0)
        raise AssertionError("should have been abandoned")

    monkeypatch.setitem(tools.HANDLERS, "get_window_summary", slow)
    gw = make_gateway(policy_override=with_limits(policy, run_timeout_s=0.3))
    started = time.monotonic()
    env = call(gw, "get_window_summary", ALL_TIME)
    assert env["reason"] == "budget_time_exceeded"
    assert time.monotonic() - started < 1.5
    assert last_audit(gw)["decision"] == "deny"


def test_row_cap(make_gateway: GatewayFactory, policy: Policy) -> None:
    gw = make_gateway(policy_override=with_limits(policy, max_rows=100))
    env = call(gw, "rank_failing_sensors", {**ALL_TIME, "top_k": 5})
    assert env["reason"] == "row_cap_exceeded"
    assert last_audit(gw)["reason"] == "row_cap_exceeded"
    small = {"start": "2030-01-01T00:00:00Z", "end": "2030-01-02T00:00:00Z"}
    assert call(gw, "get_window_summary", small)["decision"] == "allow"


def test_result_is_truncated_to_max_chars(make_gateway: GatewayFactory, policy: Policy) -> None:
    gw = make_gateway(policy_override=with_limits(policy, max_result_chars=900))
    env = call(gw, "rank_failing_sensors", {**ALL_TIME, "top_k": 50})
    assert env["decision"] == "allow"
    assert env["truncated"] is True
    assert len(canonical_json(env)) <= 900
    assert 0 < len(env["data"]["ranked"]) < 50
    assert last_audit(gw)["truncated"] is True


def test_untruncated_result_reports_false(make_gateway: GatewayFactory) -> None:
    env = call(make_gateway(), "get_window_summary", ALL_TIME)
    assert env["truncated"] is False

"""T8 for the agent side: the agent process and its server never hold the key."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from yield_triage.agent.mcp_client import server_params
from yield_triage.approvals import KEY_ENV, KEY_FILE_ENV
from yield_triage.cli import scrub_agent_environment
from yield_triage.config import ServerSettings


def test_run_scrubs_key_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY_ENV, "ab" * 32)
    monkeypatch.setenv(KEY_FILE_ENV, "somewhere")
    scrub_agent_environment()
    assert KEY_ENV not in os.environ and KEY_FILE_ENV not in os.environ


def test_server_launch_env_is_an_explicit_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY_ENV, "ab" * 32)
    params = server_params(ServerSettings.from_env({"YIELD_TRIAGE_RUN_ID": "x"}))
    assert params.env is not None
    assert set(params.env) == {
        "YIELD_TRIAGE_DATA_DIR",
        "YIELD_TRIAGE_STATE_DIR",
        "YIELD_TRIAGE_POLICY",
        "YIELD_TRIAGE_PROFILE",
        "YIELD_TRIAGE_RUN_ID",
    }


def test_agent_package_never_imports_approvals() -> None:
    agent_dir = Path(__file__).parents[2] / "src" / "yield_triage" / "agent"
    for py in agent_dir.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        assert not re.search(r"^\s*(from|import)\s+yield_triage\.approvals", text, re.M), py.name
        assert "load_signing_key" not in text, py.name


def test_commit_is_not_an_mcp_tool() -> None:
    from yield_triage.server.tools import HANDLERS

    assert not [name for name in HANDLERS if "commit" in name or "approve" in name]

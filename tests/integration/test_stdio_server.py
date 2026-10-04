"""The real server process over stdio, driven by the official MCP client.

Starts ``python -m yield_triage.server`` as a subprocess on a SYNTHETIC
dataset written to a temp dir, lists tools, calls every read tool, and checks
the audit chain the subprocess wrote. This is the same path the agent uses.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import anyio
from mcp import Client
from mcp.client.stdio import StdioServerParameters

from tests.conftest import ALL_TIME
from yield_triage.audit import verify_audit
from yield_triage.data import save_dataset
from yield_triage.synthetic import SyntheticConfig, make_synthetic

READ_CALLS: list[tuple[str, dict[str, Any]]] = [
    ("get_window_summary", ALL_TIME),
    ("rank_failing_sensors", {**ALL_TIME, "top_k": 5}),
    (
        "check_sensor_drift",
        {"start": "2030-01-10T00:00:00Z", "end": "2031-01-01T00:00:00Z", "sensor_id": "s001"},
    ),
    ("get_maintenance_notes", ALL_TIME),
]


def test_server_over_stdio(tmp_path: Path) -> None:
    ds, _ = make_synthetic(3, SyntheticConfig(n_units=400, n_sensors=60))
    save_dataset(ds, tmp_path / "data")
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "yield_triage.server"],
        env={
            "YIELD_TRIAGE_DATA_DIR": str(tmp_path / "data"),
            "YIELD_TRIAGE_STATE_DIR": str(tmp_path / "state"),
            "YIELD_TRIAGE_RUN_ID": "stdio-test",
        },
    )

    async def scenario() -> list[dict[str, Any]]:
        async with Client(params) as client:
            names = {t.name for t in (await client.list_tools()).tools}
            assert names == {name for name, _ in READ_CALLS} | {"propose_ticket"}
            out = []
            for name, args in READ_CALLS:
                res = await client.call_tool(name, args)
                assert not res.is_error, res.structured_content
                assert res.structured_content is not None
                out.append(res.structured_content)
            return out

    envelopes = anyio.run(scenario)
    assert [e["call_id"] for e in envelopes] == ["c0001", "c0002", "c0003", "c0004"]
    assert all(e["dataset"] == "synthetic" for e in envelopes)
    audit = tmp_path / "state" / "audit.jsonl"
    result = verify_audit(audit)
    assert result.ok and result.n_records == 4

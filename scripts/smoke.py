"""End-to-end smoke test using real processes only (``make smoke``).

What, in order, in a fresh temporary directory:
  1. generate SYNTHETIC data with ``scripts/fetch_data.py --synthetic``;
  2. start the MCP server over stdio and call every read tool;
  3. ``yield-triage run --llm scripted`` until it pauses for approval;
  4. ``yield-triage approve --yes`` (human step) and ``commit``;
  5. ``yield-triage resume`` and expect status ``committed``;
  6. assert the committed ticket file exists and ``verify-audit`` passes.
Exits non-zero on the first failure. The same script runs inside the Docker
image for the container smoke test.

Why: unit tests call functions; this proves the shipped commands work
together the way a user runs them.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import anyio
from mcp import Client
from mcp.client.stdio import StdioServerParameters

START, END = "2030-01-12T00:00:00Z", "2030-02-01T00:00:00Z"
READ_CALLS: list[tuple[str, dict[str, Any]]] = [
    ("get_window_summary", {"start": START, "end": END}),
    ("rank_failing_sensors", {"start": START, "end": END, "top_k": 5}),
    ("check_sensor_drift", {"start": START, "end": END, "sensor_id": "s000"}),
    ("get_maintenance_notes", {"start": START, "end": END}),
]


def step(msg: str) -> None:
    print(f"[smoke] {msg}", flush=True)


def expect(ok: object, detail: object) -> None:
    """Explicit check (not assert, which python -O would strip)."""
    if not ok:
        raise SystemExit(f"[smoke] FAILED: {detail}")


def run_cli(env: dict[str, str], *args: str) -> dict[str, Any]:
    proc = subprocess.run(  # noqa: S603 - fixed argv, our own CLI
        [sys.executable, "-m", "yield_triage.cli", *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if proc.returncode != 0:
        print(proc.stdout, proc.stderr, sep="\n", file=sys.stderr)
        raise SystemExit(f"[smoke] FAILED: yield-triage {args[0]} exited {proc.returncode}")
    result: dict[str, Any] = json.loads(proc.stdout)
    return result


async def call_read_tools(env: dict[str, str]) -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "yield_triage.server"],
        env={k: v for k, v in env.items() if k.startswith("YIELD_TRIAGE_")},
    )
    async with Client(params) as client:
        names = {t.name for t in (await client.list_tools()).tools}
        expect({n for n, _ in READ_CALLS} <= names, names)
        for name, args in READ_CALLS:
            res = await client.call_tool(name, args)
            expect(not res.is_error and res.structured_content, (name, res.structured_content))
            step(f"  {name}: allow ({res.structured_content['call_id']})")


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="yt-smoke-"))
    env = {
        **os.environ,
        "YIELD_TRIAGE_DATA_DIR": str(root / "data" / "synthetic"),
        "YIELD_TRIAGE_STATE_DIR": str(root / "state"),
        "YIELD_TRIAGE_KEY_FILE": str(root / "keys" / "approval.key"),
        "YIELD_TRIAGE_RUN_ID": "smoke-read",
    }
    env.pop("YIELD_TRIAGE_APPROVAL_KEY", None)

    step(f"workdir {root.name}: generating synthetic data")
    fetch = Path(__file__).with_name("fetch_data.py")
    subprocess.run(  # noqa: S603 - fixed argv, our own script
        [sys.executable, str(fetch), "--synthetic", "--out", str(root / "data")],
        env=env,
        check=True,
        capture_output=True,
        timeout=300,
    )

    step("MCP server over stdio: calling every read tool")
    anyio.run(call_read_tools, env)

    step("agent run (ScriptedLLM)")
    run = run_cli(
        env, "run", "--llm", "scripted", "--start", START, "--end", END, "--run-id", "smoke"
    )
    expect(run["status"] == "awaiting_approval", run)
    tid = run["ticket"]["ticket_id"]
    step(f"  paused for approval on {tid}")

    step("human approve + commit")
    token = run_cli(env, "approve", tid, "--yes")["token"]
    committed = run_cli(env, "commit", tid, "--token", token)
    expect(committed["committed"] is True, committed)

    step("resume")
    resumed = run_cli(env, "resume", "smoke")
    expect(resumed["status"] == "committed", resumed)

    ticket_file = root / "state" / "tickets" / "committed" / f"{tid}.json"
    expect(ticket_file.exists(), ticket_file)
    audit = run_cli(env, "verify-audit")
    expect(audit["ok"] is True, audit)
    step(f"audit chain verified ({audit['n_records']} records); committed ticket present")
    step("PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

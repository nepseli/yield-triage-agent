"""Runtime settings, read only from environment variables.

What: ``ServerSettings.from_env`` returns where the dataset, state (audit log,
tickets, nonce ledger) and policy live, which policy profile this run uses, and
the run ID.

Why: no configuration or secrets in code. Everything has a safe local default
except things that must be chosen deliberately (the model ID, read by the
agent in Phase 3, has no default). The approval key is deliberately *not* a
server setting.

Connects to: ``server/app.py`` and the CLI. ``.env.example`` lists the names.
"""

from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from yield_triage.policy.config import DEFAULT_POLICY_PATH

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@dataclass(frozen=True)
class ServerSettings:
    data_dir: Path
    state_dir: Path
    policy_path: Path
    profile: str
    run_id: str

    @property
    def audit_path(self) -> Path:
        return self.state_dir / "audit.jsonl"

    @property
    def tickets_dir(self) -> Path:
        return self.state_dir / "tickets"

    @property
    def ledger_path(self) -> Path:
        return self.state_dir / "used_nonces.txt"

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> ServerSettings:
        env = dict(os.environ) if env is None else env
        run_id = env.get("YIELD_TRIAGE_RUN_ID") or uuid.uuid4().hex
        if not _RUN_ID_RE.fullmatch(run_id):
            raise ValueError("YIELD_TRIAGE_RUN_ID must be 1-64 of [A-Za-z0-9_-]")
        return cls(
            data_dir=Path(env.get("YIELD_TRIAGE_DATA_DIR") or "data/synthetic"),
            state_dir=Path(env.get("YIELD_TRIAGE_STATE_DIR") or ".local/state"),
            policy_path=Path(env.get("YIELD_TRIAGE_POLICY") or DEFAULT_POLICY_PATH),
            profile=env.get("YIELD_TRIAGE_PROFILE") or "triage_run",
            run_id=run_id,
        )

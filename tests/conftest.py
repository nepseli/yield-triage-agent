"""Shared fixtures: a small SYNTHETIC dataset, a policy, and a gateway factory.

Every gateway gets a fresh temporary state directory (audit log, tickets,
nonce ledger), so tests never share state and never touch ``.local/``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import anyio
import pytest

from yield_triage.audit import AuditLog
from yield_triage.config import ServerSettings
from yield_triage.data import Dataset
from yield_triage.policy.config import DEFAULT_POLICY_PATH, Policy, load_policy
from yield_triage.policy.gateway import PolicyGateway
from yield_triage.server.app import build_gateway
from yield_triage.synthetic import SyntheticConfig, make_synthetic

ALL_TIME = {"start": "2030-01-01T00:00:00Z", "end": "2031-01-01T00:00:00Z"}

GatewayFactory = Callable[..., PolicyGateway]


@pytest.fixture(scope="session")
def small_dataset() -> Dataset:
    ds, _ = make_synthetic(3, SyntheticConfig(n_units=400, n_sensors=60))
    return ds


@pytest.fixture(scope="session")
def policy() -> Policy:
    return load_policy(DEFAULT_POLICY_PATH)


def with_limits(policy: Policy, **limits: Any) -> Policy:
    """A copy of ``policy`` with some limits overridden."""
    return policy.model_copy(update={"limits": policy.limits.model_copy(update=limits)})


@pytest.fixture
def make_gateway(tmp_path: Path, small_dataset: Dataset, policy: Policy) -> GatewayFactory:
    def factory(
        *, profile: str = "triage_run", policy_override: Policy | None = None
    ) -> PolicyGateway:
        settings = ServerSettings(
            data_dir=tmp_path / "unused",
            state_dir=tmp_path / "state",
            policy_path=DEFAULT_POLICY_PATH,
            profile=profile,
            run_id="test-run",
        )
        return build_gateway(settings, dataset=small_dataset, policy=policy_override or policy)

    return factory


def call(gateway: PolicyGateway, name: Any, args: Any) -> dict[str, Any]:
    """Synchronous wrapper so tests read top to bottom."""
    result: dict[str, Any] = anyio.run(gateway.call, name, args)
    return result


def last_audit(gateway: PolicyGateway) -> dict[str, Any]:
    return gateway.audit.records()[-1]


def audit_of(gateway: PolicyGateway) -> AuditLog:
    return gateway.audit

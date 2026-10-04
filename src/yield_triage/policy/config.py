"""Load and validate ``policy.yaml`` into typed, immutable settings.

What: pydantic models for the policy file (tools and their scopes, run
profiles, limits, analysis parameters, approval TTL) and ``load_policy``.

Why: the policy is the security boundary, so it is parsed strictly
(``extra="forbid"``) and cross-checked: a profile cannot allow a tool that
does not exist or that needs a scope the profile lacks. A typo fails loudly at
start-up instead of silently widening access.

Connects to: ``policy/gateway.py`` enforces it; ``server/app.py`` loads it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_POLICY_PATH = Path(__file__).with_name("policy.yaml")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolRule(_Strict):
    scope: str


class Profile(_Strict):
    scopes: tuple[str, ...]
    allowed_tools: tuple[str, ...]


class Limits(_Strict):
    max_tool_calls: int = Field(gt=0)
    max_rows: int = Field(gt=0)
    max_result_chars: int = Field(gt=200)
    run_timeout_s: float = Field(gt=0)
    max_arg_chars: int = Field(gt=0)


class AnalysisSettings(_Strict):
    fdr_q: float = Field(gt=0, lt=1)
    min_fail_units: int = Field(ge=1)
    drift_baseline_units: int = Field(ge=20)


class ApprovalSettings(_Strict):
    token_ttl_s: int = Field(gt=0, le=86400)


class Policy(_Strict):
    tools: dict[str, ToolRule]
    profiles: dict[str, Profile]
    limits: Limits
    analysis: AnalysisSettings
    approval: ApprovalSettings

    @model_validator(mode="after")
    def _profiles_are_consistent(self) -> Self:
        for name, profile in self.profiles.items():
            for tool in profile.allowed_tools:
                if tool not in self.tools:
                    raise ValueError(f"profile {name!r} allows unknown tool {tool!r}")
                if self.tools[tool].scope not in profile.scopes:
                    raise ValueError(f"profile {name!r} lacks scope for {tool!r}")
        return self


def load_policy(path: Path = DEFAULT_POLICY_PATH) -> Policy:
    """Parse YAML with ``safe_load`` (no object construction) and validate."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Policy.model_validate(raw)

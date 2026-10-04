"""Strict argument schemas for every MCP tool.

What: one pydantic model per tool. ``extra="forbid"`` rejects unknown keys and
``strict=True`` rejects type coercion (``"5"`` is not an int). Timestamps must
be full ISO-8601 date-times with an explicit offset. Sensor IDs must match the
``sNNN`` pattern *and* be in the allowlist taken from the loaded dataset
(passed in as validation context, never from the caller). Free text is
length-capped and rejected if it looks like a path or contains control
characters.

Why: the model chooses these arguments, and the model may be confused or
manipulated, so they are treated as hostile input (THREAT_MODEL.md T2).

Connects to: ``policy/gateway.py`` validates with ``TOOL_ARG_MODELS``; the
JSON schemas advertised over MCP are generated from the same models.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated, Self

import pandas as pd
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

# Full date-time with seconds optional and a mandatory Z or +HH:MM offset.
_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2}(\.\d{1,6})?)?(Z|[+-]\d{2}:\d{2})$")
SENSOR_RE = re.compile(r"^s\d{3}$")
CALL_ID_RE = re.compile(r"^c\d{4}$")
# Path-like fragments: traversal, absolute POSIX/Windows paths, home, UNC, file URLs.
# A lone " / " in prose is allowed; "/etc", "~/x", "C:\x", "..\x" are not.
_PATH_LIKE_RE = re.compile(
    r"(\.\.[/\\])|((^|\s)/[\w.-])|(~[/\\])|([A-Za-z]:[/\\])|(\\\\)|(file:)", re.I
)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _iso_timestamp(value: str) -> str:
    # SECURITY: T2 - fixed grammar first, then a real parse; no free-form dates.
    if not _ISO_RE.fullmatch(value):
        raise ValueError("timestamp must be ISO-8601 like 2030-01-01T00:00:00Z")
    datetime.fromisoformat(value.replace("Z", "+00:00"))  # rejects month 13 etc.
    return value


def _safe_text(value: str) -> str:
    # SECURITY: T2 - ticket text is stored and later shown to a human; refuse
    # control characters (terminal escape tricks) and path-like strings.
    if _CONTROL_RE.search(value):
        raise ValueError("control characters are not allowed")
    if _PATH_LIKE_RE.search(value):
        raise ValueError("path-like text is not allowed")
    return value


IsoTimestamp = Annotated[str, Field(max_length=40), AfterValidator(_iso_timestamp)]
SafeTitle = Annotated[str, Field(min_length=3, max_length=120), AfterValidator(_safe_text)]
SafeBody = Annotated[str, Field(min_length=10, max_length=4000), AfterValidator(_safe_text)]


def to_timestamp(value: str) -> pd.Timestamp:
    """Convert a validated ISO string to a UTC ``pd.Timestamp``."""
    return pd.Timestamp(value).tz_convert("UTC")


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class WindowArgs(_Args):
    start: IsoTimestamp
    end: IsoTimestamp

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if to_timestamp(self.end) <= to_timestamp(self.start):
            raise ValueError("end must be after start")
        return self


class RankArgs(WindowArgs):
    top_k: int = Field(ge=1, le=50)


class DriftArgs(WindowArgs):
    sensor_id: str = Field(max_length=8)

    @field_validator("sensor_id")
    @classmethod
    def _known_sensor(cls, value: str, info: ValidationInfo) -> str:
        # SECURITY: T2 - pattern check, then membership in the allowlist that
        # the server derived from the loaded dataset. No context = no allowlist
        # = reject (fail closed).
        if not SENSOR_RE.fullmatch(value):
            raise ValueError("sensor_id must look like s000")
        allowed = (info.context or {}).get("sensor_ids")
        if not allowed or value not in allowed:
            raise ValueError("unknown sensor_id")
        return value


class ProposeTicketArgs(_Args):
    title: SafeTitle
    body: SafeBody
    evidence_refs: list[Annotated[str, Field(pattern=CALL_ID_RE.pattern)]] = Field(
        min_length=1, max_length=20
    )


TOOL_ARG_MODELS: dict[str, type[_Args]] = {
    "get_window_summary": WindowArgs,
    "rank_failing_sensors": RankArgs,
    "check_sensor_drift": DriftArgs,
    "get_maintenance_notes": WindowArgs,
    "propose_ticket": ProposeTicketArgs,
}

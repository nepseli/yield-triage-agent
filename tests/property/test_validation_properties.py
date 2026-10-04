"""Property tests (Hypothesis) for argument validation, T2.

Properties: validation either accepts or raises ``ValidationError`` (never
crashes with anything else); accepted timestamps always match the strict
grammar; sensor IDs outside the allowlist are never accepted; path-like or
control-character text never reaches a ticket.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from yield_triage.policy.models import TOOL_ARG_MODELS, DriftArgs, ProposeTicketArgs, WindowArgs

ALLOWED = frozenset(f"s{i:03d}" for i in range(60))
ISO = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2}(\.\d{1,6})?)?(Z|[+-]\d{2}:\d{2})$")
CTX = {"sensor_ids": ALLOWED}

json_values = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False) | st.text(max_size=50),
    lambda inner: (
        st.lists(inner, max_size=4) | st.dictionaries(st.text(max_size=10), inner, max_size=4)
    ),
    max_leaves=10,
)


def _accepts(model: Any, data: Any) -> bool:
    try:
        model.model_validate(data, context=CTX)
    except ValidationError:
        return False
    return True


@settings(max_examples=300)
@given(
    st.sampled_from(sorted(TOOL_ARG_MODELS)),
    st.dictionaries(st.text(max_size=12), json_values, max_size=5),
)
def test_arbitrary_json_only_raises_validation_error(tool: str, data: dict[str, Any]) -> None:
    _accepts(TOOL_ARG_MODELS[tool], data)  # any other exception fails the test


@settings(max_examples=300)
@given(st.text(max_size=45), st.text(max_size=45))
def test_accepted_timestamps_match_strict_grammar(start: str, end: str) -> None:
    if _accepts(WindowArgs, {"start": start, "end": end}):
        assert ISO.fullmatch(start) and ISO.fullmatch(end)


@settings(max_examples=200)
@given(st.datetimes(timezones=st.just(UTC)))
def test_well_formed_utc_timestamps_are_accepted(dt: datetime) -> None:
    # Explicit zero padding: strftime("%Y") does not pad years < 1000 on Linux.
    start = (
        f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}T{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}Z"
    )
    end = "9999-12-31T23:59:59Z"
    if start < end:
        assert _accepts(WindowArgs, {"start": start, "end": end})


@settings(max_examples=300)
@given(st.text(max_size=12) | st.from_regex(r"s\d{3}", fullmatch=True))
def test_sensor_ids_outside_allowlist_never_accepted(sensor_id: str) -> None:
    args = {"start": "2030-01-01T00:00:00Z", "end": "2030-02-01T00:00:00Z", "sensor_id": sensor_id}
    assert _accepts(DriftArgs, args) == (sensor_id in ALLOWED)


@settings(max_examples=300)
@given(
    st.text(min_size=0, max_size=30),
    st.sampled_from(["../", "..\\", " /etc", "~/", "C:\\", "\\\\host", "file:", "\x1b[2J", "\x00"]),
    st.text(min_size=0, max_size=30),
)
def test_path_like_or_control_text_never_accepted(prefix: str, bad: str, suffix: str) -> None:
    text = f"{prefix}{bad}{suffix}"
    body = {"title": "valid title", "body": text.ljust(12, "x"), "evidence_refs": ["c0001"]}
    title = {"title": text[:120].ljust(3, "x"), "body": "a" * 20, "evidence_refs": ["c0001"]}
    assert not _accepts(ProposeTicketArgs, body)
    if bad in title["title"]:
        assert not _accepts(ProposeTicketArgs, title)


def test_no_allowlist_context_fails_closed() -> None:
    args = {"start": "2030-01-01T00:00:00Z", "end": "2030-02-01T00:00:00Z", "sensor_id": "s001"}
    try:
        DriftArgs.model_validate(args)
    except ValidationError:
        return
    raise AssertionError("sensor accepted without an allowlist")

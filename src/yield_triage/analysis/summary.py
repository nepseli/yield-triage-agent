"""Summarise a window of production units: counts, fail rate, missingness.

What: a single pure function that turns a (sensors, is_fail) slice into a
small, JSON-friendly summary.

Why: it is the agent's first look at a window ("how many units, how bad is the
fail rate, how much data is missing") and gives it numbers to cite instead of
computing them itself.

Connects to: the MCP tool ``get_window_summary``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd


@dataclass(frozen=True)
class WindowSummary:
    n_units: int
    n_fail: int
    n_pass: int
    fail_rate: float | None
    missing_fraction: float | None
    n_sensors: int
    n_sensors_all_missing: int


def summarize_window(
    sensors: pd.DataFrame,
    is_fail: npt.NDArray[np.bool_] | pd.Series,
) -> WindowSummary:
    """Counts and rates for one window. Rates are None for an empty window."""
    # STATS: missing_fraction is the share of all sensor cells that are
    # missing, a data-quality signal, not a per-unit quantity. Fail rate is a
    # plain proportion with no interval; at SECOM's size a window may hold only
    # a few failures, and callers should look at n_fail, not just the rate.
    fail = np.asarray(is_fail, dtype=bool)
    n_units = len(fail)
    n_fail = int(fail.sum())
    cells = sensors.size
    missing = int(sensors.isna().to_numpy().sum())
    return WindowSummary(
        n_units=n_units,
        n_fail=n_fail,
        n_pass=n_units - n_fail,
        fail_rate=(n_fail / n_units) if n_units else None,
        missing_fraction=(missing / cells) if cells else None,
        n_sensors=sensors.shape[1],
        n_sensors_all_missing=int(sensors.isna().all().sum()) if n_units else sensors.shape[1],
    )

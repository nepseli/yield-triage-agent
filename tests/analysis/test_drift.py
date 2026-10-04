"""Tests for EWMA and Western Electric drift rules.

Generated series are SYNTHETIC with fixed seeds. "Quiet on clean data" is
asserted as a low flagged-point rate across many seeds, because any control
chart has a non-zero false-alarm rate by design.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yield_triage.analysis import (
    Baseline,
    InsufficientDataError,
    estimate_baseline,
    ewma_violations,
    western_electric_violations,
)
from yield_triage.synthetic import make_synthetic

UNIT = Baseline(mean=0.0, std=1.0, n=100)


def _series(values: list[float]) -> pd.Series:
    return pd.Series(values, dtype=float)


def test_ewma_quiet_on_clean_data() -> None:
    flagged = total = 0
    for seed in range(20):
        rng = np.random.default_rng(seed)
        base = estimate_baseline(pd.Series(rng.normal(5, 2, 300)))
        monitored = pd.Series(rng.normal(5, 2, 300))
        flagged += len(ewma_violations(monitored, base))
        total += len(monitored)
    assert flagged / total < 0.01


def test_ewma_fires_quickly_after_injected_shift() -> None:
    rng = np.random.default_rng(3)
    base = estimate_baseline(pd.Series(rng.normal(0, 1, 300)))
    values = rng.normal(0, 1, 300)
    values[100:] += 1.5
    hits = [v.row for v in ewma_violations(pd.Series(values), base)]
    assert hits, "shift was not detected"
    after = [h for h in hits if h >= 100]
    assert min(after) - 100 <= 15
    assert len(after) / len(hits) >= 0.9


def test_ewma_skips_missing_values() -> None:
    values = [np.nan] * 5 + [10.0] * 3
    hits = ewma_violations(_series(values), UNIT)
    assert [v.row for v in hits] == [5, 6, 7]


def test_ewma_rejects_bad_lambda() -> None:
    with pytest.raises(ValueError, match="lam"):
        ewma_violations(_series([0.0]), UNIT, lam=0.0)


def test_planted_drift_detected_in_synthetic_data() -> None:
    ds, truth = make_synthetic(7)
    col = ds.sensors[truth.drift_sensor]
    n_base = int(0.4 * len(col))
    base = estimate_baseline(col.iloc[:n_base])
    hits = ewma_violations(col.iloc[n_base:], base)
    stamps = ds.timestamps
    in_drift = [v for v in hits if truth.drift_start <= stamps[v.row] < truth.drift_end]
    assert len(in_drift) >= 0.8 * len(hits)
    first = min(v.row for v in in_drift)
    drift_row = int((stamps < truth.drift_start).sum())  # first row inside the drift
    assert first - drift_row <= 20


def test_other_synthetic_sensors_mostly_quiet() -> None:
    ds, truth = make_synthetic(7)
    skip = {truth.drift_sensor, *truth.constant, *truth.sparse}
    names = [c for c in ds.sensors.columns if c not in skip][:100]
    n_base = int(0.4 * len(ds.sensors))
    flagged = total = 0
    for name in names:
        col = ds.sensors[name]
        flagged += len(ewma_violations(col.iloc[n_base:], estimate_baseline(col.iloc[:n_base])))
        total += int(col.iloc[n_base:].notna().sum())
    assert flagged / total < 0.01


@pytest.mark.parametrize(
    ("values", "rule", "index"),
    [
        ([0.0, 0.0, 3.5], "we1", 2),
        ([2.5, 0.0, 2.5], "we2", 2),
        ([1.5, 1.5, 0.0, 1.5, 1.5], "we3", 4),
        ([0.5] * 8, "we4", 7),
        ([-0.5] * 8, "we4", 7),
    ],
)
def test_western_electric_rules(values: list[float], rule: str, index: int) -> None:
    hits = western_electric_violations(_series(values), UNIT)
    assert [(v.row, v.rule) for v in hits] == [(index, rule)]


def test_western_electric_quiet_on_alternating_small_values() -> None:
    values = [0.5, -0.5] * 20
    assert western_electric_violations(_series(values), UNIT) == []


def test_baseline_too_short_rejected() -> None:
    with pytest.raises(InsufficientDataError, match="at least"):
        estimate_baseline(_series([1.0, 2.0, np.nan]))


def test_constant_baseline_rejected() -> None:
    with pytest.raises(InsufficientDataError, match="constant"):
        estimate_baseline(_series([4.0] * 50))

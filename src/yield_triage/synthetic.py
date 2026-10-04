"""Seeded generator for SYNTHETIC SECOM-shaped data with known ground truth.

What: produces a ``Dataset`` named ``"synthetic"`` (about 590 numeric sensors,
missing values, timestamps, a rare fail label) plus a ``GroundTruth`` record of
what was planted: five sensors whose distribution shifts for failing units,
the constant and sparse columns, and one sensor with a time-window drift.

Why: tests, CI and the evaluation need offline data with a known right answer.
Real SECOM has no ground truth, so recall and precision can only be measured
here. Everything this module produces is labelled synthetic.

Connects to: ``scripts/fetch_data.py --synthetic`` saves it to disk; tests and
``eval/run_eval.py`` call ``make_synthetic`` directly. ``GroundTruth`` is never
given to the MCP server, so tools cannot leak it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from yield_triage.data import Dataset, sensor_id

START = pd.Timestamp("2030-01-01T00:00:00Z")  # obviously not a real date range


@dataclass(frozen=True)
class SyntheticConfig:
    n_units: int = 1500
    n_sensors: int = 590
    fail_rate: float = 0.066  # close to SECOM's 104 / 1567
    n_planted: int = 5
    planted_shift_sd: tuple[float, ...] = (1.5, -1.2, 1.0, -0.9, 0.8)
    n_constant: int = 8
    n_sparse: int = 12
    drift_shift_sd: float = 1.5
    drift_fraction: tuple[float, float] = (0.6, 0.75)  # unit-index span of the drift


@dataclass(frozen=True)
class GroundTruth:
    planted: tuple[str, ...]
    constant: tuple[str, ...]
    sparse: tuple[str, ...]
    drift_sensor: str
    drift_start: pd.Timestamp
    drift_end: pd.Timestamp  # exclusive


def make_synthetic(seed: int, config: SyntheticConfig | None = None) -> tuple[Dataset, GroundTruth]:
    """Build a synthetic dataset and the record of what was planted in it."""
    cfg = config or SyntheticConfig()
    if cfg.n_planted > len(cfg.planted_shift_sd):
        raise ValueError("need one shift per planted sensor")
    if cfg.n_sensors <= cfg.n_planted + cfg.n_constant + cfg.n_sparse + 1:
        raise ValueError("n_sensors too small for the planted, constant, sparse and drift roles")
    rng = np.random.default_rng(seed)
    n, m = cfg.n_units, cfg.n_sensors
    stamps = _timestamps(rng, n)
    is_fail = rng.random(n) < cfg.fail_rate

    # Distinct roles for distinct columns, chosen at random from the seed.
    roles = rng.permutation(m)
    k = cfg.n_planted
    planted = roles[:k]
    constant = roles[k : k + cfg.n_constant]
    sparse = roles[k + cfg.n_constant : k + cfg.n_constant + cfg.n_sparse]
    drift = roles[k + cfg.n_constant + cfg.n_sparse]

    values = _base_values(rng, n, m)
    for col, shift in zip(planted, cfg.planted_shift_sd, strict=False):
        values[is_fail, col] += shift  # base columns have unit sd before scaling
    lo, hi = (int(f * n) for f in cfg.drift_fraction)
    values[lo:hi, drift] += cfg.drift_shift_sd
    values = _scale(rng, values)
    values[:, constant] = np.round(rng.normal(0, 10, size=len(constant)), 3)
    _add_missing(rng, values, sparse_cols=sparse)

    frame = pd.DataFrame(values, columns=[sensor_id(i) for i in range(m)])
    ds = Dataset(frame, pd.Series(stamps), pd.Series(is_fail), "synthetic")
    truth = GroundTruth(
        planted=tuple(sorted(sensor_id(int(c)) for c in planted)),
        constant=tuple(sorted(sensor_id(int(c)) for c in constant)),
        sparse=tuple(sorted(sensor_id(int(c)) for c in sparse)),
        drift_sensor=sensor_id(int(drift)),
        drift_start=stamps[lo],
        drift_end=stamps[hi] if hi < n else stamps[-1] + pd.Timedelta(seconds=1),
    )
    return ds, truth


def _timestamps(rng: np.random.Generator, n: int) -> pd.DatetimeIndex:
    """Irregular arrivals, 10 to 60 minutes apart, strictly increasing."""
    gaps = rng.integers(10, 61, size=n)
    return START + pd.to_timedelta(np.cumsum(gaps), unit="min")


def _base_values(rng: np.random.Generator, n: int, m: int) -> np.ndarray:
    """Unit-variance noise; one sensor in five is heavy-tailed (Student t, 3 df)."""
    values = rng.standard_normal((n, m))
    heavy = rng.random(m) < 0.2
    t3 = rng.standard_t(3, size=(n, int(heavy.sum()))) / np.sqrt(3.0)  # unit variance
    values[:, heavy] = t3
    return values


def _scale(rng: np.random.Generator, values: np.ndarray) -> np.ndarray:
    """Give each sensor its own location and scale, like real instruments."""
    m = values.shape[1]
    loc = rng.normal(0, 100, size=m)
    scale = 10 ** rng.uniform(-2, 3, size=m)
    return np.asarray(values * scale + loc)


def _add_missing(
    rng: np.random.Generator,
    values: np.ndarray,
    sparse_cols: np.ndarray,
) -> None:
    """Most sensors miss 0-5%, a few miss 30-60%, sparse columns keep only 2 values."""
    n, m = values.shape
    rate = np.asarray(rng.uniform(0.0, 0.05, size=m), dtype=float)
    lossy = np.asarray(rng.random(m) < 0.05, dtype=bool)
    rate[lossy] = rng.uniform(0.3, 0.6, size=int(lossy.sum()))
    mask = rng.random((n, m)) < rate
    values[mask] = np.nan
    for col in sparse_cols:
        keep = rng.choice(n, size=2, replace=False)
        column = np.full(n, np.nan)
        column[keep] = values[keep, col]
        values[:, col] = column

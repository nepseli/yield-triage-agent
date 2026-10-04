"""Detect drift in one sensor with an EWMA control chart against a baseline.

What: estimate in-control mean and standard deviation from a baseline window,
run an EWMA chart over a monitored window, and optionally apply the four
classic Western Electric rules to the individual readings. Return the points
that violate a rule.

Why: ranking (``ranking.py``) answers "which sensors differ on failures";
this answers "did this sensor move over time", which is how an excursion is
usually spotted. Pure function, no I/O, so it can be tested on planted shifts.

Connects to: the MCP tool ``check_sensor_drift`` picks the baseline and
monitored rows and attaches timestamps to the returned row labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

DEFAULT_LAMBDA = 0.2
DEFAULT_L = 3.0
MIN_BASELINE_POINTS = 20


class InsufficientDataError(ValueError):
    """Raised when the baseline cannot support control limits."""


@dataclass(frozen=True)
class Baseline:
    mean: float
    std: float
    n: int


@dataclass(frozen=True)
class Violation:
    """One flagged point.

    ``row`` is the integer row index of the unit (the caller maps it to a
    timestamp). ``statistic`` is what was charted (the EWMA value for "ewma",
    the standardised reading for "we*") and ``lower``/``upper`` are the limits
    in the same units as ``statistic``.
    """

    row: int
    rule: str
    value: float
    statistic: float
    lower: float
    upper: float


def estimate_baseline(baseline: pd.Series) -> Baseline:
    """Mean and sample std of the non-missing baseline readings."""
    # STATS: the baseline is assumed to be in control (no excursion). Using
    # the plain sample std of individual values, not a moving range, keeps the
    # maths obvious; the cost is that a baseline containing drift inflates the
    # limits and hides later shifts. The caller is responsible for choosing a
    # clean baseline window.
    vals = baseline.dropna().to_numpy(dtype=float)
    if vals.size < MIN_BASELINE_POINTS:
        raise InsufficientDataError(
            f"baseline has {vals.size} readings; need at least {MIN_BASELINE_POINTS}"
        )
    std = float(np.std(vals, ddof=1))
    if std == 0.0:
        raise InsufficientDataError("baseline is constant; control limits undefined")
    return Baseline(float(np.mean(vals)), std, int(vals.size))


def ewma_violations(
    monitored: pd.Series,  # integer row index, readings in time order
    base: Baseline,
    *,
    lam: float = DEFAULT_LAMBDA,
    width: float = DEFAULT_L,
) -> list[Violation]:
    """EWMA chart with exact (time-varying) control limits.

    z_t = lam * x_t + (1 - lam) * z_{t-1}, z_0 = baseline mean. Limits are
    mean +/- width * std * sqrt(lam / (2 - lam) * (1 - (1 - lam)^(2t))).
    """
    # STATS: EWMA is chosen over a plain 3-sigma chart because it accumulates
    # evidence and detects small sustained shifts (~0.5-1.5 sigma) much faster.
    # lam=0.2 with L=3 gives an in-control average run length of several hundred
    # points under independence. Units are assumed roughly independent in time;
    # autocorrelated sensors will alarm more often than this nominal rate.
    # Missing readings are skipped: they neither update z nor advance t.
    if not 0.0 < lam <= 1.0:
        raise ValueError("lam must be in (0, 1]")
    out: list[Violation] = []
    z, t = base.mean, 0
    rows = monitored.index.to_numpy(dtype=np.int64)
    for row, x in zip(rows, monitored.to_numpy(dtype=float), strict=True):
        if np.isnan(x):
            continue
        t += 1
        z = lam * float(x) + (1.0 - lam) * z
        half = width * base.std * np.sqrt(lam / (2.0 - lam) * (1.0 - (1.0 - lam) ** (2 * t)))
        lo, hi = base.mean - half, base.mean + half
        if z < lo or z > hi:
            out.append(Violation(int(row), "ewma", float(x), z, lo, hi))
    return out


def western_electric_violations(
    monitored: pd.Series,
    base: Baseline,
) -> list[Violation]:
    """The four Western Electric rules on individual readings (missing skipped).

    WE1: one point beyond 3 sigma. WE2: 2 of 3 beyond 2 sigma, same side.
    WE3: 4 of 5 beyond 1 sigma, same side. WE4: 8 in a row on one side.
    A point is reported for the first rule (in that order) it completes.
    """
    # STATS: running all four rules together raises the false-alarm rate
    # (roughly one alarm per ~90 in-control points instead of ~370 for WE1
    # alone), so they are optional and off by default in the tool.
    series = monitored.dropna()
    rows = series.index.to_numpy(dtype=np.int64)
    z = (series.to_numpy(dtype=float) - base.mean) / base.std
    out: list[Violation] = []
    for i, row in enumerate(rows):
        rule = _first_we_rule(z, i)
        if rule is not None:
            out.append(Violation(int(row), rule, float(series.iloc[i]), float(z[i]), -3.0, 3.0))
    return out


def _first_we_rule(z: npt.NDArray[np.float64], i: int) -> str | None:
    if abs(z[i]) > 3.0:
        return "we1"
    for rule, window, need, limit in (("we2", 3, 2, 2.0), ("we3", 5, 4, 1.0)):
        if i + 1 >= window:
            recent = z[i + 1 - window : i + 1]
            for sign in (1.0, -1.0):
                if sign * z[i] > limit and np.sum(sign * recent > limit) >= need:
                    return rule
    if i + 1 >= 8:
        recent = z[i - 7 : i + 1]
        if np.all(recent > 0) or np.all(recent < 0):
            return "we4"
    return None

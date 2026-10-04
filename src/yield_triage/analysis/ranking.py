"""Rank sensors by how differently they behave on failing versus passing units.

What: per-sensor Mann-Whitney U test, a rank-biserial effect size, and
Benjamini-Hochberg false-discovery-rate control across all tested sensors.

Why: the agent must never do statistics itself; this is the single place the
"which sensors look different on failures" answer comes from, so it is pure,
deterministic and unit-tested.

Connects to: the MCP tool ``rank_failing_sensors`` calls ``rank_failing_sensors``
here; ``eval/run_eval.py`` scores its output against synthetic ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

DEFAULT_Q = 0.05
DEFAULT_MIN_PER_GROUP = 5


@dataclass(frozen=True)
class SensorRank:
    """One tested sensor. ``effect_size`` > 0 means failing units read higher."""

    sensor: str
    effect_size: float
    p_value: float
    p_adj: float
    n_pass: int
    n_fail: int
    significant: bool


@dataclass(frozen=True)
class RankingResult:
    """All tested sensors sorted by adjusted p-value, plus bookkeeping."""

    ranked: list[SensorRank]
    q: float
    n_tested: int
    n_dropped_constant: int
    n_dropped_sparse: int

    @property
    def significant(self) -> list[SensorRank]:
        return [r for r in self.ranked if r.significant]


def benjamini_hochberg(p_values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return BH-adjusted p-values (step-up), capped at 1, in the input order."""
    # STATS: with ~590 sensors, testing each at 0.05 would flag ~30 sensors by
    # chance alone. BH bounds the expected share of false discoveries among the
    # flagged sensors at q, which is the question an engineer cares about
    # ("how many of these leads are noise?"). It assumes independent or
    # positively dependent tests; sensors are often positively correlated, which
    # is the favourable case.
    m = len(p_values)
    if m == 0:
        return p_values.copy()
    order = np.argsort(p_values)
    scaled = p_values[order] * m / np.arange(1, m + 1)
    # Enforce monotonicity from the largest p downwards, then cap at 1.
    adjusted_sorted = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted = np.empty(m)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def _test_sensor(
    column: npt.NDArray[np.float64], fail: npt.NDArray[np.bool_], min_per_group: int
) -> tuple[float, float, int, int] | None:
    """Return (effect, p, n_pass, n_fail) on non-missing values, None if too sparse."""
    present = ~np.isnan(column)
    fail_vals = column[present & fail]
    pass_vals = column[present & ~fail]
    if min(len(fail_vals), len(pass_vals)) < min_per_group:
        return None
    return _mann_whitney(fail_vals, pass_vals)


def _mann_whitney(
    fail_vals: npt.NDArray[np.float64], pass_vals: npt.NDArray[np.float64]
) -> tuple[float, float, int, int]:
    # STATS: Mann-Whitney U instead of a t-test because sensor readings are
    # often skewed, heavy-tailed or discrete, and a rank test makes no
    # normality assumption. Ties are handled by scipy's tie correction.
    # Effect size is the rank-biserial correlation r = 2U/(n1*n2) - 1, which
    # lies in [-1, 1] and equals P(fail > pass) - P(fail < pass).
    n_f, n_p = len(fail_vals), len(pass_vals)
    res = stats.mannwhitneyu(fail_vals, pass_vals, alternative="two-sided")
    effect = 2.0 * float(res.statistic) / (n_f * n_p) - 1.0
    return effect, float(res.pvalue), n_p, n_f


def rank_failing_sensors(
    sensors: pd.DataFrame,
    is_fail: npt.NDArray[np.bool_] | pd.Series,
    *,
    q: float = DEFAULT_Q,
    min_per_group: int = DEFAULT_MIN_PER_GROUP,
) -> RankingResult:
    """Test every usable sensor and control the FDR at ``q`` across all of them.

    Constant columns and columns with fewer than ``min_per_group`` non-missing
    values in either group are dropped *before* BH so they neither get a
    meaningless p-value nor inflate the number of tests.
    """
    if not 0.0 < q < 1.0:
        raise ValueError("q must be in (0, 1)")
    fail = np.asarray(is_fail, dtype=bool)
    if len(fail) != len(sensors):
        raise ValueError("is_fail length must match number of units")

    names: list[str] = []
    rows: list[tuple[float, float, int, int]] = []
    constant = sparse = 0
    for name in sensors.columns:
        col = sensors[name].to_numpy(dtype=float)
        present = col[~np.isnan(col)]
        # STATS: missing values are dropped per sensor (available-case
        # analysis), never imputed. Imputing hundreds of sparse sensors would
        # invent data; the cost is that different sensors use different units,
        # which is why n_pass / n_fail are returned per sensor.
        if present.size == 0 or np.ptp(present) == 0.0:
            constant += 1
            continue
        out = _test_sensor(col, fail, min_per_group)
        if out is None:
            sparse += 1
            continue
        names.append(str(name))
        rows.append(out)
    return _assemble(names, rows, q, constant, sparse)


def _assemble(
    names: list[str],
    rows: list[tuple[float, float, int, int]],
    q: float,
    constant: int,
    sparse: int,
) -> RankingResult:
    p = np.array([r[1] for r in rows], dtype=float)
    p_adj = benjamini_hochberg(p)
    ranked = [
        SensorRank(n, r[0], r[1], float(pa), r[2], r[3], bool(pa <= q))
        for n, r, pa in zip(names, rows, p_adj, strict=True)
    ]
    ranked.sort(key=lambda s: (s.p_adj, -abs(s.effect_size), s.sensor))
    return RankingResult(ranked, q, len(ranked), constant, sparse)

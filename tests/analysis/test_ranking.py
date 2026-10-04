"""Tests for sensor ranking: BH correctness, planted recovery, FDR on null data.

All data is SYNTHETIC (``make_synthetic``) with fixed seeds, so results are
deterministic and the planted sensors are the ground truth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from yield_triage.analysis import benjamini_hochberg, rank_failing_sensors
from yield_triage.synthetic import SyntheticConfig, make_synthetic

SMALL = SyntheticConfig(n_units=600, n_sensors=200)


def test_bh_matches_hand_worked_example() -> None:
    p = np.array([0.01, 0.04, 0.03, 0.005])
    # sorted .005 .01 .03 .04 -> *4/rank = .02 .02 .04 .04 (already monotone)
    np.testing.assert_allclose(benjamini_hochberg(p), [0.02, 0.04, 0.04, 0.02])


def test_bh_matches_scipy_reference() -> None:
    p = np.random.default_rng(0).random(500) ** 3
    np.testing.assert_allclose(benjamini_hochberg(p), stats.false_discovery_control(p))


def test_bh_empty_input() -> None:
    assert benjamini_hochberg(np.array([])).size == 0


def test_effect_sign_positive_when_failures_read_higher() -> None:
    rng = np.random.default_rng(1)
    fail = np.array([True] * 30 + [False] * 300)
    frame = pd.DataFrame({"s000": np.r_[rng.normal(2, 1, 30), rng.normal(0, 1, 300)]})
    (row,) = rank_failing_sensors(frame, fail).ranked
    assert row.effect_size > 0.5
    assert (row.n_fail, row.n_pass) == (30, 300)
    assert row.significant


def test_planted_sensors_recovered_default_seed() -> None:
    ds, truth = make_synthetic(7)
    result = rank_failing_sensors(ds.sensors, ds.is_fail)
    found = {r.sensor for r in result.significant}
    assert set(truth.planted) <= found
    # The planted sensors are the five strongest signals.
    assert {r.sensor for r in result.ranked[:5]} == set(truth.planted)


def test_planted_recall_across_seeds() -> None:
    recalls = []
    for seed in range(10):
        ds, truth = make_synthetic(seed, SMALL)
        found = {r.sensor for r in rank_failing_sensors(ds.sensors, ds.is_fail).significant}
        recalls.append(len(found & set(truth.planted)) / len(truth.planted))
    assert float(np.mean(recalls)) >= 0.9


def test_constant_and_sparse_columns_are_dropped_not_tested() -> None:
    ds, truth = make_synthetic(7)
    result = rank_failing_sensors(ds.sensors, ds.is_fail)
    tested = {r.sensor for r in result.ranked}
    assert tested.isdisjoint(truth.constant)
    assert tested.isdisjoint(truth.sparse)
    dropped = result.n_dropped_constant + result.n_dropped_sparse
    assert dropped == len(truth.constant) + len(truth.sparse)
    assert result.n_tested + dropped == ds.sensors.shape[1]


def test_all_missing_column_counts_as_constant() -> None:
    frame = pd.DataFrame({"s000": [np.nan] * 20, "s001": np.arange(20.0)})
    result = rank_failing_sensors(frame, np.arange(20) % 2 == 0)
    assert result.n_dropped_constant == 1
    assert [r.sensor for r in result.ranked] == ["s001"]


def test_fdr_on_global_null_stays_near_q() -> None:
    """Shuffled labels: every discovery is false, so FDR = P(any discovery) <= q."""
    q = 0.05
    ds, _ = make_synthetic(11, SMALL)
    rng = np.random.default_rng(123)
    fdp = []
    for _ in range(100):
        shuffled = rng.permutation(ds.is_fail.to_numpy())
        n_disc = len(rank_failing_sensors(ds.sensors, shuffled, q=q).significant)
        fdp.append(1.0 if n_disc > 0 else 0.0)
    # Nominal 0.05; 100 reps gives SE ~0.022, so 0.10 is a ~2 SE allowance.
    assert float(np.mean(fdp)) <= 0.10


def test_fdr_with_planted_signal_stays_near_q() -> None:
    q = 0.05
    fdp = []
    for seed in range(30):
        ds, truth = make_synthetic(seed, SMALL)
        found = {r.sensor for r in rank_failing_sensors(ds.sensors, ds.is_fail, q=q).significant}
        false = len(found - set(truth.planted))
        fdp.append(false / max(len(found), 1))
    assert float(np.mean(fdp)) <= 0.10


@pytest.mark.parametrize("q", [0.0, 1.0, -0.1])
def test_invalid_q_rejected(q: float) -> None:
    frame = pd.DataFrame({"s000": np.arange(20.0)})
    with pytest.raises(ValueError, match="q must be"):
        rank_failing_sensors(frame, np.arange(20) % 2 == 0, q=q)


def test_label_length_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="length"):
        rank_failing_sensors(pd.DataFrame({"s000": [1.0, 2.0]}), np.array([True]))

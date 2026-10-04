"""Tests for window summaries, SECOM parsing, save/load and the generator.

The SECOM-format text below is a tiny hand-made SYNTHETIC sample in the
verified file layout; it is not real data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from yield_triage.analysis import summarize_window
from yield_triage.data import Dataset, load_dataset, parse_secom, save_dataset
from yield_triage.synthetic import SyntheticConfig, make_synthetic

FAKE_DATA = "1.0 NaN 3.5\n2.0 5.0 3.5\n3.0 6.0 NaN\n"
# Third unit comes first in time to check that parsing sorts by timestamp.
FAKE_LABELS = '-1 "02/01/2030 10:00:00"\n1 "03/01/2030 09:30:00"\n-1 "01/01/2030 08:00:00"\n'


def test_summary_counts_and_rates() -> None:
    frame = pd.DataFrame({"s000": [1.0, np.nan, 3.0, 4.0], "s001": [np.nan] * 4})
    s = summarize_window(frame, np.array([True, False, False, False]))
    assert (s.n_units, s.n_fail, s.n_pass) == (4, 1, 3)
    assert s.fail_rate == pytest.approx(0.25)
    assert s.missing_fraction == pytest.approx(5 / 8)
    assert s.n_sensors_all_missing == 1


def test_summary_empty_window_has_no_rates() -> None:
    s = summarize_window(pd.DataFrame({"s000": pd.Series([], dtype=float)}), np.array([], bool))
    assert s.n_units == 0
    assert s.fail_rate is None
    assert s.missing_fraction is None


def test_parse_secom_layout_labels_and_sorting() -> None:
    ds = parse_secom(FAKE_DATA, FAKE_LABELS)
    assert ds.name == "secom"
    assert list(ds.sensors.columns) == ["s000", "s001", "s002"]
    assert ds.timestamps.iloc[0] == pd.Timestamp("2030-01-01T08:00:00Z")
    assert ds.sensors["s000"].tolist() == [3.0, 1.0, 2.0]
    assert ds.is_fail.tolist() == [False, False, True]
    assert np.isnan(ds.sensors.loc[1, "s001"])


def test_parse_secom_rejects_row_mismatch() -> None:
    with pytest.raises(ValueError, match="row counts"):
        parse_secom(FAKE_DATA, '-1 "01/01/2030 08:00:00"\n')


def test_parse_secom_rejects_unknown_label() -> None:
    bad = FAKE_LABELS.replace('1 "03', '0 "03')
    with pytest.raises(ValueError, match="label"):
        parse_secom(FAKE_DATA, bad)


def test_save_load_round_trip(tmp_path: Path) -> None:
    ds, _ = make_synthetic(1, SyntheticConfig(n_units=50, n_sensors=40))
    save_dataset(ds, tmp_path)
    back = load_dataset(tmp_path)
    assert back.name == "synthetic" and back.synthetic
    pd.testing.assert_frame_equal(back.sensors, ds.sensors, check_exact=False, rtol=1e-12)
    assert back.is_fail.tolist() == ds.is_fail.tolist()
    assert (back.timestamps == ds.timestamps.dt.floor("s")).all()


def test_window_is_half_open() -> None:
    ds = parse_secom(FAKE_DATA, FAKE_LABELS)
    w = ds.window(pd.Timestamp("2030-01-01T08:00:00Z"), pd.Timestamp("2030-01-02T10:00:00Z"))
    assert len(w.sensors) == 1
    with pytest.raises(ValueError, match="after start"):
        ds.window(pd.Timestamp("2030-01-02", tz="UTC"), pd.Timestamp("2030-01-01", tz="UTC"))


def test_dataset_rejects_unknown_name_and_unsorted_time() -> None:
    frame = pd.DataFrame({"s000": [1.0, 2.0]})
    fail = pd.Series([False, True])
    stamps = pd.Series(pd.to_datetime(["2030-01-02", "2030-01-01"], utc=True))
    with pytest.raises(ValueError, match="sorted"):
        Dataset(frame, stamps, fail, "synthetic")
    with pytest.raises(ValueError, match="name"):
        Dataset(frame, stamps.sort_values().reset_index(drop=True), fail, "real")


def test_generator_is_deterministic_and_labelled() -> None:
    a, ta = make_synthetic(5, SyntheticConfig(n_units=100, n_sensors=40))
    b, tb = make_synthetic(5, SyntheticConfig(n_units=100, n_sensors=40))
    pd.testing.assert_frame_equal(a.sensors, b.sensors)
    assert ta == tb
    assert a.name == "synthetic"
    assert a.sensor_ids == frozenset(f"s{i:03d}" for i in range(40))


def test_generator_shape_matches_secom_scale() -> None:
    ds, truth = make_synthetic(7)
    assert ds.sensors.shape == (1500, 590)
    assert 0.03 < ds.is_fail.mean() < 0.10
    assert ds.sensors.isna().to_numpy().mean() > 0.01
    assert len(truth.planted) == 5

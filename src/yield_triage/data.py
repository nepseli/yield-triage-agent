"""Dataset container, SECOM parsing and on-disk format.

What: a frozen ``Dataset`` (sensor matrix, timestamps, fail labels, a
``synthetic`` flag) plus helpers to parse the raw SECOM files and to save and
load our normalised CSV.

Why: every other module (analysis callers, the MCP server, the eval) needs one
agreed in-memory shape, and the server needs the sensor-ID allowlist derived
from the loaded data rather than from model input.

Connects to: ``scripts/fetch_data.py`` writes the CSV, ``synthetic.py`` builds
Datasets in memory, ``analysis/`` consumes the pieces.

A row is a production unit. Its only identifiers are the row index and the
timestamp. Sensor IDs are ``s000``..``sNNN`` by column position and carry no
meaning beyond that.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SENSOR_PREFIX = "s"
CSV_NAME = "units.csv"
META_NAME = "dataset.json"


def sensor_id(position: int) -> str:
    """Stable sensor ID from a zero-based column position."""
    return f"{SENSOR_PREFIX}{position:03d}"


@dataclass(frozen=True)
class Dataset:
    """Sensor readings for production units, in timestamp order.

    ``sensors`` has one row per unit and one float column per sensor ID.
    ``timestamps`` is UTC-aware and sorted ascending. ``is_fail`` is boolean.
    ``name`` is ``"secom"`` or ``"synthetic"`` and is echoed in tool results so
    synthetic data can never be mistaken for real data.
    """

    sensors: pd.DataFrame
    timestamps: pd.Series
    is_fail: pd.Series
    name: str

    def __post_init__(self) -> None:
        if self.name not in ("secom", "synthetic"):
            raise ValueError("dataset name must be 'secom' or 'synthetic'")
        n = len(self.sensors)
        if len(self.timestamps) != n or len(self.is_fail) != n:
            raise ValueError("sensors, timestamps and is_fail must have equal length")
        if not self.timestamps.is_monotonic_increasing:
            raise ValueError("timestamps must be sorted ascending")

    @property
    def synthetic(self) -> bool:
        return self.name == "synthetic"

    @property
    def sensor_ids(self) -> frozenset[str]:
        """Allowlist used to validate any sensor ID that arrives as an argument."""
        return frozenset(self.sensors.columns)

    def window_mask(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        """Rows with ``start <= timestamp < end`` (half-open, so windows tile)."""
        if end <= start:
            raise ValueError("end must be after start")
        return (self.timestamps >= start) & (self.timestamps < end)

    def window(self, start: pd.Timestamp, end: pd.Timestamp) -> Dataset:
        """Sub-dataset for a half-open time window; row index labels are kept."""
        mask = self.window_mask(start, end)
        return Dataset(self.sensors[mask], self.timestamps[mask], self.is_fail[mask], self.name)


def parse_secom(data_text: str, labels_text: str) -> Dataset:
    """Parse the raw UCI SECOM files into a ``Dataset``.

    Verified format: ``secom.data`` is space-separated floats with ``NaN`` for
    missing; ``secom_labels.data`` lines are ``<-1|1> "dd/mm/yyyy HH:MM:SS"``
    where -1 is pass and 1 is fail. The source gives no timezone, so
    timestamps are treated as UTC (documented in DECISIONS.md).
    """
    values = pd.read_csv(io.StringIO(data_text), sep=r"\s+", header=None, dtype=float)
    labels = pd.read_csv(
        io.StringIO(labels_text),
        sep=r"\s+",
        header=None,
        names=["label", "stamp"],  # the quoted "date time" is a single field
        quotechar='"',
    )
    if len(values) != len(labels):
        raise ValueError("secom.data and secom_labels.data have different row counts")
    if set(labels["label"].unique()) - {-1, 1}:
        raise ValueError("unexpected label values; expected only -1 and 1")
    stamps = pd.to_datetime(labels["stamp"], format="%d/%m/%Y %H:%M:%S", utc=True)
    values.columns = pd.Index([sensor_id(i) for i in range(values.shape[1])])
    return _sorted_dataset(values, stamps, labels["label"] == 1, "secom")


def _sorted_dataset(
    sensors: pd.DataFrame,
    stamps: pd.Series,
    is_fail: pd.Series,
    name: str,
) -> Dataset:
    """Stable sort by timestamp and reset the row index to 0..n-1 (the row ID)."""
    order = np.argsort(stamps.to_numpy(), kind="stable")
    return Dataset(
        sensors.iloc[order].reset_index(drop=True),
        stamps.iloc[order].reset_index(drop=True),
        is_fail.iloc[order].reset_index(drop=True).astype(bool),
        name,
    )


def save_dataset(ds: Dataset, directory: Path, extra_meta: dict[str, Any] | None = None) -> None:
    """Write ``units.csv`` and ``dataset.json`` into ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    frame = ds.sensors.copy()
    frame.insert(0, "is_fail", ds.is_fail.astype(int))
    frame.insert(0, "timestamp", ds.timestamps.dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
    frame.to_csv(directory / CSV_NAME, index_label="row_index")
    meta: dict[str, Any] = {
        "name": ds.name,
        "synthetic": ds.synthetic,
        "n_units": len(frame),
        "n_sensors": ds.sensors.shape[1],
    }
    meta.update(extra_meta or {})
    (directory / META_NAME).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def load_dataset(directory: Path) -> Dataset:
    """Load a dataset saved by ``save_dataset``."""
    meta = json.loads((directory / META_NAME).read_text())
    frame = pd.read_csv(directory / CSV_NAME, index_col="row_index")
    stamps = pd.to_datetime(frame.pop("timestamp"), utc=True)
    fail = frame.pop("is_fail").astype(bool)
    return _sorted_dataset(
        frame.reset_index(drop=True),
        stamps.reset_index(drop=True),
        fail.reset_index(drop=True),
        str(meta["name"]),
    )

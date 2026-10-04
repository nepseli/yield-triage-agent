"""Download the UCI SECOM dataset, or generate synthetic data, into ``data/``.

What: ``python scripts/fetch_data.py`` downloads ``secom.zip`` from the UCI
repository, parses it with ``yield_triage.data.parse_secom`` and writes
``data/secom/units.csv`` plus ``dataset.json`` (with the citation and the zip's
SHA-256). ``--synthetic`` instead writes ``data/synthetic/`` from the seeded
generator, labelled synthetic.

Why: the raw files are not committed (licence attribution, size, and
``data/`` is git-ignored); this script is the reproducible way to get them.

Connects to: the MCP server loads whichever directory ``YIELD_TRIAGE_DATA_DIR``
points at. Tests never call the network path.

Dataset: McCann, M. & Johnston, A. (2008). SECOM [Dataset]. UCI Machine
Learning Repository. https://doi.org/10.24432/C54305 (CC BY 4.0).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

from yield_triage.data import parse_secom, save_dataset
from yield_triage.synthetic import make_synthetic

# Verified on the UCI dataset page (dataset 179). Fixed constant, never taken
# from user input, so the script cannot be pointed at an arbitrary URL.
SECOM_URL = "https://archive.ics.uci.edu/static/public/179/secom.zip"
CITATION = (
    "McCann, M. & Johnston, A. (2008). SECOM [Dataset]. UCI Machine Learning "
    "Repository. https://doi.org/10.24432/C54305"
)
MAX_BYTES = 20 * 1024 * 1024  # real zip is ~2 MB; refuse anything absurd
DEFAULT_SEED = 7


def download(url: str = SECOM_URL) -> bytes:
    """Fetch the zip with a size cap."""
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 - fixed https URL
        body: bytes = resp.read(MAX_BYTES + 1)
    if len(body) > MAX_BYTES:
        raise RuntimeError("download larger than expected; refusing")
    return body


def extract(zip_bytes: bytes) -> tuple[str, str]:
    """Read the two expected members by exact name (no extraction to disk)."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        data = zf.read("secom.data").decode("ascii")
        labels = zf.read("secom_labels.data").decode("ascii")
    return data, labels


def fetch_secom(out_root: Path) -> Path:
    body = download()
    ds = parse_secom(*extract(body))
    out = out_root / "secom"
    save_dataset(
        ds,
        out,
        {
            "source_url": SECOM_URL,
            "citation": CITATION,
            "licence": "CC BY 4.0",
            "zip_sha256": hashlib.sha256(body).hexdigest(),
            "n_fail": int(ds.is_fail.sum()),
        },
    )
    return out


def write_synthetic(out_root: Path, seed: int) -> Path:
    ds, truth = make_synthetic(seed)
    out = out_root / "synthetic"
    save_dataset(ds, out, {"seed": seed, "note": "SYNTHETIC data, not real measurements"})
    # Ground truth goes in a separate file that only humans and the eval read,
    # so loading the dataset can never hand the answer to the server or agent.
    truth_doc = {
        "planted": list(truth.planted),
        "drift_sensor": truth.drift_sensor,
        "drift_start": truth.drift_start.isoformat(),
        "drift_end": truth.drift_end.isoformat(),
    }
    (out / "ground_truth.json").write_text(json.dumps(truth_doc, indent=2) + "\n")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--synthetic", action="store_true", help="generate synthetic data")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=Path("data"))
    args = parser.parse_args(argv)
    out = write_synthetic(args.out, args.seed) if args.synthetic else fetch_secom(args.out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Insert the generated evaluation table into README.md between markers.

What: replaces everything between ``<!-- EVAL:START -->`` and
``<!-- EVAL:END -->`` with ``eval/results/latest.md``.

Why: README numbers must come from ``make eval`` output, never typed by hand.
Run automatically by ``make eval``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START, END = "<!-- EVAL:START -->", "<!-- EVAL:END -->"


def main() -> int:
    readme = ROOT / "README.md"
    table = (ROOT / "eval" / "results" / "latest.md").read_text(encoding="utf-8")
    text = readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        print("README.md has no EVAL markers", file=sys.stderr)
        return 1
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    readme.write_text(f"{head}{START}\n{table}{END}{tail}", encoding="utf-8")
    print("README.md evaluation table updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Pure statistical functions: no I/O, no LLM, no global state.

``ranking`` finds sensors that differ on failing units (Mann-Whitney + BH),
``drift`` finds sensors that moved over time (EWMA, Western Electric), and
``summary`` describes a window. The MCP server is the only caller in the
running system; tests and the eval call them directly.
"""

from yield_triage.analysis.drift import (
    Baseline,
    InsufficientDataError,
    Violation,
    estimate_baseline,
    ewma_violations,
    western_electric_violations,
)
from yield_triage.analysis.ranking import (
    RankingResult,
    SensorRank,
    benjamini_hochberg,
    rank_failing_sensors,
)
from yield_triage.analysis.summary import WindowSummary, summarize_window

__all__ = [
    "Baseline",
    "InsufficientDataError",
    "RankingResult",
    "SensorRank",
    "Violation",
    "WindowSummary",
    "benjamini_hochberg",
    "estimate_baseline",
    "ewma_violations",
    "rank_failing_sensors",
    "summarize_window",
    "western_electric_violations",
]

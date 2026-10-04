"""Per-run budget: tool-call count and wall-clock deadline.

What: ``RunBudget`` counts every attempted tool call (allowed or denied) and
knows when the run's wall-clock budget ends. The clock is injectable so tests
can expire a run without sleeping.

Why: a looping or manipulated agent must not be able to call tools forever or
hold the server indefinitely (THREAT_MODEL.md T9, OWASP LLM10).

Connects to: ``policy/gateway.py`` asks ``try_spend`` before every call and
uses ``remaining_s`` as the timeout for the tool's work.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class RunBudget:
    max_calls: int
    timeout_s: float
    clock: Callable[[], float] = time.monotonic
    calls: int = 0
    started: float = field(default=-1.0)

    def __post_init__(self) -> None:
        if self.started < 0:
            self.started = self.clock()

    def remaining_s(self) -> float:
        return self.timeout_s - (self.clock() - self.started)

    def try_spend(self) -> str | None:
        """Count one call; return a denial reason or None if within budget."""
        # SECURITY: T9 - denied calls count too, so hammering a blocked tool
        # still exhausts the budget.
        self.calls += 1
        if self.calls > self.max_calls:
            return "budget_calls_exceeded"
        if self.remaining_s() <= 0:
            return "budget_time_exceeded"
        return None

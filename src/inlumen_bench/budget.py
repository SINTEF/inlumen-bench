"""Per-episode limits on cost, wall time and model requests."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


class BudgetExceeded(Exception):
    pass


@dataclass
class Limits:
    max_cost_usd: float | None = None
    max_seconds: float | None = None
    max_requests: int | None = None


@dataclass
class Budget:
    limits: Limits = field(default_factory=Limits)
    cost_usd: float = 0.0
    # True once any request finished without a provider-reported cost.
    cost_incomplete: bool = False
    requests: int = 0
    started: float = field(default_factory=time.monotonic)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.started

    def check(self) -> None:
        """Called before each request; an in-flight request may still overshoot."""
        lim = self.limits
        if lim.max_requests is not None and self.requests >= lim.max_requests:
            raise BudgetExceeded(f"request limit {lim.max_requests} reached")
        if lim.max_cost_usd is not None and self.cost_usd >= lim.max_cost_usd:
            raise BudgetExceeded(f"cost limit ${lim.max_cost_usd:.2f} reached")
        if lim.max_seconds is not None and self.elapsed_s >= lim.max_seconds:
            raise BudgetExceeded(f"time limit {lim.max_seconds:.0f}s reached")

    def charge(self, cost_usd: float | None) -> None:
        with self._lock:
            self.requests += 1
            if cost_usd is None:
                self.cost_incomplete = True
            else:
                self.cost_usd += cost_usd

    def snapshot(self) -> dict:
        lim = self.limits
        return {
            "cost_usd": round(self.cost_usd, 6),
            "cost_incomplete": self.cost_incomplete,
            "elapsed_s": round(self.elapsed_s, 1),
            "requests": self.requests,
            "max_cost_usd": lim.max_cost_usd,
            "max_seconds": lim.max_seconds,
            "max_requests": lim.max_requests,
        }

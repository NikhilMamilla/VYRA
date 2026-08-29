"""In-process runtime metrics for the deployed service.

Deliberately dependency-free: a single module-level :class:`Metrics` instance
accumulates counters and a bounded window of recent request latencies, which the
``GET /metrics`` endpoint renders as JSON. This is enough to answer "is it up,
how much traffic, how slow, how many errors" from a browser or a cron check
without standing up Prometheus. The shape is flat so it can be scraped or shipped
into a time-series store later.

Counters are process-local and reset on restart; under multiple workers each
worker reports its own slice.
"""

from __future__ import annotations

import threading
import time
from collections import deque


def _status_class(status_code: int) -> str:
    return f"{status_code // 100}xx"


class Metrics:
    """Thread-safe accumulator for HTTP-level service metrics."""

    _LATENCY_WINDOW = 2048

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started = time.monotonic()
        self.requests_total = 0
        self.requests_by_status: dict[str, int] = {}
        self.requests_in_flight = 0
        self._latencies_ms: deque[float] = deque(maxlen=self._LATENCY_WINDOW)

    def request_started(self) -> None:
        with self._lock:
            self.requests_in_flight += 1

    def request_finished(self, *, status_code: int, duration_ms: float) -> None:
        cls = _status_class(status_code)
        with self._lock:
            self.requests_in_flight = max(0, self.requests_in_flight - 1)
            self.requests_total += 1
            self.requests_by_status[cls] = self.requests_by_status.get(cls, 0) + 1
            self._latencies_ms.append(duration_ms)

    def _percentile(self, samples: list[float], pct: float) -> float:
        if not samples:
            return 0.0
        rank = max(0, min(len(samples) - 1, round(pct / 100 * (len(samples) - 1))))
        return round(samples[rank], 2)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            samples = sorted(self._latencies_ms)
            by_status = dict(self.requests_by_status)
            total = self.requests_total
            in_flight = self.requests_in_flight
            uptime = time.monotonic() - self._started

        errors = by_status.get("5xx", 0)
        return {
            "uptime_seconds": round(uptime, 2),
            "requests_total": total,
            "requests_in_flight": in_flight,
            "requests_by_status_class": by_status,
            "error_rate": round(errors / total, 4) if total else 0.0,
            "latency_ms": {
                "window": len(samples),
                "p50": self._percentile(samples, 50),
                "p95": self._percentile(samples, 95),
                "p99": self._percentile(samples, 99),
                "max": round(samples[-1], 2) if samples else 0.0,
            },
        }

    def reset(self) -> None:
        """Test helper: clear all accumulated state."""
        with self._lock:
            self._started = time.monotonic()
            self.requests_total = 0
            self.requests_by_status.clear()
            self.requests_in_flight = 0
            self._latencies_ms.clear()


METRICS = Metrics()

"""In-process metrics for the /metrics endpoint.

Deliberately small: counters plus a bounded latency window, no Prometheus
client and no external collector. Scope is one worker process, which is stated
on the endpoint so the numbers are not mistaken for a fleet-wide view.
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Any

_WINDOW = 512


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    index = min(round(fraction * (len(values) - 1)), len(values) - 1)
    return round(values[index], 2)


class Metrics:
    def __init__(self, window: int = _WINDOW) -> None:
        self._latencies: deque[float] = deque(maxlen=window)
        self._status: Counter[str] = Counter()
        self.requests = 0
        self.questions = 0
        self.cache_hits = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.embedding_tokens = 0
        self.cost_usd = 0.0

    def record_request(self, *, status_code: int, latency_ms: float) -> None:
        self.requests += 1
        self._status[str(status_code)] += 1
        self._latencies.append(latency_ms)

    def record_answers(
        self,
        *,
        questions: int,
        cache_hit: bool,
        input_tokens: int,
        output_tokens: int,
        embedding_tokens: int,
        cost_usd: float,
    ) -> None:
        self.questions += questions
        self.cache_hits += int(cache_hit)
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.embedding_tokens += embedding_tokens
        self.cost_usd = round(self.cost_usd + cost_usd, 6)

    def snapshot(self) -> dict[str, Any]:
        ordered = sorted(self._latencies)
        return {
            "scope": "single worker process",
            "requests_total": self.requests,
            "requests_by_status": dict(self._status),
            "questions_total": self.questions,
            "document_cache_hits": self.cache_hits,
            "latency_ms": {
                "p50": _percentile(ordered, 0.50),
                "p95": _percentile(ordered, 0.95),
                "max": round(max(ordered), 2) if ordered else 0.0,
                "window": len(ordered),
            },
            "tokens": {
                "input": self.input_tokens,
                "output": self.output_tokens,
                "embedding": self.embedding_tokens,
                "total": self.input_tokens + self.output_tokens + self.embedding_tokens,
            },
            "estimated_cost_usd": self.cost_usd,
        }

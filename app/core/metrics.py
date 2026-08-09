"""A dependency-free metrics registry that renders Prometheus text format.

`prometheus_client` would work too, but it installs a process-global registry
that is awkward to reset between tests and pulls in a dependency for ~150 lines
of behaviour. This module implements exactly the three instrument types the
platform needs (counter, gauge, histogram) with explicit reset support.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Final, TypeVar

#: Latency buckets in seconds, tuned for HTTP handlers and source fetches.
DEFAULT_BUCKETS: Final[tuple[float, ...]] = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
)

Labels = Mapping[str, str]
_LabelKey = tuple[tuple[str, str], ...]


def _key(labels: Labels | None) -> _LabelKey:
    return tuple(sorted((str(k), str(v)) for k, v in (labels or {}).items()))


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _render_labels(key: _LabelKey) -> str:
    if not key:
        return ""
    inner = ",".join(f'{name}="{_escape(value)}"' for name, value in key)
    return "{" + inner + "}"


@dataclass
class _Series:
    value: float = 0.0
    count: int = 0
    total: float = 0.0
    buckets: dict[float, int] = field(default_factory=dict)


class Metric:
    """Base class for the three instrument types."""

    kind = "untyped"

    def __init__(
        self, name: str, description: str, buckets: tuple[float, ...] | None = None
    ) -> None:
        self.name = name
        self.description = description
        self.buckets = buckets or DEFAULT_BUCKETS
        self._series: dict[_LabelKey, _Series] = {}
        self._lock = threading.Lock()

    def _get(self, labels: Labels | None) -> _Series:
        key = _key(labels)
        series = self._series.get(key)
        if series is None:
            series = _Series(buckets=dict.fromkeys(self.buckets, 0))
            self._series[key] = series
        return series

    def reset(self) -> None:
        with self._lock:
            self._series.clear()

    def snapshot(self) -> dict[_LabelKey, _Series]:
        with self._lock:
            return dict(self._series)


class Counter(Metric):
    """Monotonically increasing value."""

    kind = "counter"

    def inc(self, amount: float = 1.0, labels: Labels | None = None) -> None:
        if amount < 0:
            raise ValueError("counters cannot decrease")
        with self._lock:
            self._get(labels).value += amount

    def value(self, labels: Labels | None = None) -> float:
        with self._lock:
            return self._get(labels).value


class Gauge(Metric):
    """Value that can go up and down."""

    kind = "gauge"

    def set(self, value: float, labels: Labels | None = None) -> None:
        with self._lock:
            self._get(labels).value = float(value)

    def inc(self, amount: float = 1.0, labels: Labels | None = None) -> None:
        with self._lock:
            self._get(labels).value += amount

    def dec(self, amount: float = 1.0, labels: Labels | None = None) -> None:
        self.inc(-amount, labels)

    def value(self, labels: Labels | None = None) -> float:
        with self._lock:
            return self._get(labels).value


class Histogram(Metric):
    """Cumulative distribution over configured buckets."""

    kind = "histogram"

    def observe(self, value: float, labels: Labels | None = None) -> None:
        if math.isnan(value) or value < 0:
            return
        with self._lock:
            series = self._get(labels)
            series.count += 1
            series.total += value
            for bound in self.buckets:
                if value <= bound:
                    series.buckets[bound] += 1

    @contextmanager
    def time(self, labels: Labels | None = None) -> Iterator[None]:
        """Context manager that observes the wall-clock duration of a block."""
        started = time.perf_counter()
        try:
            yield
        finally:
            self.observe(time.perf_counter() - started, labels)

    def count(self, labels: Labels | None = None) -> int:
        with self._lock:
            return self._get(labels).count

    def average(self, labels: Labels | None = None) -> float:
        with self._lock:
            series = self._get(labels)
            return series.total / series.count if series.count else 0.0


MetricT = TypeVar("MetricT", bound=Metric)


class MetricsRegistry:
    """Holds every instrument and renders the ``/metrics`` payload."""

    def __init__(self) -> None:
        self._metrics: dict[str, Metric] = {}
        self._lock = threading.Lock()

    def counter(self, name: str, description: str = "") -> Counter:
        return self._register(Counter(name, description))

    def gauge(self, name: str, description: str = "") -> Gauge:
        return self._register(Gauge(name, description))

    def histogram(
        self, name: str, description: str = "", buckets: tuple[float, ...] | None = None
    ) -> Histogram:
        return self._register(Histogram(name, description, buckets))

    def _register(self, metric: MetricT) -> MetricT:
        with self._lock:
            existing = self._metrics.get(metric.name)
            if existing is not None:
                if type(existing) is not type(metric):
                    raise ValueError(f"metric '{metric.name}' already registered with another type")
                return existing
            self._metrics[metric.name] = metric
            return metric

    def reset(self) -> None:
        """Clear all samples (tests, and the ``metrics reset`` CLI command)."""
        with self._lock:
            metrics = list(self._metrics.values())
        for metric in metrics:
            metric.reset()

    def render(self) -> str:
        """Prometheus text exposition format (version 0.0.4)."""
        lines: list[str] = []
        with self._lock:
            metrics = sorted(self._metrics.values(), key=lambda m: m.name)
        for metric in metrics:
            if metric.description:
                lines.append(f"# HELP {metric.name} {metric.description}")
            lines.append(f"# TYPE {metric.name} {metric.kind}")
            for key, series in sorted(metric.snapshot().items()):
                labels = _render_labels(key)
                if isinstance(metric, Histogram):
                    # ``observe`` increments every bucket whose bound >= value,
                    # so the stored counts are already cumulative.
                    for bound in metric.buckets:
                        bucket_labels = _render_labels((*key, ("le", _fmt(bound))))
                        lines.append(
                            f"{metric.name}_bucket{bucket_labels} {series.buckets.get(bound, 0)}"
                        )
                    inf_labels = _render_labels((*key, ("le", "+Inf")))
                    lines.append(f"{metric.name}_bucket{inf_labels} {series.count}")
                    lines.append(f"{metric.name}_sum{labels} {series.total}")
                    lines.append(f"{metric.name}_count{labels} {series.count}")
                else:
                    lines.append(f"{metric.name}{labels} {series.value}")
        return "\n".join(lines) + "\n"

    def as_dict(self) -> dict[str, float]:
        """Flat mapping used by the dashboard and the ``stats`` CLI command."""
        out: dict[str, float] = {}
        with self._lock:
            metrics = list(self._metrics.values())
        for metric in metrics:
            for key, series in metric.snapshot().items():
                suffix = "".join(f".{value}" for _, value in key)
                if isinstance(metric, Histogram):
                    out[f"{metric.name}{suffix}.count"] = float(series.count)
                    out[f"{metric.name}{suffix}.avg"] = (
                        series.total / series.count if series.count else 0.0
                    )
                else:
                    out[f"{metric.name}{suffix}"] = series.value
        return out


def _fmt(value: float) -> str:
    return f"{value:g}"


registry = MetricsRegistry()

# --------------------------------------------------------------------------- #
# Platform instruments
# --------------------------------------------------------------------------- #
http_requests_total = registry.counter("http_requests_total", "HTTP requests handled")
http_request_duration_seconds = registry.histogram(
    "http_request_duration_seconds", "HTTP request latency in seconds"
)
http_errors_total = registry.counter("http_errors_total", "HTTP responses with status >= 400")

articles_fetched_total = registry.counter("articles_fetched_total", "Raw articles fetched")
articles_stored_total = registry.counter("articles_stored_total", "Articles persisted")
articles_rejected_total = registry.counter(
    "articles_rejected_total", "Articles rejected by validation"
)
duplicates_detected_total = registry.counter(
    "duplicates_detected_total", "Duplicate articles detected, by level"
)
source_errors_total = registry.counter("source_errors_total", "Source failures, by source and kind")
source_fetch_duration_seconds = registry.histogram(
    "source_fetch_duration_seconds", "Source fetch latency in seconds"
)
processing_duration_seconds = registry.histogram(
    "processing_duration_seconds", "Per-stage pipeline latency in seconds"
)
db_query_duration_seconds = registry.histogram(
    "db_query_duration_seconds", "Repository query latency in seconds"
)
cache_hits_total = registry.counter("cache_hits_total", "Cache hits")
cache_misses_total = registry.counter("cache_misses_total", "Cache misses")
queue_depth = registry.gauge("queue_depth", "Pending jobs in the background queue")
circuit_state = registry.gauge("circuit_breaker_state", "0=closed 1=half-open 2=open, by source")
auth_failures_total = registry.counter("auth_failures_total", "Failed authentication attempts")
rate_limited_total = registry.counter("rate_limited_total", "Requests rejected by the rate limiter")


__all__ = [
    "Counter",
    "Gauge",
    "Histogram",
    "MetricsRegistry",
    "articles_fetched_total",
    "articles_rejected_total",
    "articles_stored_total",
    "auth_failures_total",
    "cache_hits_total",
    "cache_misses_total",
    "circuit_state",
    "db_query_duration_seconds",
    "duplicates_detected_total",
    "http_errors_total",
    "http_request_duration_seconds",
    "http_requests_total",
    "processing_duration_seconds",
    "queue_depth",
    "rate_limited_total",
    "registry",
    "source_errors_total",
    "source_fetch_duration_seconds",
]

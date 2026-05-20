"""Abstract base classes for the pts.metrics framework."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod

import polars as pl
from pydantic import BaseModel

_ENVELOPE_FIELDS = frozenset({
    'name', 'metric_type', 'release', 'run', 'dataset', 'source', 'destination',
})


class UnifiedMetricRecord(BaseModel):
    """Flat, parquet-compatible record written as one JSONL line per metric."""

    name: str
    metric_type: str
    release: str
    run: str
    dataset: str
    source: str
    destination: str
    result: str  # JSON blob of metric-specific fields (no envelope keys)


class MetricResult(BaseModel, ABC):
    """Base class for all metric result types.

    Subclasses add metric-specific fields (e.g. value, bins, groups).
    Envelope fields (release, run, dataset, source, destination) are injected
    by MetricRunner after compute() returns.
    """

    name: str
    metric_type: str
    release: str
    run: str
    dataset: str = ''
    source: str = ''
    destination: str = ''

    def to_unified_record(self) -> UnifiedMetricRecord:
        """Produce a flat UnifiedMetricRecord; metric-specific fields go into result."""
        data = self.model_dump()
        payload = {k: v for k, v in data.items() if k not in _ENVELOPE_FIELDS}
        return UnifiedMetricRecord(
            name=self.name,
            metric_type=self.metric_type,
            release=self.release,
            run=self.run,
            dataset=self.dataset,
            source=self.source,
            destination=self.destination,
            result=json.dumps(payload),
        )


class Metric(BaseModel, ABC):
    """Base class for all metric definitions.

    Subclasses declare their parameters as fields and implement compute().
    compute() must not set release or run — MetricRunner stamps those.
    """

    name: str

    @property
    def required_columns(self) -> list[str] | None:
        """Columns needed by compute(). None = all columns; [] = row count only."""
        return None

    @abstractmethod
    def compute(self, df: pl.DataFrame) -> MetricResult:
        """Compute the metric from a Polars DataFrame."""

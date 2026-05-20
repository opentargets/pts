"""Abstract base classes for the pts.metrics framework."""

from __future__ import annotations

from abc import ABC, abstractmethod

import polars as pl
from pydantic import BaseModel


class MetricResult(BaseModel, ABC):
    """Base class for all metric result types.

    Subclasses add metric-specific fields (e.g. value, bins, groups).
    ``release`` and ``run`` are injected by MetricRunner after compute() returns.
    """

    name: str
    metric_type: str
    release: str
    run: str


class Metric(BaseModel, ABC):
    """Base class for all metric definitions.

    Subclasses declare their parameters as fields and implement compute().
    compute() must not set release or run — MetricRunner stamps those.
    """

    name: str

    @abstractmethod
    def compute(self, df: pl.DataFrame) -> MetricResult:
        """Compute the metric from a Polars DataFrame."""

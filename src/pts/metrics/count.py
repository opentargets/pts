"""Count and distinct-count metric types — stub for base tests; full impl in Issue 2."""

from __future__ import annotations

from typing import Literal

import polars as pl

from pts.metrics.base import Metric, MetricResult


class CountResult(MetricResult):
    """Result for CountMetric."""

    metric_type: Literal['count'] = 'count'
    value: int


class DistinctCountResult(MetricResult):
    """Result for DistinctCountMetric."""

    metric_type: Literal['distinct_count'] = 'distinct_count'
    columns: list[str]
    value: int


class CountMetric(Metric):
    """Counts rows, or non-null values of a column."""

    type: Literal['count'] = 'count'
    column: str | None = None

    @property
    def required_columns(self) -> list[str]:
        """Columns needed for the count operation."""
        return [self.column] if self.column else []

    def compute(self, df: pl.DataFrame) -> CountResult:
        """Compute count."""
        if self.column is None:
            value = df.height
        else:
            value = int(df[self.column].is_not_null().sum())
        return CountResult(name=self.name, release='', run='', value=value)


class DistinctCountMetric(Metric):
    """Counts distinct values across one or more columns."""

    type: Literal['distinct_count'] = 'distinct_count'
    columns: list[str]

    @property
    def required_columns(self) -> list[str]:
        """Columns needed to compute the distinct key set."""
        return list(self.columns)

    def compute(self, df: pl.DataFrame) -> DistinctCountResult:
        """Compute distinct count."""
        value = df.select(self.columns).drop_nulls().n_unique()
        return DistinctCountResult(name=self.name, release='', run='', columns=self.columns, value=value)

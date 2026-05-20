"""Count and distinct-count metric types."""

from __future__ import annotations

from typing import Literal

import polars as pl

from pts.metrics.base import Metric, MetricResult


class CountResult(MetricResult):
    """Result for :class:`CountMetric`."""

    metric_type: Literal['count'] = 'count'
    """Always ``'count'``."""
    value: int
    """Row count, or non-null value count when ``column`` is specified."""


class DistinctCountResult(MetricResult):
    """Result for :class:`DistinctCountMetric`."""

    metric_type: Literal['distinct_count'] = 'distinct_count'
    """Always ``'distinct_count'``."""
    columns: list[str]
    """Column names whose combined values were counted."""
    value: int
    """Number of distinct non-null value combinations."""


class CountMetric(Metric):
    """Counts rows, or non-null values of a column."""

    type: Literal['count'] = 'count'
    """Always ``'count'``; used by the metric loader as the config discriminator."""
    column: str | None = None
    """Column to count non-null values for. When ``None``, counts all rows."""

    @property
    def required_columns(self) -> list[str]:
        """Columns needed for the count operation."""
        return [self.column] if self.column else []

    def compute(self, df: pl.DataFrame) -> CountResult:
        """Compute count.

        >>> CountMetric(name='total').compute(pl.DataFrame({'id': ['A', 'B', 'C']})).value
        3
        >>> CountMetric(name='nn', column='score').compute(pl.DataFrame({'score': [0.5, None, 0.9]})).value
        2
        """
        if self.column is None:
            value = df.height
        else:
            value = int(df[self.column].is_not_null().sum())
        return CountResult(name=self.name, value=value)


class DistinctCountMetric(Metric):
    """Counts distinct non-null value combinations across one or more columns."""

    type: Literal['distinct_count'] = 'distinct_count'
    """Always ``'distinct_count'``; used by the metric loader as the config discriminator."""
    columns: list[str]
    """Column names whose combined values form the distinct key."""

    @property
    def required_columns(self) -> list[str]:
        """Columns needed to compute the distinct key set."""
        return list(self.columns)

    def compute(self, df: pl.DataFrame) -> DistinctCountResult:
        """Compute distinct count.

        >>> DistinctCountMetric(name='d', columns=['s']).compute(pl.DataFrame({'s': ['A', 'B', 'A', None]})).value
        2
        """
        value = df.select(self.columns).drop_nulls().n_unique()
        return DistinctCountResult(name=self.name, columns=self.columns, value=value)

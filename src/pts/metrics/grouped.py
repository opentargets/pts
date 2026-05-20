"""Grouped count and grouped sum metric types."""

from __future__ import annotations

from typing import Any, Literal

import polars as pl
from pydantic import BaseModel

from pts.metrics.base import Metric, MetricResult


class GroupRow(BaseModel):
    """A single group key and its aggregated value."""

    key: dict[str, Any]
    count: int


class GroupedCountResult(MetricResult):
    """Result for GroupedCountMetric."""

    metric_type: Literal['grouped_count'] = 'grouped_count'
    group_by: list[str]
    groups: list[GroupRow]


class GroupedCountMetric(Metric):
    """Counts rows per group, sorted descending by count. Null keys excluded."""

    type: Literal['grouped_count'] = 'grouped_count'
    group_by: list[str]

    @property
    def required_columns(self) -> list[str]:
        return list(self.group_by)

    def compute(self, df: pl.DataFrame) -> GroupedCountResult:
        """Compute grouped row counts."""
        # Drop rows where ANY group_by column is null
        filtered = df.drop_nulls(subset=self.group_by)

        if filtered.is_empty():
            groups = []
        else:
            agg = (
                filtered.group_by(self.group_by)
                .agg(pl.len().alias('count'))
                .sort('count', descending=True)
            )
            groups = [
                GroupRow(key={col: row[col] for col in self.group_by}, count=row['count'])
                for row in agg.iter_rows(named=True)
            ]

        return GroupedCountResult(
            name=self.name,
            release='',
            run='',
            group_by=self.group_by,
            groups=groups,
        )


class GroupedSumResult(MetricResult):
    """Result for GroupedSumMetric."""

    metric_type: Literal['grouped_sum'] = 'grouped_sum'
    column: str
    group_by: list[str]
    groups: list[GroupRow]


class GroupedSumMetric(Metric):
    """Sums a numeric column per group, sorted descending by sum. Null keys excluded."""

    type: Literal['grouped_sum'] = 'grouped_sum'
    column: str
    group_by: list[str]

    @property
    def required_columns(self) -> list[str]:
        return [self.column, *self.group_by]

    def compute(self, df: pl.DataFrame) -> GroupedSumResult:
        """Compute grouped sums."""
        # Drop rows where ANY group_by column is null
        filtered = df.drop_nulls(subset=self.group_by)

        if filtered.is_empty():
            groups = []
        else:
            agg = (
                filtered.group_by(self.group_by)
                .agg(pl.sum(self.column).alias('count'))
                .sort('count', descending=True)
            )
            groups = [
                GroupRow(key={col: row[col] for col in self.group_by}, count=row['count'])
                for row in agg.iter_rows(named=True)
            ]

        return GroupedSumResult(
            name=self.name,
            release='',
            run='',
            column=self.column,
            group_by=self.group_by,
            groups=groups,
        )

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
        """Columns needed to construct the grouping key."""
        return list(self.group_by)

    def compute(self, df: pl.DataFrame) -> GroupedCountResult:
        """Compute grouped row counts.

        >>> df = pl.DataFrame({'studyType': ['gwas', 'eqtl', 'gwas', 'gwas']})
        >>> result = GroupedCountMetric(name='g', group_by=['studyType']).compute(df)
        >>> result.groups[0].key, result.groups[0].count
        ({'studyType': 'gwas'}, 3)
        >>> df2 = pl.DataFrame({'studyType': ['gwas', None]})
        >>> GroupedCountMetric(name='g', group_by=['studyType']).compute(df2).groups[0].key
        {'studyType': 'gwas'}
        """
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
            group_by=self.group_by,
            groups=groups,
        )


class GroupedCountExplodeResult(MetricResult):
    """Result for GroupedCountExplodeMetric."""

    metric_type: Literal['grouped_count_explode'] = 'grouped_count_explode'
    group_by: list[str]
    groups: list[GroupRow]


class GroupedCountExplodeMetric(Metric):
    """Explodes list columns in group_by, then counts rows per group, sorted descending."""

    type: Literal['grouped_count_explode'] = 'grouped_count_explode'
    group_by: list[str]

    @property
    def required_columns(self) -> list[str]:
        """Columns needed to explode and group."""
        return list(self.group_by)

    def compute(self, df: pl.DataFrame) -> GroupedCountExplodeResult:
        """Explode list columns then compute grouped row counts.

        >>> df = pl.DataFrame({'ta': [['TA1', 'TA2'], ['TA1', 'TA3'], ['TA1']]})
        >>> result = GroupedCountExplodeMetric(name='g', group_by=['ta']).compute(df)
        >>> result.groups[0].key, result.groups[0].count
        ({'ta': 'TA1'}, 3)
        >>> empty = pl.DataFrame({'ta': pl.Series([], dtype=pl.List(pl.String))})
        >>> GroupedCountExplodeMetric(name='g', group_by=['ta']).compute(empty).groups
        []
        """
        filtered = df.drop_nulls(subset=self.group_by)

        if filtered.is_empty():
            return GroupedCountExplodeResult(name=self.name, group_by=self.group_by, groups=[])

        exploded = filtered
        for col in self.group_by:
            exploded = exploded.explode(col)
        exploded = exploded.drop_nulls(subset=self.group_by)

        if exploded.is_empty():
            groups: list[GroupRow] = []
        else:
            agg = (
                exploded.group_by(self.group_by)
                .agg(pl.len().alias('count'))
                .sort('count', descending=True)
            )
            groups = [
                GroupRow(key={col: row[col] for col in self.group_by}, count=row['count'])
                for row in agg.iter_rows(named=True)
            ]

        return GroupedCountExplodeResult(
            name=self.name,
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
        """Columns needed for the grouped summation."""
        return [self.column, *self.group_by]

    def compute(self, df: pl.DataFrame) -> GroupedSumResult:
        """Compute grouped sums.

        >>> df = pl.DataFrame({'agg': ['a', 'a', 'b'], 'n': [100, 200, 50]})
        >>> result = GroupedSumMetric(name='s', column='n', group_by=['agg']).compute(df)
        >>> result.groups[0].count
        300
        """
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
            column=self.column,
            group_by=self.group_by,
            groups=groups,
        )

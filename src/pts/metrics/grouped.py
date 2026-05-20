"""Grouped count and grouped sum metric types."""

from __future__ import annotations

from typing import Any, Literal

import polars as pl
from pydantic import BaseModel

from pts.metrics.base import Metric, MetricResult


class GroupRow(BaseModel):
    """A single group key and its aggregated value."""

    key: dict[str, Any]
    """Mapping of group-by column names to their values for this group."""
    count: int
    """Aggregated count (or sum) for this group."""


class GroupedCountMetric(Metric):
    """Counts rows per group, sorted descending by count. Null keys form their own group."""

    type: Literal['grouped_count'] = 'grouped_count'
    """Always ``'grouped_count'``; used by the metric loader as the config discriminator."""
    group_by: list[str]
    """Column names to group by."""

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
        >>> df2 = pl.DataFrame({'studyType': ['gwas', 'gwas', None]})
        >>> GroupedCountMetric(name='g', group_by=['studyType']).compute(df2).groups[1].key
        {'studyType': None}
        """
        agg = (
            df.group_by(self.group_by)
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


class GroupedCountResult(MetricResult):
    """Result for :class:`GroupedCountMetric`."""

    metric_type: Literal['grouped_count'] = 'grouped_count'
    """Always ``'grouped_count'``."""
    group_by: list[str]
    """Column names used to form groups."""
    groups: list[GroupRow]
    """One entry per distinct key combination, sorted descending by count; null keys form their own group."""


class GroupedCountExplodeMetric(Metric):
    """Explodes list columns then counts rows per group, sorted descending.

    Each column in ``group_by`` is expected to hold list values. All columns are
    exploded before grouping, so a row contributing to multiple groups (e.g. a
    disease mapped to several therapeutic areas) is counted once per group.
    Null values after exploding form their own group.
    """

    type: Literal['grouped_count_explode'] = 'grouped_count_explode'
    """Always ``'grouped_count_explode'``; used by the metric loader as the config discriminator."""
    group_by: list[str]
    """List-typed column names to explode and then group by."""

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
        """
        exploded = df
        for col in self.group_by:
            exploded = exploded.explode(col)

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


class GroupedCountExplodeResult(MetricResult):
    """Result for :class:`GroupedCountExplodeMetric`."""

    metric_type: Literal['grouped_count_explode'] = 'grouped_count_explode'
    """Always ``'grouped_count_explode'``."""
    group_by: list[str]
    """Column names that were exploded then grouped."""
    groups: list[GroupRow]
    """One entry per distinct exploded value, sorted descending by count; null values form their own group."""


class GroupedSumMetric(Metric):
    """Sums a numeric column per group, sorted descending by sum. Null keys form their own group."""

    type: Literal['grouped_sum'] = 'grouped_sum'
    """Always ``'grouped_sum'``; used by the metric loader as the config discriminator."""
    column: str
    """Numeric column whose values are summed within each group."""
    group_by: list[str]
    """Column names to group by."""

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
        agg = (
            df.group_by(self.group_by)
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


class GroupedSumResult(MetricResult):
    """Result for :class:`GroupedSumMetric`."""

    metric_type: Literal['grouped_sum'] = 'grouped_sum'
    """Always ``'grouped_sum'``."""
    column: str
    """Name of the column that was summed."""
    group_by: list[str]
    """Column names used to form groups."""
    groups: list[GroupRow]
    """One entry per distinct key combination, sorted descending by sum; null keys form their own group."""

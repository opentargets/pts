"""Grouped count and grouped sum metric types."""

from __future__ import annotations

from functools import reduce
from typing import Any, Literal

import polars as pl
from pydantic import BaseModel

from pts.metrics.base import Metric, MetricResult


class GroupRow(BaseModel):
    """A single group key and its aggregated value."""

    key: dict[str, Any]
    """Mapping of group-by column names to their values for this group."""
    value: int
    """Aggregated value (count or sum) for this group."""


class GroupedCountMetric(Metric):
    """Counts rows per group, sorted descending by count. Null keys form their own group.

    ``group_by`` entries are passed directly as SQL expressions, so both plain
    column names and expressions (e.g. ``"upper(studyType)"``) are accepted.
    """

    type: Literal['grouped_count'] = 'grouped_count'
    """Always ``'grouped_count'``; used by the metric loader as the config discriminator."""
    group_by: list[str]
    """SQL expressions to group by (plain column names or SQL function calls)."""

    def compute(self, df: pl.DataFrame) -> GroupedCountResult:
        """Compute grouped row counts.

        >>> df = pl.DataFrame({'studyType': ['gwas', 'eqtl', 'gwas', 'gwas']})
        >>> result = GroupedCountMetric(name='g', group_by=['studyType']).compute(df)
        >>> result.groups[0].key, result.groups[0].value
        ({'studyType': 'gwas'}, 3)
        >>> df2 = pl.DataFrame({'studyType': ['gwas', 'gwas', None]})
        >>> GroupedCountMetric(name='g', group_by=['studyType']).compute(df2).groups[1].key
        {'studyType': None}
        >>> df3 = pl.DataFrame({'type': ['gwas', 'gwas', 'gwas', 'eqtl'], 'pop': ['EUR', 'EUR', 'AFR', 'EUR']})
        >>> result3 = GroupedCountMetric(name='g', group_by=['type', 'pop']).compute(df3)
        >>> result3.groups[0].key, result3.groups[0].value
        ({'type': 'gwas', 'pop': 'EUR'}, 2)
        >>> df4 = pl.DataFrame({'studyType': ['gwas', 'eqtl', 'gwas']})
        >>> r4 = GroupedCountMetric(name='g', group_by=["upper(studyType)"]).compute(df4)
        >>> r4.groups[0].key
        {'studyType': 'GWAS'}
        """
        exprs = [pl.sql_expr(e) for e in self.group_by]
        agg = df.group_by(exprs).agg(pl.len().alias('count')).sort('count', descending=True)
        names = [c for c in agg.columns if c != 'count']
        groups = [
            GroupRow(key={c: row[c] for c in names}, value=row['count'])
            for row in agg.iter_rows(named=True)
        ]
        return GroupedCountResult(name=self.name, group_by=names, groups=groups)


class GroupedCountResult(MetricResult):
    """Result for :class:`GroupedCountMetric`."""

    metric_type: Literal['grouped_count'] = 'grouped_count'
    """Always ``'grouped_count'``."""
    group_by: list[str]
    """Resolved column names used to form groups."""
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
    """SQL expressions (typically plain list-typed column names) to explode and group by."""

    def compute(self, df: pl.DataFrame) -> GroupedCountExplodeResult:
        """Explode list columns then compute grouped row counts.

        >>> df = pl.DataFrame({'ta': [['TA1', 'TA2'], ['TA1', 'TA3'], ['TA1']]})
        >>> result = GroupedCountExplodeMetric(name='g', group_by=['ta']).compute(df)
        >>> result.groups[0].key, result.groups[0].value
        ({'ta': 'TA1'}, 3)
        >>> df2 = pl.DataFrame({'ta': [['TA1', 'TA2'], ['TA1']], 'ds': [['D1', 'D2'], ['D1']]})
        >>> result2 = GroupedCountExplodeMetric(name='g', group_by=['ta', 'ds']).compute(df2)
        >>> result2.groups[0].key, result2.groups[0].value
        ({'ta': 'TA1', 'ds': 'D1'}, 2)
        """
        exprs = [pl.sql_expr(e) for e in self.group_by]
        names = df.select(exprs).columns
        df = df.with_columns(exprs)
        exploded = reduce(lambda acc, col: acc.explode(col), names, df)
        agg = exploded.group_by(names).agg(pl.len().alias('count')).sort('count', descending=True)
        groups = [
            GroupRow(key={c: row[c] for c in names}, value=row['count'])
            for row in agg.iter_rows(named=True)
        ]
        return GroupedCountExplodeResult(name=self.name, group_by=names, groups=groups)


class GroupedCountExplodeResult(MetricResult):
    """Result for :class:`GroupedCountExplodeMetric`."""

    metric_type: Literal['grouped_count_explode'] = 'grouped_count_explode'
    """Always ``'grouped_count_explode'``."""
    group_by: list[str]
    """Resolved column names that were exploded then grouped."""
    groups: list[GroupRow]
    """One entry per distinct exploded value, sorted descending by count; null values form their own group."""


class GroupedSumMetric(Metric):
    """Sums a numeric column per group, sorted descending by sum. Null keys form their own group."""

    type: Literal['grouped_sum'] = 'grouped_sum'
    """Always ``'grouped_sum'``; used by the metric loader as the config discriminator."""
    column: str
    """Numeric column whose values are summed within each group."""
    group_by: list[str]
    """SQL expressions to group by (plain column names or SQL function calls)."""

    def compute(self, df: pl.DataFrame) -> GroupedSumResult:
        """Compute grouped sums.

        >>> df = pl.DataFrame({'agg': ['a', 'a', 'b'], 'n': [100, 200, 50]})
        >>> result = GroupedSumMetric(name='s', column='n', group_by=['agg']).compute(df)
        >>> result.groups[0].value
        300
        """
        exprs = [pl.sql_expr(e) for e in self.group_by]
        agg = df.group_by(exprs).agg(pl.sum(self.column).alias('count')).sort('count', descending=True)
        names = [c for c in agg.columns if c != 'count']
        groups = [
            GroupRow(key={c: row[c] for c in names}, value=row['count'])
            for row in agg.iter_rows(named=True)
        ]
        return GroupedSumResult(name=self.name, column=self.column, group_by=names, groups=groups)


class GroupedSumResult(MetricResult):
    """Result for :class:`GroupedSumMetric`."""

    metric_type: Literal['grouped_sum'] = 'grouped_sum'
    """Always ``'grouped_sum'``."""
    column: str
    """Name of the column that was summed."""
    group_by: list[str]
    """Resolved column names used to form groups."""
    groups: list[GroupRow]
    """One entry per distinct key combination, sorted descending by sum; null keys form their own group."""

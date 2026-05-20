"""Distribution metric type for pts.metrics."""

from __future__ import annotations

from typing import Literal

import polars as pl
from pydantic import BaseModel

from pts.metrics.base import Metric, MetricResult


class Bin(BaseModel):
    """A single histogram bin."""

    bin_start: float
    bin_end: float
    count: int


class DistributionResult(MetricResult):
    """Result for DistributionMetric."""

    metric_type: Literal['distribution'] = 'distribution'
    column: str
    bins: list[Bin]


class DistributionMetric(Metric):
    """Computes an equal-width histogram over a numeric column."""

    column: str
    n_bins: int = 20

    @property
    def required_columns(self) -> list[str]:
        return [self.column]

    def compute(self, df: pl.DataFrame) -> DistributionResult:
        """Compute equal-width histogram, dropping nulls."""
        series = df[self.column].drop_nulls()

        if len(series) == 0:
            return DistributionResult(
                name=self.name,
                release='',
                run='',
                column=self.column,
                bins=[],
            )

        col_min = series.min()
        col_max = series.max()

        # When all values are identical, place everything in a single bin
        # and replicate it n_bins times with equal edges.
        if col_min == col_max:
            single_count = len(series)
            bins = [
                Bin(bin_start=float(col_min), bin_end=float(col_max), count=single_count if i == 0 else 0)
                for i in range(self.n_bins)
            ]
            return DistributionResult(
                name=self.name,
                release='',
                run='',
                column=self.column,
                bins=bins,
            )

        bin_width = (col_max - col_min) / self.n_bins

        # Assign each value to a 0-indexed bin using vectorised Polars expressions,
        # clamping the max value into the last bin so it is never out-of-range.
        bin_indices = (
            ((series.cast(pl.Float64) - col_min) / bin_width)
            .floor()
            .cast(pl.Int64)
            .clip(0, self.n_bins - 1)
        )

        counts_series = (
            pl.DataFrame({'bin_idx': bin_indices})
            .group_by('bin_idx')
            .agg(pl.len().alias('count'))
            .sort('bin_idx')
        )

        # Build a dict for fast lookup
        counts_map: dict[int, int] = dict(
            zip(counts_series['bin_idx'].to_list(), counts_series['count'].to_list())
        )

        bins = [
            Bin(
                bin_start=float(col_min + i * bin_width),
                bin_end=float(col_min + (i + 1) * bin_width),
                count=counts_map.get(i, 0),
            )
            for i in range(self.n_bins)
        ]

        return DistributionResult(
            name=self.name,
            release='',
            run='',
            column=self.column,
            bins=bins,
        )

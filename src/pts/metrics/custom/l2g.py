"""Custom metric: L2G significant gene count."""
from __future__ import annotations

import polars as pl

from pts.metrics.base import Metric
from pts.metrics.count import CountResult


class L2GSignificantGeneMetric(Metric):
    """Counts distinct genes with L2G score >= threshold."""

    threshold: float = 0.5

    @property
    def required_columns(self) -> list[str]:
        return ['geneId', 'score']

    def compute(self, df: pl.DataFrame) -> CountResult:
        """Count distinct geneId values where score >= threshold."""
        filtered = df.filter(pl.col("score") >= self.threshold)
        value = filtered.select("geneId").n_unique()
        return CountResult(name=self.name, release='', run='', value=value)

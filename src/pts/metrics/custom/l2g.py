"""Custom metric: L2G significant gene count."""
from __future__ import annotations

from typing import Literal

import polars as pl

from pts.metrics.base import Metric
from pts.metrics.count import CountResult


class L2GSignificantGeneMetric(Metric):
    """Counts distinct genes with an L2G score at or above a threshold."""

    type: Literal['l2g_significant_gene'] = 'l2g_significant_gene'
    """Always ``'l2g_significant_gene'``; used by the metric loader as the config discriminator."""
    threshold: float = 0.5
    """Minimum L2G score a gene must reach to be counted. Defaults to 0.5."""

    @property
    def required_columns(self) -> list[str]:
        """Columns needed to evaluate the thresholded distinct gene count."""
        return ['geneId', 'score']

    def compute(self, df: pl.DataFrame) -> CountResult:
        """Count distinct geneId values where score >= threshold.

        >>> df = pl.DataFrame({'geneId': ['G1', 'G1', 'G2', 'G3'], 'score': [0.8, 0.6, 0.3, 0.9]})
        >>> L2GSignificantGeneMetric(name='sig').compute(df).value
        2
        >>> df2 = pl.DataFrame({'geneId': ['G1', None], 'score': [0.6, 0.8]})
        >>> L2GSignificantGeneMetric(name='sig').compute(df2).value
        1
        """
        filtered = df.filter((pl.col('score') >= self.threshold) & pl.col('geneId').is_not_null())
        value = filtered.select('geneId').n_unique()
        return CountResult(name=self.name, value=value)

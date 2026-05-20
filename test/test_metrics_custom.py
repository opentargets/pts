"""Tests for custom metric types."""
import polars as pl
import pytest

from pts.metrics.custom import L2GSignificantGeneMetric


@pytest.mark.parametrize(('data', 'expected'), [
    ({'geneId': ['G1', 'G1', 'G2', 'G3'], 'score': [0.8, 0.6, 0.3, 0.9]}, 2),  # G1 max 0.8, G3 0.9
    ({'geneId': ['G1'], 'score': [0.49]}, 0),
    ({'geneId': ['G1', None], 'score': [0.6, 0.8]}, 1),
])
def test_l2g_significant_gene_count(data, expected):
    result = L2GSignificantGeneMetric(name='sig', threshold=0.5).compute(pl.DataFrame(data))
    assert result.value == expected


def test_l2g_default_threshold():
    assert L2GSignificantGeneMetric(name='sig').threshold == pytest.approx(0.5)

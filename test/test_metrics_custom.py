"""Tests for custom metric types."""
import polars as pl
import pytest

from pts.metrics.count import CountResult
from pts.metrics.custom import L2GSignificantGeneMetric

# ── L2GSignificantGeneMetric ──────────────────────────────────────────────────

def test_l2g_counts_genes_above_threshold():
    df = pl.DataFrame({
        'geneId': ['G1', 'G1', 'G2', 'G3'],
        'score': [0.8, 0.6, 0.3, 0.9],
    })
    result = L2GSignificantGeneMetric(name='sig', threshold=0.5).compute(df)
    assert isinstance(result, CountResult)
    assert result.value == 2  # G1 (max 0.8 >= 0.5) and G3 (0.9 >= 0.5)


def test_l2g_threshold_exclusive_below():
    df = pl.DataFrame({'geneId': ['G1'], 'score': [0.49]})
    result = L2GSignificantGeneMetric(name='sig', threshold=0.5).compute(df)
    assert result.value == 0


def test_l2g_excludes_null_gene_ids():
    df = pl.DataFrame({'geneId': ['G1', None], 'score': [0.6, 0.8]})
    result = L2GSignificantGeneMetric(name='sig', threshold=0.5).compute(df)
    assert result.value == 1


def test_l2g_default_threshold_is_0_5():
    m = L2GSignificantGeneMetric(name='sig')
    assert m.threshold == pytest.approx(0.5)

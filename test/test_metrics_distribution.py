"""Tests for DistributionMetric — written before implementation (TDD)."""
import polars as pl
import pytest
from pts.metrics.distribution import DistributionMetric, DistributionResult


def test_distribution_produces_n_bins():
    df = pl.DataFrame({'score': [0.1, 0.3, 0.5, 0.7, 0.9]})
    result = DistributionMetric(name='d', column='score', n_bins=5).compute(df)
    assert isinstance(result, DistributionResult)
    assert len(result.bins) == 5


def test_distribution_counts_sum_to_non_null_rows():
    df = pl.DataFrame({'score': [0.1, None, 0.5, 0.9]})
    result = DistributionMetric(name='d', column='score', n_bins=3).compute(df)
    assert sum(b.count for b in result.bins) == 3  # null excluded


def test_distribution_bins_cover_full_range():
    df = pl.DataFrame({'score': [0.0, 1.0]})
    result = DistributionMetric(name='d', column='score', n_bins=2).compute(df)
    assert result.bins[0].bin_start == pytest.approx(0.0)
    assert result.bins[-1].bin_end == pytest.approx(1.0)


def test_distribution_empty_dataframe_returns_empty_bins():
    df = pl.DataFrame({'score': pl.Series([], dtype=pl.Float64)})
    result = DistributionMetric(name='d', column='score', n_bins=10).compute(df)
    assert result.bins == []


def test_distribution_all_null_returns_empty_bins():
    df = pl.DataFrame({'score': [None, None]})
    result = DistributionMetric(name='d', column='score', n_bins=5).compute(df)
    assert result.bins == []



"""Tests for pts.metrics base module."""
import polars as pl
import pytest

from pts.metrics.base import EmptyDatasetError, Metric
from pts.metrics.count import CountMetric, DistinctCountMetric


@pytest.mark.parametrize('cls', [
    Metric,
    type('IncompleteMetric', (Metric,), {}),
])
def test_metric_requires_compute_implementation(cls):
    with pytest.raises(TypeError):
        cls(name='x')


def test_run_raises_on_empty_dataframe():
    with pytest.raises(EmptyDatasetError):
        CountMetric(name='x').run(pl.DataFrame({'a': pl.Series([], dtype=pl.Int64)}))


def test_filter_expr_pre_filters_before_compute():
    df = pl.DataFrame({'score': [0.8, 0.3, 0.9], 'x': [1, 2, 3]})
    result = CountMetric(name='n', filter_expr='score >= 0.5').run(df)
    assert result.value == 2


def test_filter_expr_on_distinct_count():
    df = pl.DataFrame({'geneId': ['G1', 'G1', 'G2', 'G3'], 'score': [0.8, 0.6, 0.3, 0.9]})
    result = DistinctCountMetric(name='sig', columns=['geneId'], filter_expr='score >= 0.5').run(df)
    assert result.value == 2


def test_filter_expr_propagates_to_unified_record():
    df = pl.DataFrame({'score': [0.8, 0.3], 'x': [1, 2]})
    result = CountMetric(name='n', filter_expr='score >= 0.5').run(df)
    record = result.to_unified_record(
        release='r', run='1', dataset='ds', source='/s', destination='/d',
        filter_expr='score >= 0.5',
    )
    assert record.filter_expr == 'score >= 0.5'


def test_no_filter_expr_record_has_none():
    df = pl.DataFrame({'x': [1, 2]})
    result = CountMetric(name='n').run(df)
    record = result.to_unified_record(
        release='r', run='1', dataset='ds', source='/s', destination='/d',
    )
    assert record.filter_expr is None

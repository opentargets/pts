"""Tests for CountMetric and DistinctCountMetric — written before implementation (TDD)."""
import polars as pl

from pts.metrics import CountMetric, DistinctCountMetric
from pts.metrics.count import CountResult, DistinctCountResult


def test_count_metric_total_rows():
    df = pl.DataFrame({'id': ['A', 'B', 'C']})
    result = CountMetric(name='total').compute(df)
    assert isinstance(result, CountResult)
    assert result.value == 3
    assert result.metric_type == 'count'


def test_count_metric_non_null_column():
    df = pl.DataFrame({'score': [0.5, None, 0.9]})
    result = CountMetric(name='nn', column='score').compute(df)
    assert result.value == 2


def test_count_metric_empty_dataframe():
    df = pl.DataFrame({'id': pl.Series([], dtype=pl.String)})
    result = CountMetric(name='total').compute(df)
    assert result.value == 0


def test_distinct_count_single_column():
    df = pl.DataFrame({'studyId': ['S1', 'S2', 'S1', 'S3']})
    result = DistinctCountMetric(name='d', columns=['studyId']).compute(df)
    assert isinstance(result, DistinctCountResult)
    assert result.value == 3
    assert result.columns == ['studyId']


def test_distinct_count_multi_column():
    df = pl.DataFrame({'diseaseId': ['D1', 'D1', 'D2'], 'targetId': ['T1', 'T2', 'T1']})
    result = DistinctCountMetric(name='pairs', columns=['diseaseId', 'targetId']).compute(df)
    assert result.value == 3  # all three rows are distinct pairs


def test_distinct_count_excludes_nulls():
    df = pl.DataFrame({'studyId': ['S1', None, 'S1']})
    result = DistinctCountMetric(name='d', columns=['studyId']).compute(df)
    assert result.value == 1

"""Tests for CountMetric and DistinctCountMetric."""
import polars as pl
import pytest

from pts.metrics import CountMetric, DistinctCountMetric


@pytest.mark.parametrize(('df', 'expected'), [
    (pl.DataFrame({'id': ['A', 'B', 'C']}), 3),
    (pl.DataFrame({'id': pl.Series([], dtype=pl.String)}), 0),
])
def test_count_metric_row_count(df, expected):
    assert CountMetric(name='total').compute(df).value == expected


def test_count_metric_non_null_column():
    df = pl.DataFrame({'score': [0.5, None, 0.9]})
    assert CountMetric(name='nn', column='score').compute(df).value == 2


def test_distinct_count_single_column():
    df = pl.DataFrame({'studyId': ['S1', 'S2', 'S1', 'S3']})
    result = DistinctCountMetric(name='d', columns=['studyId']).compute(df)
    assert result.value == 3
    assert result.columns == ['studyId']


def test_distinct_count_multi_column():
    df = pl.DataFrame({'diseaseId': ['D1', 'D1', 'D2'], 'targetId': ['T1', 'T2', 'T1']})
    assert DistinctCountMetric(name='pairs', columns=['diseaseId', 'targetId']).compute(df).value == 3


def test_distinct_count_excludes_nulls():
    df = pl.DataFrame({'studyId': ['S1', None, 'S1']})
    assert DistinctCountMetric(name='d', columns=['studyId']).compute(df).value == 1

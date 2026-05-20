"""Tests for pts.metrics base module."""
import polars as pl
import pytest

from pts.metrics.base import EmptyDatasetError, Metric
from pts.metrics.count import CountMetric


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

"""Tests for pts.metrics base module — written before implementation (TDD)."""
import pytest


def test_metric_is_abstract():
    from pts.metrics.base import Metric
    with pytest.raises(TypeError):
        Metric(name='x')


def test_metric_subclasses_must_implement_compute():
    import polars as pl
    from pts.metrics.base import Metric

    class IncompleteMetric(Metric):
        pass

    with pytest.raises(TypeError):
        IncompleteMetric(name='x')


def test_metric_result_has_release_run_fields():
    from pts.metrics.count import CountResult
    r = CountResult(name='n', release='26.06-pub', run='testrun.1', value=1)
    assert r.release == '26.06-pub'
    assert r.run == 'testrun.1'



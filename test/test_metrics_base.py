"""Tests for pts.metrics base module."""
import pytest

from pts.metrics.base import Metric


@pytest.mark.parametrize('cls', [
    Metric,
    type('IncompleteMetric', (Metric,), {}),
])
def test_metric_requires_compute_implementation(cls):
    with pytest.raises(TypeError):
        cls(name='x')

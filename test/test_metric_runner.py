"""Tests for MetricRunner (Issue 7) — TDD, written before implementation."""
import json
import polars as pl
import pytest
from pathlib import Path

from pts.metrics import CountMetric, GroupedCountMetric
from pts.metrics.runner import MetricRunner


@pytest.fixture
def dataset(tmp_path):
    df = pl.DataFrame({'studyType': ['gwas', 'eqtl', 'gwas'], 'score': [0.8, 0.4, 0.9]})
    path = tmp_path / 'dataset'
    path.mkdir()
    df.write_parquet(path / 'part.parquet')
    return path


def test_runner_writes_json_files(dataset, tmp_path):
    metrics_root = tmp_path / 'metrics'
    MetricRunner().run(
        metrics=[CountMetric(name='total_count')],
        dataset_path=dataset,
        metrics_root=metrics_root,
        dataset_name='study',
        release='26.06-pub',
        run='testrun.1',
    )
    out = metrics_root / 'study' / 'total_count.json'
    assert out.exists()


def test_runner_stamps_release_and_run(dataset, tmp_path):
    MetricRunner().run(
        metrics=[CountMetric(name='total_count')],
        dataset_path=dataset,
        metrics_root=tmp_path / 'metrics',
        dataset_name='study',
        release='26.06-pub',
        run='testrun.1',
    )
    data = json.loads((tmp_path / 'metrics' / 'study' / 'total_count.json').read_text())
    assert data['release'] == '26.06-pub'
    assert data['run'] == 'testrun.1'


def test_runner_json_content_is_valid(dataset, tmp_path):
    MetricRunner().run(
        metrics=[CountMetric(name='total_count')],
        dataset_path=dataset,
        metrics_root=tmp_path / 'metrics',
        dataset_name='study',
        release='26.06-pub',
        run='testrun.1',
    )
    data = json.loads((tmp_path / 'metrics' / 'study' / 'total_count.json').read_text())
    assert data['value'] == 3
    assert data['metric_type'] == 'count'


def test_runner_bad_column_raises(dataset, tmp_path):
    with pytest.raises(Exception, match='nonexistent'):
        MetricRunner().run(
            metrics=[GroupedCountMetric(name='g', group_by=['nonexistent'])],
            dataset_path=dataset,
            metrics_root=tmp_path / 'metrics',
            dataset_name='study',
            release='',
            run='',
        )

"""Tests for MetricRunner."""
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


def test_runner_writes_jsonl_file(dataset, tmp_path):
    out = tmp_path / 'metrics' / 'study.jsonl'
    MetricRunner().run(
        metrics=[CountMetric(name='total_count')],
        dataset_path=dataset,
        out_file=out,
        release='26.06-pub',
        run='testrun.1',
    )
    assert out.exists()


def test_runner_stamps_release_and_run(dataset, tmp_path):
    out = tmp_path / 'metrics' / 'study.jsonl'
    MetricRunner().run(
        metrics=[CountMetric(name='total_count')],
        dataset_path=dataset,
        out_file=out,
        release='26.06-pub',
        run='testrun.1',
    )
    record = json.loads(out.read_text().splitlines()[0])
    assert record['release'] == '26.06-pub'
    assert record['run'] == 'testrun.1'


def test_runner_json_content_is_valid(dataset, tmp_path):
    out = tmp_path / 'metrics' / 'study.jsonl'
    MetricRunner().run(
        metrics=[CountMetric(name='total_count')],
        dataset_path=dataset,
        out_file=out,
        release='26.06-pub',
        run='testrun.1',
    )
    record = json.loads(out.read_text().splitlines()[0])
    assert json.loads(record['result'])['value'] == 3
    assert record['metric_type'] == 'count'


def test_runner_stamps_dataset_source_destination(dataset, tmp_path):
    out = tmp_path / 'metrics' / 'dataset.jsonl'
    MetricRunner().run(
        metrics=[CountMetric(name='total_count')],
        dataset_path=dataset,
        out_file=out,
        source=str(dataset),
        destination=str(out),
        release='26.06-pub',
        run='testrun.1',
    )
    record = json.loads(out.read_text().splitlines()[0])
    assert record['dataset'] == dataset.name
    assert record['source'] == str(dataset)
    assert record['destination'] == str(out)


def test_runner_bad_column_raises(dataset, tmp_path):
    with pytest.raises(Exception, match='nonexistent'):
        MetricRunner().run(
            metrics=[GroupedCountMetric(name='g', group_by=['nonexistent'])],
            dataset_path=dataset,
            out_file=tmp_path / 'metrics' / 'study.jsonl',
            release='',
            run='',
        )

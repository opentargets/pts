"""Tests for MetricRunner."""
import json

import polars as pl
import pytest

from pts.metrics import CountMetric, GroupedCountMetric
from pts.metrics.runner import MetricRunner


@pytest.fixture
def dataset(tmp_path):
    df = pl.DataFrame({'studyType': ['gwas', 'eqtl', 'gwas'], 'score': [0.8, 0.4, 0.9]})
    path = tmp_path / 'dataset'
    path.mkdir()
    df.write_parquet(path / 'part.parquet')
    return path


def test_runner_writes_jsonl_record_with_expected_envelope(dataset, tmp_path):
    out = tmp_path / 'metrics' / 'study.jsonl'
    MetricRunner().run(
        metrics=[CountMetric(name='total_count')],
        dataset_path=dataset,
        out_file=out,
        source=str(dataset),
        destination=str(out),
        release='26.06-pub',
        run='testrun.1',
    )
    assert out.exists()
    record = json.loads(out.read_text().splitlines()[0])

    assert record == {
        'name': 'total_count',
        'metric_type': 'count',
        'release': '26.06-pub',
        'run': 'testrun.1',
        'dataset': dataset.name,
        'source': str(dataset),
        'destination': str(out),
        'result': json.dumps({'value': 3}),
    }


def test_runner_bad_column_raises(dataset, tmp_path):
    out = tmp_path / 'metrics' / 'study.jsonl'
    with pytest.raises(Exception, match='nonexistent'):
        MetricRunner().run(
            metrics=[GroupedCountMetric(name='g', group_by=['nonexistent'])],
            dataset_path=dataset,
            out_file=out,
            source=str(dataset),
            destination=str(out),
            release='26.06-pub',
            run='testrun.1',
        )

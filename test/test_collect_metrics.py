"""Tests for CollectMetrics task."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from threading import Event
from typing import Any, cast
from unittest.mock import MagicMock

import polars as pl
import pytest
import yaml
from otter.scratchpad.model import Scratchpad
from otter.task.model import TaskContext

from pts.tasks.collect_metrics import CollectMetrics, CollectMetricsSpec


def _make_context(work_path=None, release='26.06-pub', run='testrun.1'):
    config = MagicMock()
    config.release_uri = None
    if work_path is not None:
        config.work_path = work_path
    sentinel = {k: v for k, v in {'release': release, 'run': run}.items() if v is not None}
    scratchpad = Scratchpad(sentinel_dict=sentinel)
    context = TaskContext(config=config, scratchpad=scratchpad)
    context.abort = Event()
    return context


# ── Validation tests ──────────────────────────────────────────────────────────

@pytest.mark.parametrize('missing_key', ['release', 'run'])
def test_collect_metrics_raises_if_scratchpad_key_missing(missing_key):
    spec = CollectMetricsSpec(
        name='collect_metrics test',
        source='/tmp/data',
        destination='/tmp/metrics/data.parquet',
        metrics=cast(Any, [{'type': 'count', 'name': 'total_count'}]),
    )
    with pytest.raises(ValueError, match=missing_key):
        CollectMetrics(spec=spec, context=_make_context(**{missing_key: None}))


def test_spec_rejects_unknown_metric_type():
    with pytest.raises(ValueError, match='bad_type'):
        CollectMetricsSpec(
            name='collect_metrics test',
            source='/tmp/data',
            destination='/tmp/metrics/data.parquet',
            metrics=cast(Any, [{'type': 'bad_type', 'name': 'x'}]),
        )


def test_spec_auto_adds_total_count_when_metrics_empty():
    spec = CollectMetricsSpec(
        name='collect_metrics test',
        source='/tmp/data',
        destination='/tmp/metrics/data.parquet',
    )
    assert len(spec.metrics) == 1
    assert spec.metrics[0].name == 'total_count'


def test_spec_auto_adds_total_count_before_other_metrics():
    spec = CollectMetricsSpec(
        name='collect_metrics test',
        source='/tmp/data',
        destination='/tmp/metrics/data.parquet',
        metrics=cast(Any, [{'type': 'distinct_count', 'name': 'x', 'columns': ['id']}]),
    )
    assert spec.metrics[0].name == 'total_count'
    assert spec.metrics[1].name == 'x'


def test_spec_does_not_duplicate_explicit_count():
    spec = CollectMetricsSpec(
        name='collect_metrics test',
        source='/tmp/data',
        destination='/tmp/metrics/data.parquet',
        metrics=cast(Any, [{'type': 'count', 'name': 'my_count'}]),
    )
    assert sum(1 for m in spec.metrics if m.name == 'my_count') == 1
    assert len(spec.metrics) == 1


@pytest.mark.parametrize('config_name', ['config.yaml', 'metrics.yaml'])
def test_checked_in_metric_configs_are_valid(config_name):
    repo_root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((repo_root / config_name).read_text())

    collect_metric_specs = [
        task
        for tasks in config['steps'].values()
        for task in tasks
        if task['name'].startswith('collect_metrics ')
    ]

    assert collect_metric_specs
    for spec in collect_metric_specs:
        parsed = CollectMetricsSpec(**spec)
        assert any(m.name == 'total_count' for m in parsed.metrics)


# ── Task integration tests ────────────────────────────────────────────────────

def test_collect_metrics_writes_parquet(tmp_path):
    data_dir = tmp_path / 'output' / 'target'
    data_dir.mkdir(parents=True)
    pl.DataFrame({'id': ['A', 'B', 'C']}).write_parquet(data_dir / 'part.parquet')

    out = tmp_path / 'metrics' / 'target.parquet'
    spec = CollectMetricsSpec(
        name='collect_metrics target',
        source=str(data_dir),
        destination=str(out),
        metrics=cast(Any, [{'type': 'count', 'name': 'total_count'}]),
    )
    asyncio.run(CollectMetrics(spec=spec, context=_make_context()).run())

    record = pl.read_parquet(out).row(0, named=True)
    assert json.loads(record['result'])['value'] == 3
    assert record['release'] == '26.06-pub'
    assert record['run'] == 'testrun.1'


def test_collect_metrics_destination_is_exact_output_file(tmp_path):
    """destination is the Parquet file path — no extra subdirectory appended."""
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    pl.DataFrame({'x': [1]}).write_parquet(data_dir / 'part.parquet')

    out = tmp_path / 'my' / 'exact' / 'path.parquet'
    spec = CollectMetricsSpec(
        name='collect_metrics data',
        source=str(data_dir),
        destination=str(out),
        metrics=cast(Any, [{'type': 'count', 'name': 'total_count'}]),
    )
    asyncio.run(CollectMetrics(spec=spec, context=_make_context()).run())

    assert out.exists()
    assert not (tmp_path / 'my' / 'exact' / 'path' / 'total_count.parquet').exists()


def test_bad_column_raises(tmp_path):
    from pts.tasks.collect_metrics import _read
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    pl.DataFrame({'x': [1]}).write_parquet(data_dir / 'part.parquet')
    with pytest.raises(Exception, match='nonexistent'):
        _read(data_dir, ['nonexistent'])


def test_collect_metrics_resolves_relative_paths(tmp_path):
    """source and destination resolve relative to config.work_path."""
    data_dir = tmp_path / 'output' / 'my_dataset'
    data_dir.mkdir(parents=True)
    pl.DataFrame({'id': [1, 2]}).write_parquet(data_dir / 'part.parquet')

    spec = CollectMetricsSpec(
        name='collect_metrics my_dataset',
        source='output/my_dataset',
        destination='metrics/my_dataset.parquet',
        metrics=cast(Any, [{'type': 'count', 'name': 'total_count'}]),
    )
    asyncio.run(CollectMetrics(spec=spec, context=_make_context(work_path=tmp_path)).run())

    out = tmp_path / 'metrics' / 'my_dataset.parquet'
    assert json.loads(pl.read_parquet(out).row(0, named=True)['result'])['value'] == 2

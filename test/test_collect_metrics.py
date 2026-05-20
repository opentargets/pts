"""Tests for CollectMetrics task."""
from __future__ import annotations

import asyncio
import json
from threading import Event
from unittest.mock import MagicMock

import polars as pl
import pytest
from otter.scratchpad.model import Scratchpad
from otter.task.model import TaskContext

from pts.tasks.collect_metrics import CollectMetrics, CollectMetricsSpec


def _make_context(work_path=None, release='26.06-pub', run='testrun.1'):
    config = MagicMock()
    config.release_uri = None
    if work_path is not None:
        config.work_path = work_path
    scratchpad = Scratchpad(sentinel_dict={'release': release, 'run': run})
    context = TaskContext(config=config, scratchpad=scratchpad)
    context.abort = Event()
    return context


# ── Spec field tests ──────────────────────────────────────────────────────────

def test_spec_rejects_unknown_metric_type():
    with pytest.raises(ValueError, match='bad_type'):
        CollectMetricsSpec(
            name='collect_metrics test',
            source='/tmp/data',
            destination='/tmp/metrics/data.jsonl',
            metrics=[{'type': 'bad_type', 'name': 'x'}],
        )


# ── Task integration tests ────────────────────────────────────────────────────

def test_collect_metrics_writes_jsonl(tmp_path):
    data_dir = tmp_path / 'output' / 'target'
    data_dir.mkdir(parents=True)
    pl.DataFrame({'id': ['A', 'B', 'C']}).write_parquet(data_dir / 'part.parquet')

    out = tmp_path / 'metrics' / 'target.jsonl'
    spec = CollectMetricsSpec(
        name='collect_metrics target',
        source=str(data_dir),
        destination=str(out),
        metrics=[{'type': 'count', 'name': 'total_count'}],
    )
    asyncio.run(CollectMetrics(spec=spec, context=_make_context()).run())

    assert out.exists()
    record = json.loads(out.read_text().splitlines()[0])
    assert json.loads(record['result'])['value'] == 3
    assert record['release'] == '26.06-pub'
    assert record['run'] == 'testrun.1'


def test_collect_metrics_destination_is_exact_output_file(tmp_path):
    """destination is the JSONL file path — no extra subdirectory appended."""
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    pl.DataFrame({'x': [1]}).write_parquet(data_dir / 'part.parquet')

    out = tmp_path / 'my' / 'exact' / 'path.jsonl'
    spec = CollectMetricsSpec(
        name='collect_metrics data',
        source=str(data_dir),
        destination=str(out),
        metrics=[{'type': 'count', 'name': 'total_count'}],
    )
    asyncio.run(CollectMetrics(spec=spec, context=_make_context()).run())

    assert out.exists()
    assert not (tmp_path / 'my' / 'exact' / 'path' / 'total_count.json').exists()


def test_collect_metrics_resolves_relative_paths(tmp_path):
    """source and destination resolve relative to config.work_path."""
    data_dir = tmp_path / 'output' / 'my_dataset'
    data_dir.mkdir(parents=True)
    pl.DataFrame({'id': [1, 2]}).write_parquet(data_dir / 'part.parquet')

    spec = CollectMetricsSpec(
        name='collect_metrics my_dataset',
        source='output/my_dataset',
        destination='metrics/my_dataset.jsonl',
        metrics=[{'type': 'count', 'name': 'total_count'}],
    )
    asyncio.run(CollectMetrics(spec=spec, context=_make_context(work_path=tmp_path)).run())

    out = tmp_path / 'metrics' / 'my_dataset.jsonl'
    assert out.exists()
    assert json.loads(json.loads(out.read_text().splitlines()[0])['result'])['value'] == 2

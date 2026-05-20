"""Tests for CollectMetrics task (TDD) — written before implementation."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from threading import Event
from unittest.mock import patch

import polars as pl
import pytest
from otter.scratchpad.model import Scratchpad
from otter.task.model import TaskContext

from pts.tasks.collect_metrics import CollectMetrics, CollectMetricsSpec


def _make_context(release='26.06-pub', run='testrun.1'):
    from unittest.mock import MagicMock
    config = MagicMock()
    config.release_uri = None
    scratchpad = Scratchpad(sentinel_dict={'release': release, 'run': run})
    context = TaskContext(config=config, scratchpad=scratchpad)
    context.abort = Event()
    return context


# ── CollectMetricsSpec field tests ────────────────────────────────────────────

def test_spec_rejects_unknown_metric_type():
    with pytest.raises(ValueError, match='bad_type'):
        CollectMetricsSpec(
            name='collect_metrics test',
            dataset_path='/tmp/data',
            metrics_root='/tmp/metrics',
            metrics=[{'type': 'bad_type', 'name': 'x'}],
        )


def test_spec_accepts_empty_metrics():
    spec = CollectMetricsSpec(
        name='collect_metrics test',
        dataset_path='/tmp/data',
        metrics_root='/tmp/metrics',
        metrics=[],
    )
    assert spec.metrics == []


# ── CollectMetrics task integration tests ────────────────────────────────────

def test_collect_metrics_writes_json(tmp_path):
    df = pl.DataFrame({'id': ['A', 'B', 'C']})
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    df.write_parquet(data_dir / 'part.parquet')

    spec = CollectMetricsSpec(
        name='collect_metrics test',
        dataset_path=str(data_dir),
        metrics_root=str(tmp_path / 'metrics'),
        metrics=[{'type': 'count', 'name': 'total_count'}],
    )
    task = CollectMetrics(spec=spec, context=_make_context())
    asyncio.run(task.run())

    out = tmp_path / 'metrics' / 'data' / 'total_count.json'
    assert out.exists()
    data = json.loads(out.read_text())
    assert data['value'] == 3
    assert data['release'] == '26.06-pub'
    assert data['run'] == 'testrun.1'


def test_collect_metrics_dataset_name_from_path(tmp_path):
    df = pl.DataFrame({'val': [1.0, 2.0, 3.0]})
    data_dir = tmp_path / 'my_dataset'
    data_dir.mkdir()
    df.write_parquet(data_dir / 'part.parquet')

    spec = CollectMetricsSpec(
        name='collect_metrics my_dataset',
        dataset_path=str(data_dir),
        metrics_root=str(tmp_path / 'metrics'),
        metrics=[{'type': 'count', 'name': 'total_count'}],
    )
    task = CollectMetrics(spec=spec, context=_make_context())
    asyncio.run(task.run())

    assert (tmp_path / 'metrics' / 'my_dataset' / 'total_count.json').exists()

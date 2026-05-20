"""Tests for Transform task metric integration (Issue 8) — TDD."""
from __future__ import annotations

import asyncio
from threading import Event
from unittest.mock import MagicMock, patch

import pytest
from otter.scratchpad.model import Scratchpad
from otter.task.model import TaskContext

from pts.tasks.transform import Transform, TransformSpec


def _make_spec(metrics):
    return TransformSpec(
        name='transform test step',
        transformer='disease',
        source='input/x',
        destination='output/disease/disease.parquet',
        metrics=metrics,
    )


def _make_context(release='26.06-pub', run='testrun.1'):
    config = MagicMock()
    config.release_uri = True  # truthy → skip _prepare_dirs in Transform.run
    scratchpad = Scratchpad(sentinel_dict={'release': release, 'run': run})
    context = TaskContext(config=config, scratchpad=scratchpad)
    context.abort = Event()
    return context


# ── TransformSpec field tests (no file I/O) ──────────────────────────────────

def test_transform_spec_accepts_metrics_field():
    spec = _make_spec([{'type': 'count', 'name': 'total_count'}])
    assert len(spec.metrics) == 1


def test_transform_spec_metrics_defaults_to_empty():
    spec = _make_spec([])
    assert spec.metrics == []


def test_unknown_metric_type_raises():
    with pytest.raises(ValueError, match='unknown_type'):
        _make_spec([{'type': 'unknown_type', 'name': 'x'}])


# ── Transform task integration tests (file I/O mocked) ───────────────────────

@patch('pts.tasks.transform.MetricRunner')
@patch('pts.tasks.transform.make_absolute', side_effect=lambda files, config: files)
@patch('pts.tasks.transform.import_module')
def test_metric_runner_called_when_metrics_present(mock_import, mock_abs, mock_runner_cls):
    mock_import.return_value = MagicMock()
    mock_runner = MagicMock()
    mock_runner_cls.return_value = mock_runner

    spec = _make_spec([{'type': 'count', 'name': 'total_count'}])
    context = _make_context()
    task = Transform(spec=spec, context=context)
    asyncio.run(task.run())

    mock_runner.run.assert_called_once()
    call_kwargs = mock_runner.run.call_args.kwargs
    assert call_kwargs['release'] == '26.06-pub'
    assert call_kwargs['run'] == 'testrun.1'
    assert call_kwargs['dataset_name'] == 'disease'


@patch('pts.tasks.transform.MetricRunner')
@patch('pts.tasks.transform.make_absolute', side_effect=lambda files, config: files)
@patch('pts.tasks.transform.import_module')
def test_metric_runner_not_called_when_no_metrics(mock_import, mock_abs, mock_runner_cls):
    mock_import.return_value = MagicMock()
    mock_runner = MagicMock()
    mock_runner_cls.return_value = mock_runner

    spec = _make_spec([])
    context = _make_context()
    task = Transform(spec=spec, context=context)
    asyncio.run(task.run())

    mock_runner.run.assert_not_called()

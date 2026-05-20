"""Task that runs metric collection on an already-produced dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

from loguru import logger
from otter.task.model import Spec, Task, TaskContext
from otter.task.task_reporter import report
from pydantic import field_validator

from pts.metrics.base import Metric
from pts.metrics.runner import MetricRunner
from pts.tasks.transform import _load_metric


class CollectMetricsSpec(Spec):
    """Configuration for the CollectMetrics task."""

    dataset_path: str
    """Absolute path to the directory containing parquet files."""
    metrics_root: str
    """Absolute path to the root directory for metrics JSON output."""
    metrics: list[Metric] = []
    """Metric definitions to compute on the dataset."""

    @field_validator('metrics', mode='before')
    @classmethod
    def _parse_metrics(cls, v: list[dict[str, Any]]) -> list[Metric]:
        return [_load_metric(cfg) for cfg in v]


class CollectMetrics(Task):
    """Task that computes metrics on an already-produced dataset."""

    def __init__(self, spec: CollectMetricsSpec, context: TaskContext) -> None:
        super().__init__(spec, context)
        self.spec: CollectMetricsSpec

    @report
    def run(self) -> Self:
        dataset_path = Path(self.spec.dataset_path)
        dataset_name = dataset_path.name
        metrics_root = Path(self.spec.metrics_root)
        release = self.context.scratchpad.sentinel_dict.get('release', '')
        run = self.context.scratchpad.sentinel_dict.get('run', '')

        logger.info(f'collecting {len(self.spec.metrics)} metrics for {dataset_name}')
        MetricRunner().run(
            metrics=self.spec.metrics,
            dataset_path=dataset_path,
            metrics_root=metrics_root,
            dataset_name=dataset_name,
            release=release,
            run=run,
        )
        return self

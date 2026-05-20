"""Task that runs metric collection on an already-produced dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self, cast

from loguru import logger
from otter.storage.util import make_absolute
from otter.task.model import Spec, Task, TaskContext
from otter.task.task_reporter import report
from pydantic import field_validator

from pts.metrics.base import Metric
from pts.metrics.loader import load_metric, metric_to_dict
from pts.metrics.runner import MetricRunner


class CollectMetricsSpec(Spec):
    """Configuration for the CollectMetrics task."""

    source: str
    """Path to the directory containing parquet files (absolute or relative to work_path)."""
    destination: str
    """Exact output directory for metric JSON files (absolute or relative to work_path)."""
    metrics: list[Metric] = []
    """Metric definitions to compute on the dataset."""

    @field_validator('metrics', mode='before')
    @classmethod
    def _parse_metrics(cls, v: list[Any]) -> list[Metric]:
        return [cfg if isinstance(cfg, Metric) else load_metric(cfg) for cfg in v]

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        d = super().model_dump(**kwargs)
        d['metrics'] = [metric_to_dict(m) for m in self.metrics]
        return d


class CollectMetrics(Task):
    """Task that computes metrics on an already-produced dataset."""

    def __init__(self, spec: CollectMetricsSpec, context: TaskContext) -> None:
        super().__init__(spec, context)
        self.spec: CollectMetricsSpec
        release = context.scratchpad.sentinel_dict.get('release')
        run = context.scratchpad.sentinel_dict.get('run')
        if not release:
            raise ValueError("scratchpad is missing required key 'release'")
        if not run:
            raise ValueError("scratchpad is missing required key 'run'")
        self._release = release
        self._run = run

    @report
    def run(self) -> Self:
        dataset_path = Path(cast(str, make_absolute(self.spec.source, self.context.config)))
        out_file = Path(cast(str, make_absolute(self.spec.destination, self.context.config)))

        logger.info(f'collecting {len(self.spec.metrics)} metrics for {out_file.stem}')
        MetricRunner().run(
            metrics=self.spec.metrics,
            dataset_path=dataset_path,
            out_file=out_file,
            release=self._release,
            run=self._run,
            source=str(dataset_path),
            destination=str(out_file),
        )
        return self

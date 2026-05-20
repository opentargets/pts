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
    """Configuration for the CollectMetrics task.

    Attributes:
        source: Path to the directory containing parquet files
            (absolute, or relative to ``config.work_path``).
        destination: Exact path of the output JSONL file
            (absolute, or relative to ``config.work_path``).
        metrics: One or more metric definitions to compute on the dataset.
            Must not be empty.
    """

    source: str
    destination: str
    metrics: list[Metric]

    @field_validator('metrics', mode='before')
    @classmethod
    def _parse_metrics(cls, v: list[Any]) -> list[Metric]:
        if not v:
            raise ValueError("'metrics' must contain at least one metric")
        return [cfg if isinstance(cfg, Metric) else load_metric(cfg) for cfg in v]

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        d = super().model_dump(**kwargs)
        d['metrics'] = [metric_to_dict(m) for m in self.metrics]
        return d


class CollectMetrics(Task):
    """Reads ``release`` and ``run`` from the scratchpad and runs all configured metrics.

    Raises ``ValueError`` at construction time if either scratchpad key is absent,
    so misconfiguration is caught before the pipeline starts.
    """

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

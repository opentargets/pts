"""Task that runs metric collection on an already-produced dataset."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Self, cast

import polars as pl
from loguru import logger
from otter.storage.util import make_absolute
from otter.task.model import Spec, Task, TaskContext
from otter.task.task_reporter import report
from pydantic import field_serializer, field_validator

from pts.metrics.base import Metric
from pts.metrics.loader import MetricType


def _read(dataset_path: Path, columns: list[str] | None) -> pl.DataFrame:
    """Load parquet files with optional column projection to minimise memory use."""
    lf = pl.scan_parquet(dataset_path / '**' / '*.parquet')
    if columns is None:
        return cast(pl.DataFrame, lf.collect())
    if columns:
        return cast(pl.DataFrame, lf.select(columns).collect())
    # empty list → just need row count; read the first column only
    first = lf.collect_schema().names()[0]
    return cast(pl.DataFrame, lf.select(first).collect())


class CollectMetricsSpec(Spec):
    """Configuration for the CollectMetrics task."""

    source: str
    """Path to the directory containing parquet files (absolute, or relative to ``config.work_path``)."""
    destination: str
    """Exact path of the output Parquet file (absolute, or relative to ``config.work_path``)."""
    metrics: list[Metric]
    """One or more metric definitions to compute on the dataset. Must not be empty."""

    @field_validator('metrics', mode='before')
    @classmethod
    def _parse_metrics(cls, v: list[Any]) -> list[Metric]:
        if not v:
            raise ValueError("'metrics' must contain at least one metric")
        return [cfg if isinstance(cfg, Metric) else MetricType.load(cfg) for cfg in v]

    @field_serializer('metrics')
    def _serialize_metrics(self, v: list[Metric]) -> list[dict[str, Any]]:
        return [m.model_dump() for m in v]


class CollectMetrics(Task):
    """Reads ``release`` and ``run`` from the scratchpad and runs all configured metrics concurrently.

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

        def _compute(metric: Metric) -> dict:
            try:
                df = _read(dataset_path, metric.required_columns)
                result = metric.run(df)
                return result.to_unified_record(
                    release=self._release,
                    run=self._run,
                    dataset=dataset_path.name,
                    source=str(dataset_path),
                    destination=str(out_file),
                ).model_dump()
            except Exception:
                logger.error('metric {} failed on dataset {}', metric.name, dataset_path.name)
                raise

        with ThreadPoolExecutor() as executor:
            records = list(executor.map(_compute, self.spec.metrics))

        pl.DataFrame(records).write_parquet(str(out_file), mkdir=True)
        return self

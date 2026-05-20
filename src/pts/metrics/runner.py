"""MetricRunner: reads parquet, stamps release/run, writes JSON."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import polars as pl

from pts.metrics.base import Metric

log = logging.getLogger(__name__)


class MetricRunner:
    def run(
        self,
        *,
        metrics: Sequence[Metric],
        dataset_path: Path,
        metrics_root: Path,
        dataset_name: str,
        release: str,
        run: str,
    ) -> None:
        df = pl.read_parquet(dataset_path / '*.parquet')
        out_dir = metrics_root / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for metric in metrics:
            try:
                result = metric.compute(df)
                stamped = result.model_copy(update={'release': release, 'run': run})
                (out_dir / f'{metric.name}.json').write_text(
                    stamped.model_dump_json()
                )
            except Exception:
                log.error('metric %s failed on dataset %s', metric.name, dataset_name)
                raise

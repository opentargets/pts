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
    def _read(self, dataset_path: Path, columns: list[str] | None) -> pl.DataFrame:
        """Read parquet with optional column projection to minimise memory."""
        lf = pl.scan_parquet(dataset_path / '*.parquet')
        if columns is None:
            return lf.collect()
        if columns:
            return lf.select(columns).collect()
        # empty list → just need row count; read the first column only
        first = lf.collect_schema().names()[0]
        return lf.select(first).collect()

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
        out_dir = metrics_root / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for metric in metrics:
            try:
                df = self._read(dataset_path, metric.required_columns)
                result = metric.compute(df)
                stamped = result.model_copy(update={'release': release, 'run': run})
                (out_dir / f'{metric.name}.json').write_text(
                    stamped.model_dump_json()
                )
            except Exception:
                log.error('metric %s failed on dataset %s', metric.name, dataset_name)
                raise

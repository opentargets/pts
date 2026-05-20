"""MetricRunner: reads parquet, computes metrics, writes unified JSONL output."""
from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import polars as pl

from pts.metrics.base import Metric

log = logging.getLogger(__name__)


class MetricRunner:
    """Run configured metrics against a parquet dataset and emit JSONL records."""

    def _read(self, dataset_path: Path, columns: list[str] | None) -> pl.DataFrame:
        """Read parquet with optional column projection to minimise memory."""
        lf = pl.scan_parquet(dataset_path / '*.parquet')
        if columns is None:
            return cast(pl.DataFrame, lf.collect())
        if columns:
            return cast(pl.DataFrame, lf.select(columns).collect())
        # empty list → just need row count; read the first column only
        first = lf.collect_schema().names()[0]
        return cast(pl.DataFrame, lf.select(first).collect())

    def run(
        self,
        *,
        metrics: Sequence[Metric],
        dataset_path: Path,
        out_file: Path,
        release: str,
        run: str,
        source: str = '',
        destination: str = '',
    ) -> None:
        """Compute metrics for one dataset and write one unified record per metric."""
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with out_file.open('w') as fh:
            for metric in metrics:
                try:
                    df = self._read(dataset_path, metric.required_columns)
                    result = metric.compute(df)
                    stamped = result.model_copy(update={
                        'release': release,
                        'run': run,
                        'dataset': dataset_path.name,
                        'source': source,
                        'destination': destination,
                    })
                    record = stamped.to_unified_record()
                    fh.write(record.model_dump_json() + '\n')
                except Exception:
                    log.error('metric %s failed on dataset %s', metric.name, dataset_path.name)
                    raise

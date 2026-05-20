"""MetricRunner: reads parquet, computes metrics, writes unified JSONL output."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import polars as pl
from loguru import logger

from pts.metrics.base import Metric


class MetricRunner:
    """Reads a parquet dataset, runs each configured metric, and writes JSONL output.

    Each metric produces one line in the output file via
    :meth:`~pts.metrics.base.MetricResult.to_unified_record`.
    """

    def _read(self, dataset_path: Path, columns: list[str] | None) -> pl.DataFrame:
        """Load parquet files with optional column projection to minimise memory use."""
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
        source: str,
        destination: str,
    ) -> None:
        """Compute all metrics for one dataset and write one JSONL record per metric.

        Args:
            metrics: Ordered sequence of metrics to compute.
            dataset_path: Directory containing ``*.parquet`` files.
            out_file: JSONL file to create (parent directories are created if absent).
            release: Pipeline release string stamped into every record.
            run: Pipeline run identifier stamped into every record.
            source: Absolute path of the input dataset, recorded in each record.
            destination: Absolute path of the output file, recorded in each record.
        """
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with out_file.open('w') as fh:
            for metric in metrics:
                try:
                    df = self._read(dataset_path, metric.required_columns)
                    result = metric.run(df)
                    record = result.to_unified_record(
                        release=release,
                        run=run,
                        dataset=dataset_path.name,
                        source=source,
                        destination=destination,
                    )
                    fh.write(record.model_dump_json() + '\n')
                except Exception:
                    logger.error('metric {} failed on dataset {}', metric.name, dataset_path.name)
                    raise

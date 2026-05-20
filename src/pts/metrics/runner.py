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
        lf = pl.scan_parquet(dataset_path / '**' / '*.parquet')
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
        out_file: str | Path,
        release: str,
        run: str,
        source: str,
        destination: str,
    ) -> None:
        """Compute all metrics for one dataset and write one Parquet record per metric.

        ``out_file`` may be a local path or a cloud URI (``gs://``, ``s3://``,
        ``az://``). Polars handles the storage backend transparently; local
        parent directories are created automatically.

        Args:
            metrics: Ordered sequence of metrics to compute.
            dataset_path: Directory containing ``*.parquet`` files.
            out_file: Destination Parquet path — local or cloud URI.
            release: Pipeline release string stamped into every record.
            run: Pipeline run identifier stamped into every record.
            source: Absolute path of the input dataset, recorded in each record.
            destination: Absolute path of the output file, recorded in each record.
        """
        out_path = str(out_file)
        if not out_path.startswith(('gs://', 's3://', 'az://')):
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)

        records = []
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
                records.append(record.model_dump())
            except Exception:
                logger.error('metric {} failed on dataset {}', metric.name, dataset_path.name)
                raise

        pl.DataFrame(records).write_parquet(out_path)

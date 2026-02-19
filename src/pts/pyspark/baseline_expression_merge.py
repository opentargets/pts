"""Merges multiple parquet datasets into a single parquet output and (optionally) writes a small JSON sample.

Requires Apache Spark (pyspark) for distributed processing.

Expected input layout (per dataset):
- Base directory: <base>/<aggregation>/<dataset>/parquet/*
  where <aggregation> is one of: aggregated, unaggregated
  and <dataset> is a directory name (e.g., gtex, pride, tabula_sapiens, dice)

Outputs:
- Merged parquet: <base>/<aggregation>/merged/parquet
- Small validation sample (JSON, optional, disable with --no-sample): <base>/<aggregation>/merged_sample/json
"""

from __future__ import annotations

import glob
import os
from typing import Any

from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand, row_number
from pyspark.sql.window import Window

from pts.pyspark.common.session import Session

# Default Spark properties for the merge step
_MERGE_DEFAULT_PROPERTIES: dict[str, str] = {
    'spark.driver.memory': '50g',
    'spark.executor.memory': '70g',
    'spark.memory.offHeap.enabled': 'true',
    'spark.memory.offHeap.size': '16g',
    'spark.sql.shuffle.partitions': '200',
    'spark.default.parallelism': '200',
    'spark.sql.files.maxPartitionBytes': '134217728',  # 128MB per partition
}


class MergeParquetDatasets:
    """Collection of steps to read, merge, repartition, and write parquet datasets."""

    def read_input_data(self):
        """Read the input parquet datasets and union them with schema alignment."""
        # Expand globs first to handle files individually and avoid schema inference conflicts
        expanded_paths = []
        for d in self.datasets:
            pattern = f"{self.base_directory_path.rstrip('/')}/{self.aggregation}/{d}/parquet/*"
            matches = glob.glob(pattern)
            if matches:
                for m in matches:
                    # Filter out _SUCCESS or other metadata files if they are not directories
                    if os.path.basename(m).startswith('_') and os.path.isfile(m):
                        continue
                    expanded_paths.append(m)
            else:
                logger.warning(f'Warning: No files found for {pattern}')

        if not expanded_paths:
            raise ValueError('No dataset paths were constructed. Check the datasets setting.')

        logger.info(f'Reading {len(expanded_paths)} parquet inputs in parallel:')
        for p in expanded_paths:
            logger.info(f'  • {p}')

        # Read all files at once with mergeSchema so Spark parallelises across all cores.
        # This avoids the deep linear DAG from sequential unionByName calls.
        df = (self.spark.read
              .option('mergeSchema', 'true')
              .parquet(*expanded_paths))

        # Handle schema mismatch for 'specificity score'
        # It might be inferred as Binary in some files but Double in others.
        if 'specificity score' in df.columns:
            df = df.withColumn('specificity score', col('specificity score').cast('string').cast('double'))

        if self.aggregation == 'aggregated':
            # Filter any rows where tissue/celltypeBiosampleId is populated but tissue/celltypeBiosampleParentId is null
            for coltype in ['tissue', 'celltype']:
                biosample_id_col = f'{coltype}BiosampleId'
                biosample_parent_id_col = f'{coltype}BiosampleParentId'
                if biosample_id_col in df.columns and biosample_parent_id_col in df.columns:
                    df = df.filter(~(
                            (col(biosample_id_col).isNotNull()) &
                            (col(biosample_parent_id_col).isNull())))

        # Reorder the columns so that identifier columns come first (if present)
        id_cols = ['targetId', 'datasourceId', 'datatypeId', 'unit', 'tissueBiosampleId', 'tissueBiosampleParentId',
                   'celltypeBiosampleId', 'celltypeBiosampleParentId']
        existing_id_cols = [c for c in id_cols if c in df.columns]
        other_cols = [c for c in df.columns if c not in existing_id_cols]
        df = df.select(existing_id_cols + other_cols)

        self.df = df
        logger.info('Finished reading and merging input data.')

    def pack_data_for_output(self):
        """Repartition and write merged parquet; optionally write a small JSON sample."""
        # Resolve output base (optionally prefixed with file:// for local mode)
        def _out(path):
            if self.local:
                return f'file://{path}'
            return path

        merged_parquet_out = (
            self.output_directory_path or f"{self.base_directory_path.rstrip('/')}/{self.aggregation}/merged/parquet"
        )
        sample_json_out = (
            self.sample_output_directory_path or
            f"{self.base_directory_path.rstrip('/')}/{self.aggregation}/merged_sample/json"
        )

        if self.num_output_files is not None and self.num_output_files > 0:
            # Explicit file count: skip the expensive .count() entirely
            num_files = int(self.num_output_files)
            logger.info(f'Target number of output files (explicit): {num_files}')
        else:
            # Need row count to calculate partition count
            logger.info('Counting rows...')
            total_rows = self.df.count()
            logger.info(f'Total rows detected: {total_rows:,}')
            rpf = int(self.rows_per_file) if self.rows_per_file and self.rows_per_file > 0 else 10_000_000
            num_files = max(1, (total_rows + rpf - 1) // rpf)
            logger.info(f'Target number of output files: {num_files}')

        # Repartition for balanced file sizes
        logger.info('Repartitioning (shuffle)...')
        df_out = self.df.repartition(num_files)

        # Write merged parquet
        logger.info(f'Writing merged parquet to: {merged_parquet_out}')
        df_out.write.mode('overwrite').parquet(_out(merged_parquet_out))
        logger.info('Finished writing merged parquet output.')

        # Optionally write small JSON sample, balanced across datasourceId if present
        if self.no_sample:
            logger.info('Skipping sample JSON (no_sample set).')
            return

        n_rows = int(self.sample_rows) if self.sample_rows and self.sample_rows > 0 else 0
        if n_rows <= 0:
            logger.info('Skipping sample JSON (sample_rows <= 0).')
            return

        logger.info(f'Preparing JSON sample of ~{n_rows} rows...')
        cols = set(self.df.columns)
        if 'datasourceId' in cols:
            n_datasources = self.df.select('datasourceId').distinct().count()
            # ceil division
            k = (n_rows + n_datasources - 1) // max(1, n_datasources)
            w = Window.partitionBy('datasourceId').orderBy(rand())
            sample_df = (
                self.df.withColumn('rn', row_number().over(w))
                       .where(col('rn') <= k)
                       .drop('rn')
                       .limit(n_rows)
            )
        else:
            # Fallback: simple random sample if no datasourceId column
            logger.warning("Column 'datasourceId' not found; sampling without stratification.")
            sample_df = self.df.orderBy(rand()).limit(n_rows)

        logger.info(f'Writing JSON sample to: {sample_json_out}')
        sample_df.write.mode('overwrite').json(_out(sample_json_out))
        logger.info('Finished writing sample JSON output.')

    def __init__(
        self,
        spark: SparkSession,
        base_directory_path: str,
        aggregation: str,
        datasets: str | list[str],
        output_directory_path: str | None = None,
        sample_output_directory_path: str | None = None,
        rows_per_file: int = 10_000_000,
        num_output_files: int | None = None,
        sample_rows: int = 100,
        local: bool = False,
        no_sample: bool = False,
    ):
        self.spark = spark
        self.base_directory_path = base_directory_path
        self.aggregation = aggregation
        # Support comma-separated string or list
        if isinstance(datasets, str):
            self.datasets = [d.strip() for d in datasets.split(',') if d.strip()]
        else:
            self.datasets = list(datasets)

        self.output_directory_path = output_directory_path
        self.sample_output_directory_path = sample_output_directory_path
        self.rows_per_file = rows_per_file
        self.num_output_files = num_output_files
        self.sample_rows = sample_rows
        self.local = local
        self.no_sample = no_sample

    def run(self):
        """Execute the merge pipeline."""
        logger.info('Starting merge pipeline...')
        self.read_input_data()
        logger.info('Packing data for output...')
        self.pack_data_for_output()
        logger.info('All done.')


def baseline_expression_merge(
    source: str | dict[str, str],
    destination: str | dict[str, str],
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    logger.info('Starting baseline expression merge')

    # Initialize Spark Session
    if properties is None:
        properties = {}

    # Merge step defaults with any caller-supplied overrides
    effective_properties = {**_MERGE_DEFAULT_PROPERTIES, **properties}

    session = Session(app_name='baseline_expression_merge', properties=effective_properties)
    spark = session.spark

    # Extract arguments from source/destination/settings
    if isinstance(source, str):
        base_directory_path = source
    else:
        base_directory_path = source.get('base_directory_path', '')

    if isinstance(destination, str):
        output_directory_path = destination
        sample_output_directory_path = None
    else:
        output_directory_path = destination.get('merged_output')
        sample_output_directory_path = destination.get('sample_output')

    aggregation = settings.get('aggregation', 'aggregated')
    datasets = settings.get('datasets', 'gtex,pride,tabula_sapiens,dice')
    local = settings.get('local', False)

    merger = MergeParquetDatasets(
        spark=spark,
        base_directory_path=base_directory_path,
        aggregation=aggregation,
        datasets=datasets,
        output_directory_path=output_directory_path,
        sample_output_directory_path=sample_output_directory_path,
        rows_per_file=settings.get('rows_per_file', 10_000_000),
        num_output_files=settings.get('num_output_files'),
        sample_rows=settings.get('sample_rows', 100),
        local=local,
        no_sample=settings.get('no_sample', False),
    )
    merger.run()
    logger.info('Baseline expression merge completed successfully')

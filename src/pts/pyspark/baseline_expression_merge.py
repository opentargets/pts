from __future__ import annotations
from typing import Any
from loguru import logger
from pyspark.sql import SparkSession
from pts.pyspark.common.session import Session


def _merge_baseline_expression(
    spark: SparkSession,
    source: str,
    destination: str,
    settings: dict[str, Any],
) -> None:
    aggregation = settings.get('aggregation')
    if not aggregation:
        raise ValueError("settings['aggregation'] is required")

    raw_datasets = settings.get('datasets', [])
    if isinstance(raw_datasets, str):
        raw_datasets = raw_datasets.split(',')
    datasets = [d.strip().strip('/') for d in raw_datasets if d.strip()]
    if not datasets:
        raise ValueError("settings['datasets'] must contain at least one dataset")

    source_root = source.rstrip('/')
    input_paths = [f'{source_root}/{aggregation}/{d}/parquet' for d in datasets]

    missing_paths = []
    for path in input_paths:
        hadoop_path = spark._jvm.org.apache.hadoop.fs.Path(path)
        fs = hadoop_path.getFileSystem(spark._jsc.hadoopConfiguration())
        if not fs.exists(hadoop_path):
            missing_paths.append(path)

    if missing_paths:
        raise FileNotFoundError(f"Missing baseline expression parquet inputs: {', '.join(missing_paths)}")

    logger.info(f'Resolved {len(input_paths)} baseline expression parquet inputs')
    for input_path in input_paths:
        logger.info(f'input path: {input_path}')
    logger.info(f'output path: {destination}')

    (
        spark.read.option('mergeSchema', 'true')
        .parquet(*input_paths)
        .write.mode('overwrite')
        .parquet(destination)
    )


def baseline_expression_merge(
    source: str,
    destination: str,
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    logger.info('Starting baseline expression merge')

    session = Session(app_name='baseline_expression_merge', properties=properties)
    try:
        _merge_baseline_expression(session.spark, source, destination, settings)
    finally:
        session.stop()

    logger.info('Baseline expression merge completed successfully')
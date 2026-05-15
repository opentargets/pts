"""Generate target-prioritisation factors from aggregated baseline expression."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as f
from pyspark.sql.window import Window

from pts.pyspark.common.session import Session

_DEFAULT_EXCLUDED_DATASOURCES = ['PRIDE', 'DICE']
_KIND_COLS = {
    'tissue': ('Tissue', 'tissueBiosampleFromSource'),
    'celltype': ('Cell type', 'celltypeBiosampleFromSource'),
}


def _capped(column: Column) -> Column:
    """Return -1.0 sentinel when value is null or non-positive, otherwise the value."""
    return f.when(column.isNull() | (column <= 0), f.lit(-1.0)).otherwise(column)


def _project(best: DataFrame, kind: str) -> DataFrame:
    prefix, source_col = _KIND_COLS[kind]
    return (
        best.filter(f.col('biosample_type') == kind)
        .select(
            f.col('targetId').alias('GeneID'),
            f
            .when(f.col('distribution_score') == 1, f.lit(-1.0))
            .otherwise(1 - f.col('distribution_score'))
            .alias(f'{prefix} distribution score'),
            _capped(f.col('specificity_score')).alias(f'{prefix} specificity score'),
            f.col(source_col).alias(f'{prefix} specificity score annotation'),
        )
    )


def _build_target_prioritisation_factors(df: DataFrame, excluded_datasources: list[str]) -> DataFrame:
    typed = (
        df.filter(
            ~f.col('datasourceId').isin(excluded_datasources)
            & ~(f.col('tissueBiosampleId').isNotNull() & f.col('celltypeBiosampleId').isNotNull())
            & f.col('specificity_score').isNotNull()
        )
        .withColumn(
            'biosample_type',
            f
            .when(f.col('tissueBiosampleId').isNotNull(), f.lit('tissue'))
            .when(f.col('celltypeBiosampleId').isNotNull(), f.lit('celltype')),
        )
        .filter(f.col('biosample_type').isNotNull())
    )

    rank_window = Window.partitionBy('targetId', 'biosample_type').orderBy(f.col('specificity_score').desc())
    best = typed.withColumn('rank', f.row_number().over(rank_window)).filter(f.col('rank') == 1).drop('rank')

    return (
        _project(best, 'tissue')
        .join(_project(best, 'celltype'), on='GeneID', how='outer')
        .withColumn('Tissue specificity score', _capped(f.col('Tissue specificity score')))
        .withColumn('Cell type specificity score', _capped(f.col('Cell type specificity score')))
    )


def baseline_expression_target_prioritisation(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    logger.info('Starting baseline expression target prioritisation')
    excluded = settings.get('excluded_datasources', _DEFAULT_EXCLUDED_DATASOURCES)

    session = Session(app_name='baseline_expression_target_prioritisation', properties=properties)
    try:
        agg = session.load_data(source['aggregated'], format='parquet')
        result = _build_target_prioritisation_factors(agg, excluded)
        result.write.mode('overwrite').parquet(destination['parquet'])
        if 'tsv' in destination:
            tsv_path = Path(destination['tsv'])
            tsv_path.parent.mkdir(parents=True, exist_ok=True)
            result.orderBy('GeneID').toPandas().to_csv(tsv_path, sep='\t', index=False)
    finally:
        session.stop()
    logger.info('Baseline expression target prioritisation completed successfully')

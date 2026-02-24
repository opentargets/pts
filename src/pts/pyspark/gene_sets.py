"""Computes the gene_sets dataset."""

from typing import Any

from loguru import logger
from pyspark.sql import functions as f

from pts.pyspark.common import Session
from pts.pyspark.gene_sets_utils.gene_sets import FacetSearchCategories, compute_all_target_facets


def gene_sets(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    session = Session(app_name='target_facets', properties=properties)
    spark = session.spark
    category_config = settings.get('category_config')

    logger.info(f'loading targets from: {source["targets"]}')
    targets_df = spark.read.parquet(source['targets'])

    logger.info(f'loading go from: {source["go_processed"]}')
    go_df = spark.read.parquet(source['go_processed'])
    go_df = go_df.filter(f.col('isObsolete').isNull() | (~f.col('isObsolete')))  # ty:ignore[missing-argument]

    logger.info(f'loading reactome from: {source["reactome"]}')
    reactome_df = spark.read.parquet(source['reactome'])

    category_values = FacetSearchCategories(category_config)

    all_facets = compute_all_target_facets(targets_df, go_df, reactome_df, category_values)

    logger.info(f'writing gene sets to: {destination}')
    all_facets.write.mode('overwrite').parquet(destination)

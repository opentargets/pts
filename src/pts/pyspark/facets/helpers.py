"""Helper functions for computing search facets.
"""

from __future__ import annotations

from loguru import logger
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def compute_simple_facet(
    dataframe: DataFrame,
    label_field: str,
    category_value: str,
    entity_id_field: str,
    spark: SparkSession,
) -> DataFrame:
    """Compute simple facet dataset for the given DataFrame.

    This function groups entities by a label field and collects their IDs into sets,
    creating a facet structure with a fixed category value and null datasourceId.

    Args:
        dataframe: Input DataFrame to compute facets from
        label_field: Field name to use as the facet label
        category_value: Fixed string value to use as the facet category
        entity_id_field: Field name to use as entity ID
        spark: SparkSession instance

    Returns:
        DataFrame with facet schema (label, category, entityIds, datasourceId)

    Example:
        >>> compute_simple_facet(targets_df, 'approvedSymbol', 'Approved Symbol', 'id', spark)
    """
    logger.debug(f'Computing simple facet: label={label_field}, category={category_value}')

    return (
        dataframe
        .select(
            F.col(label_field).alias('label'),
            F.lit(category_value).alias('category'),
            F.col(entity_id_field).alias('id')
        )
        .groupBy('label', 'category')
        .agg(F.collect_set('id').alias('entityIds'))
        .withColumn('datasourceId', F.lit(None).cast('string'))
        .withColumn('parentId', F.array().cast('array<string>'))
        .distinct()
    )


def get_relevant_dataset(
    dataframe: DataFrame,
    id_field: str,
    id_alias: str,
    facet_field: str,
) -> DataFrame:
    """Extract relevant columns from a DataFrame and filter out null facet values.

    This helper selects specific columns from the input DataFrame, renames the ID field,
    and filters out rows where the facet field is null.

    Args:
        dataframe: Input DataFrame
        id_field: Name of the ID field in the input DataFrame
        id_alias: Alias to use for the ID field in the output
        facet_field: Name of the field containing facet data

    Returns:
        Filtered DataFrame with renamed ID field and facet field

    Example:
        >>> get_relevant_dataset(targets_df, 'id', 'ensemblGeneId', 'tractability')
    """
    return (
        dataframe
        .select(F.col(id_field).alias(id_alias), F.col(facet_field))
        .where(F.col(facet_field).isNotNull())
    )

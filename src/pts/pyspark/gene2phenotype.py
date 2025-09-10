"""Gene2Phenotype PySpark module for processing G2P data.

This module processes Gene2Phenotype data and generates evidence strings
using PySpark and ontoma for EFO mapping.
"""

from loguru import logger
from ontoma import OnToma
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as f
from pyspark.sql import types as t

from pts.pyspark.common.session import Session


def _map_phenotype_to_efo(phenotype: str) -> list[str]:
    """Map phenotype to EFO IDs using ontoma.

    Args:
        phenotype: Phenotype string to map

    Returns:
        List of EFO IDs
    """
    try:
        if not phenotype or not phenotype.strip():
            return []
        ontoma = OnToma()
        mappings = ontoma.find_term(phenotype)
        return [m.id_normalised for m in mappings] if mappings else []
    except Exception as e:
        logger.warning(f'Error mapping phenotype {phenotype}: {e}')
        return []


def _add_efo_mapping(df: DataFrame, phenotype_col: str = 'phenotype') -> DataFrame:
    """Add EFO mapping to phenotypes using ontoma.

    Args:
        df: Input DataFrame containing phenotype column
        phenotype_col: Name of the phenotype column

    Returns:
        DataFrame with added EFO mappings
    """
    # Create UDF for EFO mapping
    map_to_efo_udf = f.udf(_map_phenotype_to_efo, t.ArrayType(t.StringType()))

    # Add EFO mappings
    return df.withColumn('efo_ids', map_to_efo_udf(f.col(phenotype_col)))


def _process_g2p_data(spark: SparkSession, dataset_path: str) -> DataFrame:
    """Process Gene2Phenotype data.

    Args:
        spark: SparkSession instance
        dataset_path: Path to G2P dataset

    Returns:
        Processed DataFrame with G2P data
    """
    # Read G2P data
    df = spark.read.csv(dataset_path, header=True)

    # Clean and transform data
    df = df.withColumn('gene_symbol', f.trim(f.col('gene_symbol')))
    df = df.withColumn('phenotype', f.trim(f.col('phenotype')))

    # Add EFO mappings
    df = _add_efo_mapping(df)

    # Filter out rows without EFO mappings
    return df.filter(f.size('efo_ids') > 0)


def _generate_evidence(df: DataFrame) -> DataFrame:
    """Generate evidence strings from G2P data.

    Args:
        df: Input DataFrame with G2P data

    Returns:
        DataFrame with evidence strings
    """
    # Explode EFO IDs to create one row per gene-phenotype-EFO combination
    df = df.withColumn('efo_id', f.explode('efo_ids')).drop('efo_ids')

    # Generate evidence strings
    return df.select(
        f.col('gene_symbol'),
        f.col('efo_id'),
        f.lit('gene2phenotype').alias('source_id'),
        f.current_timestamp().alias('date'),
        f.to_json(
            f.struct(
                f.col('phenotype').alias('phenotype_label'),
                f.col('efo_id').alias('phenotype_id'),
                f.col('gene_symbol').alias('gene_symbol'),
                # Add other evidence fields as needed
            )
        ).alias('evidence'),
    )


def gene2phenotype(
    source: dict[str, str],
    destination: dict[str, str],
    properties: dict[str, str] | None = None,
) -> None:
    """Process Gene2Phenotype data and generate evidence.

    Args:
        source: Dictionary containing:
            - dataset: Path to G2P dataset
        destination: Dictionary containing:
            - output: Path to save evidence strings
        properties: Optional Spark properties
    """
    logger.info('Starting Gene2Phenotype processing')

    # Initialize Spark session
    session = Session(app_name='gene2phenotype', properties=properties)

    # Get input/output paths
    dataset_path = source['dataset']
    output_path = destination['output']

    # Process G2P data
    g2p_data = _process_g2p_data(session.spark, dataset_path)

    # Generate evidence strings
    evidence = _generate_evidence(g2p_data)

    # Write evidence strings
    evidence.write.mode('overwrite').parquet(output_path)

    logger.info('Gene2Phenotype processing completed')

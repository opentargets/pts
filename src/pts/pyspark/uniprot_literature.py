"""Evidence parser for UniProt disease-association literature curation."""

from typing import Any

from loguru import logger
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as f

from pts.pyspark.common.ontology import add_efo_mapping
from pts.pyspark.common.session import Session
from pts.pyspark.evidence_utils.uniprot import (
    DATASOURCE_LITERATURE,
    DATATYPE_GENETIC_LITERATURE,
    confidence_from_literature,
    uniprot_urls_struct_array,
)


def _compute_literature(
    spark: SparkSession,
    parsed_path: str,
    disease_label_lut_path: str,
    disease_id_lut_path: str,
) -> DataFrame:
    parsed = spark.read.parquet(parsed_path)
    exploded = parsed.select(
        f.col('accession'),
        f.explode('diseases').alias('disease'),
    )

    with_literature = exploded.filter(f.size(f.col('disease.evidencePmids')) > 0)

    projected = with_literature.select(
        f.lit(DATASOURCE_LITERATURE).alias('datasourceId'),
        f.lit(DATATYPE_GENETIC_LITERATURE).alias('datatypeId'),
        f.col('accession').alias('targetFromSourceId'),
        f.col('disease.name').alias('diseaseFromSource'),
        f.concat(f.lit('OMIM:'), f.col('disease.omimId')).alias('diseaseFromSourceId'),
        f.col('disease.evidencePmids').alias('literature'),
        confidence_from_literature(f.col('disease.evidencePmids')).alias('confidence'),
        uniprot_urls_struct_array(f.col('accession')).alias('urls'),
    )

    logger.info('map uniprot literature diseases to EFO')
    return add_efo_mapping(
        spark=spark,
        evidence_df=projected,
        disease_label_lut_path=disease_label_lut_path,
        disease_id_lut_path=disease_id_lut_path,
    )


def uniprot_literature(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    spark = Session(app_name='uniprot_literature', properties=properties)
    logger.info(f'load data from {source}')

    result_df = _compute_literature(
        spark=spark.spark,
        parsed_path=source['uniprot_evidence'],
        disease_label_lut_path=source['ontoma_disease_label_lut'],
        disease_id_lut_path=source['ontoma_disease_id_lut'],
    )

    logger.info(f'write uniprot literature evidence to {destination}')
    result_df.write.parquet(destination, mode='overwrite')

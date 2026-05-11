"""Evidence parser for UniProt disease-association variant features."""

from typing import Any

from loguru import logger
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as f

from pts.pyspark.common.ontology import add_efo_mapping
from pts.pyspark.common.session import Session
from pts.pyspark.evidence_utils.uniprot import (
    DATASOURCE_VARIANTS,
    DATATYPE_GENETIC_ASSOCIATION,
    DATATYPE_SOMATIC_MUTATIONS,
    confidence_from_literature,
    load_somatic_rsids,
    uniprot_urls_struct_array,
)


def _compute_variants(
    spark: SparkSession,
    parsed_path: str,
    somatic_census_path: str,
    disease_label_lut_path: str,
    disease_id_lut_path: str,
) -> DataFrame:
    parsed = spark.read.parquet(parsed_path)
    exploded_variants = parsed.select(
        f.col('accession'),
        f.col('diseases'),
        f.explode('variants').alias('variant'),
    )

    with_linked = exploded_variants.filter(f.size(f.col('variant.linkedOmimIds')) > 0)
    exploded_omim = with_linked.select(
        f.col('accession'),
        f.col('diseases'),
        f.col('variant'),
        f.explode(f.col('variant.linkedOmimIds')).alias('omimId'),
    )

    # Resolve disease name from the entry's diseases array by matching omimId
    resolved = exploded_omim.withColumn(
        'disease',
        f.element_at(
            f.filter(f.col('diseases'), lambda d: d['omimId'] == f.col('omimId')),
            1,
        ),
    ).filter(f.col('disease').isNotNull())

    somatic = load_somatic_rsids(spark, somatic_census_path).withColumnRenamed(
        'dbSnpRsId', 'somaticRsId'
    ).withColumn('isSomatic', f.lit(True))

    joined = resolved.join(
        somatic,
        resolved['variant.dbSnpRsId'] == somatic['somaticRsId'],
        'left',
    ).drop('somaticRsId')

    projected = joined.select(
        f.lit(DATASOURCE_VARIANTS).alias('datasourceId'),
        f.when(f.col('isSomatic').isNotNull(), f.lit(DATATYPE_SOMATIC_MUTATIONS))
            .otherwise(f.lit(DATATYPE_GENETIC_ASSOCIATION))
            .alias('datatypeId'),
        f.col('accession').alias('targetFromSourceId'),
        f.col('disease.name').alias('diseaseFromSource'),
        f.concat(f.lit('OMIM:'), f.col('omimId')).alias('diseaseFromSourceId'),
        f.col('variant.dbSnpRsId').alias('variantRsId'),
        f.array(f.col('variant.aminoacidChange')).alias('variantAminoacidDescriptions'),
        f.col('variant.evidencePmids').alias('literature'),
        confidence_from_literature(f.col('variant.evidencePmids')).alias('confidence'),
        uniprot_urls_struct_array(f.col('accession')).alias('urls'),
        f.when(f.col('isSomatic').isNotNull(), f.array(f.lit('somatic')))
            .otherwise(f.array(f.lit('germline')))
            .alias('alleleOrigins'),
    )

    logger.info('map uniprot variant diseases to EFO')
    return add_efo_mapping(
        spark=spark,
        evidence_df=projected,
        disease_label_lut_path=disease_label_lut_path,
        disease_id_lut_path=disease_id_lut_path,
    )


def uniprot_variants(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    spark = Session(app_name='uniprot_variants', properties=properties)
    logger.info(f'load data from {source}')

    result_df = _compute_variants(
        spark=spark.spark,
        parsed_path=source['uniprot_evidence'],
        somatic_census_path=source['somatic_census'],
        disease_label_lut_path=source['ontoma_disease_label_lut'],
        disease_id_lut_path=source['ontoma_disease_id_lut'],
    )

    logger.info(f'write uniprot variant evidence to {destination}')
    result_df.write.parquet(destination, mode='overwrite')

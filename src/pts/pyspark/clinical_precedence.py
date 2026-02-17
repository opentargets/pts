"""Generate clinical precedence evidence from clinical reports and drug mechanisms of action."""

from typing import Any

import pyspark.sql.functions as f
from loguru import logger

from pts.pyspark.common.session import Session


def clinical_precedence(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Generate clinical precedence evidence by joining clinical reports with drug mechanisms of action.

    Args:
        source: Dictionary with paths to:
            - clinical_report: Clinical report parquet from clinical_report step
            - drug_mechanism_of_action: Drug mechanism of action parquet
        destination: Path to write the output parquet file.
        settings: Custom settings (not used).
        properties: Spark configuration options.
    """
    spark = Session(app_name='clinical_precedence', properties=properties)

    logger.info(f'Loading data from {source}')
    clinical_report_df = spark.load_data(source['clinical_report'])
    drug_moa_df = spark.load_data(source['drug_mechanism_of_action'])

    # Explode clinical_report: diseases and drugs
    exploded_cr = (
        clinical_report_df.withColumnRenamed('id', 'clinicalReportId')
        .withColumn('studyStartDate', f.col('trialStartDate').cast('string'))
        .withColumn('literature', f.col('trialLiterature'))
        .withColumn('_disease', f.explode_outer('diseases'))
        .withColumn('diseaseFromSource', f.col('_disease.diseaseFromSource'))
        .withColumn('diseaseFromSourceMappedId', f.col('_disease.diseaseId'))
        .withColumn('_drug', f.explode_outer('drugs'))
        .withColumn('drugFromSource', f.col('_drug.drugFromSource'))
        .withColumn('drugId', f.col('_drug.drugId'))
        .filter(f.col('diseaseFromSourceMappedId').isNotNull() & f.col('drugId').isNotNull())
    )

    # Prepare drug_mechanism_of_action: explode chemblIds to get drugId, keep targets
    drug_moa_exploded = drug_moa_df.select(f.explode('chemblIds').alias('drugId'), 'targets')

    # Join and explode targets
    evidence = (
        exploded_cr.join(drug_moa_exploded, on='drugId', how='inner')
        .withColumn('targetFromSourceId', f.explode_outer('targets'))
        .filter(f.col('targetFromSourceId').isNotNull())
        .select(
            'clinicalReportId',
            'clinicalStage',
            'trialWhyStopped',
            'trialStopReasonCategories',
            'studyStartDate',
            'literature',
            'diseaseFromSource',
            'diseaseFromSourceMappedId',
            'drugFromSource',
            'drugId',
            'targetFromSourceId',
            f.lit('clinical_precedence').alias('datasourceId'),
            f.lit('clinical').alias('datatypeId'),
        )
    )

    logger.info(f'Writing clinical precedence evidence to {destination}')
    evidence.write.parquet(destination, mode='overwrite')

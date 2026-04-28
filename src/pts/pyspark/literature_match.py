"""Extract matches and map labels."""

from typing import Any

from literature.dataset.publication import Publication
from loguru import logger
from pyspark.sql import functions as f

from pts.pyspark.common.session import Session


def literature_match(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    spark = Session(app_name='literature', properties=properties)

    logger.info(f'load publications from: {source['publication']}')
    publication = spark.load_data(path=source['publication'])

    logger.info('extract matches and map labels')
    match_mapped = (
        Publication(publication)
        .extract_matches()
        .map_labels(
            session=spark,
            label_lut_path=source['ontoma_disease_target_drug_label_lut'],
            label_col_name='label',
            type_col_name='type'
        )
    )

    logger.info('disambiguate')
    match_disambiguated = (
        match_mapped
        .disambiguate(
            trusted_sources=[
                'name',
                'ot_curation',
                'eva_clinvar',
                'clinvar_xrefs',
                'approved_name',
                'approved_symbol'
            ]
        )
    )

    match_valid = (
        match_disambiguated.df
        .filter(f.col('isValid') == True)
    )

    match_failed = (
        match_mapped.df
        .filter(f.col('isMapped') == False)
        .unionByName(
            match_disambiguated.df.filter(f.col('isValid') == False),
            allowMissingColumns=True
        )
    )

    logger.info(f'write valid matches to {destination['match_valid']}')
    (
        match_valid
        .write.mode('overwrite')
        .parquet(destination['match_valid'])
    )

    logger.info(f'write failed matches to {destination['match_failed']}')
    (
        match_failed
        .write.mode('overwrite')
        .parquet(destination['match_failed'])
    )

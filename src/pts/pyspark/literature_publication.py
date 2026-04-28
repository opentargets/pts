"""Process publications from EPMC."""

from typing import Any

from literature.datasource.epmc.publication import EPMCPublication
from literature.datasource.epmc.publication_id_lut import PublicationIdLUT
from loguru import logger

from pts.pyspark.common.session import Session


def literature_publication(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    spark = Session(app_name='literature', properties=properties)

    logger.info(f'load data from: {source['pub_id_lut']}')
    pub_id_lut = PublicationIdLUT.from_csv(spark, source['pub_id_lut']).persist()

    logger.info(f'load data from: {source['epmc_publication']}')
    publication = EPMCPublication.from_source(spark, source['epmc_publication'], pub_id_lut)

    logger.info(f'write processed publications to {destination['publication']}')
    (
        publication.df
        .write.mode('overwrite')
        .parquet(destination['publication'])
    )

"""Generate cooccurrences."""

from typing import Any

from literature.dataset.match_mapped import MatchMapped
from loguru import logger

from pts.pyspark.common.session import Session


def literature_cooccurrence(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    spark = Session(app_name='literature', properties=properties)

    logger.info(f'load matches from: {source['match']}')
    match = spark.load_data(path=source['match'])

    logger.info(f'write cooccurrences to {destination['cooccurrence']}')
    (
        MatchMapped(match)
        .generate_target_disease_cooccurrences()
        .df
        .write.mode('overwrite')
        .parquet(destination['cooccurrence'])
    )

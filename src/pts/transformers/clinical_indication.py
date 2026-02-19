"""Clinical indication dataset generation."""

from pathlib import Path

import polars as pl
from clinical_mining.dataset import ClinicalIndication
from loguru import logger


def clinical_indication(source: Path, destination: Path) -> None:
    """Generate clinical indication dataset from clinical report data.

    Args:
        source: Path to clinical report data (output/clinical_report)
        destination: Path to write clinical indication data (output/clinical_indication)
    """
    logger.info(f'Source path: {source}')
    reports = pl.read_parquet(source)
    indications = (
        ClinicalIndication.from_report(reports)
        .df.filter(pl.col('mappingStatus') == 'FULLY_MAPPED')
        .drop('drugName', 'diseaseName', 'mappingStatus')
    )

    logger.info(f'Destination path: {destination}')
    indications.write_parquet(destination, mkdir=True)

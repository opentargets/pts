"""Clinical target dataset generation."""

from pathlib import Path

import polars as pl
import polars_hash as plh
from clinical_mining.dataset.clinical_indication import (
    CATEGORY_RANKS,
    CATEGORY_RANKS_STR,
    RANK_TO_CATEGORY_STR,
    ClinicalStageCategory,
)
from loguru import logger


def clinical_target(source: dict[str, Path], destination: Path) -> None:
    """Generate clinical target dataset from clinical report and drug mechanism of action data.

    Args:
        source: Dictionary containing paths to input data:
            - clinical_report: Path to clinical report data (output/clinical_report)
            - drug_mechanism_of_action: Path to drug mechanism of action data (output/drug_mechanism_of_action)
        destination: Path to write clinical target data (output/clinical_target)
    """
    logger.info(f'Source paths: {source}')
    reports = pl.read_parquet(source['clinical_report'])
    # moa = pl.read_parquet(source['drug_mechanism_of_action'])
    moa = pl.read_parquet('/Users/irenelopez/EBI/repos/pts/work/output/drug_mechanism_of_action')

    drug_max_stage = (
        # TODO: bring this from drug molecule AND treat phase iv/withdrawn as approval
        reports.explode('drugs')
        .unnest('drugs')
        .filter(pl.col('drugId').is_not_null())
        .with_columns(
            clinicalStageRank=pl.col('clinicalStage').replace_strict(
                CATEGORY_RANKS_STR,
                default=CATEGORY_RANKS[ClinicalStageCategory.UNKNOWN],
            )
        )
        .group_by('drugId')
        .agg(pl.col('clinicalStageRank').min().alias('drugMaxClinicalStageRank'))
        .with_columns(
            # Map rank back to stage string
            maxClinicalStage=pl.col('drugMaxClinicalStageRank').replace_strict(RANK_TO_CATEGORY_STR)
        )
        .drop('drugMaxClinicalStageRank')
    )
    moa_lut = (
        moa.explode('targets')
        .explode('chemblIds')
        .select(pl.col('chemblIds').alias('drugId'), pl.col('targets').alias('targetId'))
        .filter(pl.col('drugId').is_not_null())
        .filter(pl.col('targetId').is_not_null())
        .unique()
    )
    clinical_target = (
        reports.explode('drugs')
        .explode('diseases')
        .unnest('drugs')
        .filter(pl.col('drugId').is_not_null())
        .rename({'id': 'reportId'})
        .join(moa_lut, 'drugId')
        .with_columns(
            id=plh.concat_str('drugId', 'targetId').chash.sha2_256(),
            clinicalStageRank=pl.col('clinicalStage').replace_strict(
                CATEGORY_RANKS_STR,
                default=CATEGORY_RANKS[ClinicalStageCategory.UNKNOWN],
            ),
        )
        .sort('clinicalStageRank')
        .group_by(['id', 'drugId', 'targetId'], maintain_order=True)
        .agg(
            pl.col('diseases').unique().alias('diseases'),
            pl.col('reportId').unique().alias('clinicalReportIds'),
        )
        .join(drug_max_stage, 'drugId')
    )
    logger.info(f'Destination path: {destination}')
    clinical_target.write_parquet(destination, mkdir=True)

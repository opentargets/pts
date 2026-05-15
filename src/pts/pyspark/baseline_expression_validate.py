"""Validate merged baseline-expression rows against target and biosample indexes."""

from __future__ import annotations

from typing import Any

from loguru import logger

from pts.pyspark.common.session import Session
from pts.pyspark.expression_utils.expression import (
    split_valid_invalid,
    validate_biosample,
    validate_target,
)
from pts.pyspark.expression_utils.validation_lut import (
    prepare_biosample_lut,
    prepare_target_lut,
)


def baseline_expression_validate(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    logger.info('Starting baseline expression validation')
    session = Session(app_name='baseline_expression_validate', properties=properties)
    try:
        merged = session.load_data(source['merged'], format='parquet')
        target_lut = prepare_target_lut(session.load_data(source['target_path'], format='parquet'))
        biosample_lut = prepare_biosample_lut(session.load_data(source['biosample_path'], format='parquet'))

        validated = validate_target(merged, target_lut, settings.get('excluded_biotypes'))
        validated = validate_biosample(validated, biosample_lut, 'tissueBiosampleId', 'tissueBiosampleFromSource')
        validated = validate_biosample(validated, biosample_lut, 'celltypeBiosampleId', 'celltypeBiosampleFromSource')

        valid, invalid = split_valid_invalid(validated)
        valid.write.mode('overwrite').parquet(destination['valid'])
        invalid.write.mode('overwrite').parquet(destination['failed'])
    finally:
        session.stop()
    logger.info('Baseline expression validation completed successfully')

"""Validation task for PTS pipeline.

This module provides a task interface for JSON schema validation
that can be integrated into the PTS pipeline.
"""

from pathlib import Path

from loguru import logger

from pts.validation.schema_validator import validate_json_schema, validate_json_schema_batch


def validation_task(
    source: dict[str, str],
    destination: dict[str, str],
    properties: dict[str, str] | None = None,
) -> None:
    """Task function for JSON schema validation.

    This function can be called from the PTS pipeline to validate
    JSON data against schemas.

    Args:
        source: Dictionary containing:
            - schema: Path to JSON schema file
            - data: Path to JSON data file (single file)
            - data_dir: Path to directory containing JSON files (batch validation)
        destination: Dictionary containing:
            - output: Path to save validation results
        properties: Optional properties for the task
    """
    logger.info('Starting JSON schema validation task')

    # Get schema path
    schema_path = source.get('schema')
    if not schema_path:
        raise ValueError('Schema path not specified in source')

    # Determine if single file or batch validation
    data_path = source.get('data')
    data_dir = source.get('data_dir')

    if data_path and data_dir:
        raise ValueError("Cannot specify both 'data' and 'data_dir' in source")

    if not data_path and not data_dir:
        raise ValueError("Must specify either 'data' or 'data_dir' in source")

    # Get output path
    output_path = destination.get('output')
    if not output_path:
        raise ValueError('Output path not specified in destination')

    # Perform validation
    if data_path:
        # Single file validation
        logger.info(f'Validating single file: {data_path}')
        result = validate_json_schema(schema_path, data_path, output_path)

        if result.get('valid', False):
            logger.info('Validation completed successfully')
        else:
            logger.error(f'Validation failed: {result.get("errors", [])}')
            raise ValueError(f'Schema validation failed: {result.get("errors", [])}')

    else:
        # Batch validation
        logger.info(f'Validating files in directory: {data_dir}')
        data_dir = Path(data_dir)

        if not data_dir.exists():
            raise FileNotFoundError(f'Data directory not found: {data_dir}')

        # Find all JSON files in the directory
        json_files = list(data_dir.glob('*.json'))
        if not json_files:
            logger.warning(f'No JSON files found in {data_dir}')
            return

        logger.info(f'Found {len(json_files)} JSON files to validate')

        # Create output directory
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Perform batch validation
        results = validate_json_schema_batch(schema_path, json_files, output_dir)

        # Check overall results
        failed_validations = [path for path, result in results.items() if not result.get('valid', False)]

        if failed_validations:
            logger.error(f'Validation failed for {len(failed_validations)} files: {failed_validations}')
            raise ValueError(f'Schema validation failed for files: {failed_validations}')
        else:
            logger.info('All validations completed successfully')

    logger.info('JSON schema validation task completed')

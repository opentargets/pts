"""JSON Schema validation functionality for PTS.

This module provides functions to validate JSON data against schemas
using the opentargets-validator package.
"""

import json
import subprocess
from pathlib import Path
from typing import Any

from loguru import logger


def validate_json_schema(
    schema_path: str | Path,
    data_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate JSON data against a schema using opentargets-validator.

    Args:
        schema_path: Path to the JSON schema file or URL
        data_path: Path to the JSON data file to validate
        output_path: Optional path to save validation results

    Returns:
        Dictionary containing validation results

    Raises:
        FileNotFoundError: If schema or data files don't exist
        ValueError: If validation fails
        subprocess.CalledProcessError: If opentargets-validator command fails
    """
    schema_path = Path(schema_path) if not str(schema_path).startswith(('http://', 'https://')) else str(schema_path)
    data_path = Path(data_path)

    # Check if data file exists (schema can be URL)
    if not str(schema_path).startswith(('http://', 'https://')) and not Path(schema_path).exists():
        raise FileNotFoundError(f'Schema file not found: {schema_path}')

    if not data_path.exists():
        raise FileNotFoundError(f'Data file not found: {data_path}')

    logger.info(f'Validating {data_path} against schema {schema_path}')

    try:
        # Use opentargets-validator command-line tool
        cmd = ['opentargets_validator', '--schema', str(schema_path), str(data_path)]

        logger.debug(f'Running command: {" ".join(cmd)}')

        # Run the validator
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Parse the output
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'message': 'Validation completed successfully',
            'stdout': result.stdout,
            'stderr': result.stderr,
        }

        logger.info('Validation completed successfully using opentargets-validator')

    except subprocess.CalledProcessError as e:
        # Validation failed
        validation_result = {
            'valid': False,
            'errors': [e.stderr] if e.stderr else ['Validation failed'],
            'warnings': [],
            'message': f'Validation failed: {e.stderr}',
            'stdout': e.stdout,
            'stderr': e.stderr,
            'return_code': e.returncode,
        }

        logger.error(f'Validation failed: {e.stderr}')

    except FileNotFoundError:
        # opentargets-validator not found, use fallback
        logger.warning('opentargets-validator not found, using fallback validation')
        validation_result = _fallback_validation(schema_path, data_path)

    # Save results if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(validation_result, f, indent=2)
        logger.info(f'Validation results saved to {output_path}')

    return validation_result


def _fallback_validation(schema_path: str | Path, data_path: Path) -> dict[str, Any]:
    """Fallback validation when opentargets-validator is not available.

    Args:
        schema_path: Path to JSON schema file or URL
        data_path: Path to data file to validate

    Returns:
        Basic validation result
    """
    logger.warning('Using fallback validation - install opentargets-validator for full functionality')

    # Basic validation - check if data is valid JSON
    result = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'message': 'Basic validation completed (opentargets-validator not available)',
    }

    try:
        # Load and validate JSON data
        with open(data_path) as f:
            data = json.load(f)

        # Load schema if it's a local file
        _validate_required_fields(schema_path, data, result)

        result['warnings'].append('Full schema validation not available - install opentargets-validator')

    except json.JSONDecodeError as e:
        result['valid'] = False
        result['errors'].append(f'Invalid JSON in data file: {e}')
    except Exception as e:
        result['valid'] = False
        result['errors'].append(f'Error during validation: {e}')

    return result


def _validate_required_fields(schema_path: str | Path, data: Any, result: dict[str, Any]) -> None:
    """Validate required fields from schema.

    Args:
        schema_path: Path to schema file
        data: Data to validate
        result: Result dictionary to update
    """
    if str(schema_path).startswith(('http://', 'https://')):
        return

    schema_path = Path(schema_path)
    if not schema_path.exists():
        return

    with open(schema_path) as f:
        schema = json.load(f)

    if 'required' in schema and isinstance(data, dict):
        missing_fields = [field for field in schema['required'] if field not in data]

        if missing_fields:
            result['valid'] = False
            result['errors'].append(f'Missing required fields: {missing_fields}')


def validate_json_schema_batch(
    schema_path: str | Path,
    data_files: list[str | Path],
    output_dir: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Validate multiple JSON files against a schema.

    Args:
        schema_path: Path to the JSON schema file
        data_files: List of paths to JSON data files to validate
        output_dir: Optional directory to save individual validation results

    Returns:
        Dictionary mapping file paths to validation results
    """
    results = {}

    for data_file in data_files:
        try:
            output_path = None
            if output_dir:
                output_dir = Path(output_dir)
                output_path = output_dir / f'{Path(data_file).stem}_validation.json'

            result = validate_json_schema(schema_path, data_file, output_path)
            results[str(data_file)] = result

        except Exception as e:
            logger.error(f'Validation failed for {data_file}: {e}')
            results[str(data_file)] = {
                'valid': False,
                'errors': [str(e)],
                'warnings': [],
                'message': f'Validation failed: {e}',
            }

    return results

"""Tests for the JSON schema validation module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from pts.validation.schema_validator import (
    _fallback_validation,
    _validate_required_fields,
    validate_json_schema,
    validate_json_schema_batch,
)


class TestValidateJsonSchema:
    """Test the main validate_json_schema function."""

    def test_validate_json_schema_success(self):
        """Test successful validation with valid data."""
        schema = {
            'type': 'object',
            'properties': {
                'name': {'type': 'string'},
                'age': {'type': 'number'},
            },
            'required': ['name', 'age'],
        }

        data = {'name': 'John Doe', 'age': 30}

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            schema_file = temp_path / 'schema.json'
            data_file = temp_path / 'data.json'

            with open(schema_file, 'w') as f:
                json.dump(schema, f)

            with open(data_file, 'w') as f:
                json.dump(data, f)

            result = validate_json_schema(schema_file, data_file)

            assert result['valid'] is True
            assert result['errors'] == []
            assert 'Validation completed successfully' in result['message']

    def test_validate_json_schema_with_output_path(self):
        """Test validation with output path specified."""
        schema = {'type': 'object', 'properties': {'name': {'type': 'string'}}}
        data = {'name': 'John Doe'}

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            schema_file = temp_path / 'schema.json'
            data_file = temp_path / 'data.json'
            output_file = temp_path / 'result.json'

            with open(schema_file, 'w') as f:
                json.dump(schema, f)

            with open(data_file, 'w') as f:
                json.dump(data, f)

            result = validate_json_schema(schema_file, data_file, output_file)

            assert result['valid'] is True
            assert output_file.exists()

            # Check that output file contains the result
            with open(output_file) as f:
                saved_result = json.load(f)
            assert saved_result['valid'] is True

    def test_validate_json_schema_schema_not_found(self):
        """Test validation when schema file doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            schema_file = temp_path / 'nonexistent_schema.json'
            data_file = temp_path / 'data.json'

            with open(data_file, 'w') as f:
                json.dump({'name': 'John'}, f)

            with pytest.raises(FileNotFoundError, match='Schema file not found'):
                validate_json_schema(schema_file, data_file)

    def test_validate_json_schema_data_not_found(self):
        """Test validation when data file doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            schema_file = temp_path / 'schema.json'
            data_file = temp_path / 'nonexistent_data.json'

            with open(schema_file, 'w') as f:
                json.dump({'type': 'object'}, f)

            with pytest.raises(FileNotFoundError, match='Data file not found'):
                validate_json_schema(schema_file, data_file)

    @patch('subprocess.run')
    def test_validate_json_schema_command_not_found(self, mock_run):
        """Test fallback validation when opentargets-validator is not found."""
        mock_run.side_effect = FileNotFoundError()

        schema = {'type': 'object', 'required': ['name']}
        data = {'name': 'John Doe'}

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            schema_file = temp_path / 'schema.json'
            data_file = temp_path / 'data.json'

            with open(schema_file, 'w') as f:
                json.dump(schema, f)

            with open(data_file, 'w') as f:
                json.dump(data, f)

            result = validate_json_schema(schema_file, data_file)

            assert result['valid'] is True
            assert 'basic validation completed' in result['message'].lower()

    @patch('subprocess.run')
    def test_validate_json_schema_validation_failure(self, mock_run):
        """Test validation failure handling."""
        from subprocess import CalledProcessError

        mock_run.side_effect = CalledProcessError(
            returncode=1,
            cmd=['opentargets_validator'],
            stderr='Validation failed: missing required field',
        )

        schema = {'type': 'object', 'required': ['name']}
        data = {'age': 30}  # Missing required 'name' field

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            schema_file = temp_path / 'schema.json'
            data_file = temp_path / 'data.json'

            with open(schema_file, 'w') as f:
                json.dump(schema, f)

            with open(data_file, 'w') as f:
                json.dump(data, f)

            result = validate_json_schema(schema_file, data_file)

            assert result['valid'] is False
            assert 'Validation failed' in result['message']
            assert result['return_code'] == 1


class TestValidateJsonSchemaBatch:
    """Test the batch validation function."""

    def test_validate_json_schema_batch_success(self):
        """Test successful batch validation."""
        schema = {'type': 'object', 'properties': {'name': {'type': 'string'}}}

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            schema_file = temp_path / 'schema.json'
            output_dir = temp_path / 'output'

            with open(schema_file, 'w') as f:
                json.dump(schema, f)

            # Create multiple data files
            data_files = []
            for i in range(3):
                data_file = temp_path / f'data_{i}.json'
                with open(data_file, 'w') as f:
                    json.dump({'name': f'Person {i}'}, f)
                data_files.append(data_file)

            results = validate_json_schema_batch(schema_file, data_files, output_dir)

            assert len(results) == 3
            for result in results.values():
                assert result['valid'] is True

            # Check that output files were created
            assert output_dir.exists()
            output_files = list(output_dir.glob('*.json'))
            assert len(output_files) == 3

    def test_validate_json_schema_batch_with_failures(self):
        """Test batch validation with some failures."""
        schema = {'type': 'object', 'required': ['name']}

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            schema_file = temp_path / 'schema.json'

            with open(schema_file, 'w') as f:
                json.dump(schema, f)

            # Create one valid and one invalid file
            valid_file = temp_path / 'valid.json'
            invalid_file = temp_path / 'invalid.json'

            with open(valid_file, 'w') as f:
                json.dump({'name': 'John'}, f)

            with open(invalid_file, 'w') as f:
                json.dump({'age': 30}, f)  # Missing required 'name'

            data_files = [valid_file, invalid_file]

            # Test with fallback validation by mocking subprocess to fail
            with patch('subprocess.run') as mock_run:
                mock_run.side_effect = FileNotFoundError()

                results = validate_json_schema_batch(schema_file, [str(f) for f in data_files])

                assert len(results) == 2
                # Valid file should pass, invalid file should fail due to missing required field
                assert results[str(valid_file)]['valid'] is True
                assert results[str(invalid_file)]['valid'] is False
                assert 'Missing required fields' in results[str(invalid_file)]['errors'][0]


class TestFallbackValidation:
    """Test the fallback validation function."""

    def test_fallback_validation_success(self):
        """Test successful fallback validation."""
        schema = {'type': 'object', 'required': ['name']}
        data = {'name': 'John Doe'}

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            schema_file = temp_path / 'schema.json'
            data_file = temp_path / 'data.json'

            with open(schema_file, 'w') as f:
                json.dump(schema, f)

            with open(data_file, 'w') as f:
                json.dump(data, f)

            result = _fallback_validation(schema_file, data_file)

            assert result['valid'] is True
            assert result['errors'] == []
            assert 'basic validation completed' in result['message'].lower()

    def test_fallback_validation_missing_required_fields(self):
        """Test fallback validation with missing required fields."""
        schema = {'type': 'object', 'required': ['name', 'age']}
        data = {'name': 'John Doe'}  # Missing 'age'

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            schema_file = temp_path / 'schema.json'
            data_file = temp_path / 'data.json'

            with open(schema_file, 'w') as f:
                json.dump(schema, f)

            with open(data_file, 'w') as f:
                json.dump(data, f)

            result = _fallback_validation(schema_file, data_file)

            assert result['valid'] is False
            assert len(result['errors']) == 1
            assert 'Missing required fields' in result['errors'][0]
            assert 'age' in result['errors'][0]

    def test_fallback_validation_invalid_json(self):
        """Test fallback validation with invalid JSON."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            schema_file = temp_path / 'schema.json'
            data_file = temp_path / 'data.json'

            with open(schema_file, 'w') as f:
                json.dump({'type': 'object'}, f)

            data_file.write_text('invalid json content')

            result = _fallback_validation(schema_file, data_file)

            assert result['valid'] is False
            assert len(result['errors']) == 1
            assert 'Invalid JSON' in result['errors'][0]

    def test_fallback_validation_url_schema(self):
        """Test fallback validation with URL schema."""
        data = {'name': 'John Doe'}

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            data_file = temp_path / 'data.json'

            with open(data_file, 'w') as f:
                json.dump(data, f)

            result = _fallback_validation('https://example.com/schema.json', data_file)

            assert result['valid'] is True
            assert result['errors'] == []


class TestValidateRequiredFields:
    """Test the _validate_required_fields helper function."""

    def test_validate_required_fields_success(self):
        """Test successful required fields validation."""
        schema = {'required': ['name', 'age']}
        data = {'name': 'John', 'age': 30}
        result = {'valid': True, 'errors': []}

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            schema_file = temp_path / 'schema.json'

            with open(schema_file, 'w') as f:
                json.dump(schema, f)

            _validate_required_fields(schema_file, data, result)

            assert result['valid'] is True
            assert result['errors'] == []

    def test_validate_required_fields_missing_fields(self):
        """Test required fields validation with missing fields."""
        schema = {'required': ['name', 'age', 'email']}
        data = {'name': 'John'}  # Missing 'age' and 'email'
        result = {'valid': True, 'errors': []}

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            schema_file = temp_path / 'schema.json'

            with open(schema_file, 'w') as f:
                json.dump(schema, f)

            _validate_required_fields(schema_file, data, result)

            assert result['valid'] is False
            assert len(result['errors']) == 1
            assert 'Missing required fields' in result['errors'][0]
            assert 'age' in result['errors'][0]
            assert 'email' in result['errors'][0]

    def test_validate_required_fields_no_required_section(self):
        """Test validation when schema has no required section."""
        schema = {'type': 'object'}
        data = {'name': 'John'}
        result = {'valid': True, 'errors': []}

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            schema_file = temp_path / 'schema.json'

            with open(schema_file, 'w') as f:
                json.dump(schema, f)

            _validate_required_fields(schema_file, data, result)

            assert result['valid'] is True
            assert result['errors'] == []

    def test_validate_required_fields_non_dict_data(self):
        """Test validation with non-dictionary data."""
        schema = {'required': ['name']}
        data = 'not a dictionary'
        result = {'valid': True, 'errors': []}

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            schema_file = temp_path / 'schema.json'

            with open(schema_file, 'w') as f:
                json.dump(schema, f)

            _validate_required_fields(schema_file, data, result)

            assert result['valid'] is True
            assert result['errors'] == []

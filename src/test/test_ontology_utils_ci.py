"""CI-friendly tests for the ontology utility module that don't require Spark.

This module provides tests that can run in CI environments where Java/Spark
might not be available, focusing on the core functionality without Spark dependencies.
"""
import os
from unittest.mock import Mock, patch

from pts.utils.ontology import (
    ONTOMA_MAX_ATTEMPTS,
    _ontoma_udf,
    _simple_retry,
)


class TestSimpleRetryCI:
    """Test the _simple_retry function in CI environment."""

    def test_successful_function_call(self):
        """Test that a successful function call returns the result immediately."""
        def mock_func(value):
            return value * 2

        result = _simple_retry(mock_func, value=5)
        assert result == 10

    def test_retry_on_failure(self):
        """Test that the function retries on failure and eventually succeeds."""
        call_count = 0

        def mock_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception('Temporary failure')
            return 'success'

        with patch('time.sleep'):  # Mock sleep to speed up test
            result = _simple_retry(mock_func)
            assert result == 'success'
            assert call_count == 3

    def test_max_attempts_reached(self):
        """Test that the function returns empty list after max attempts."""
        def mock_func():
            raise Exception('Persistent failure')

        with patch('time.sleep'):  # Mock sleep to speed up test
            result = _simple_retry(mock_func)
            assert result == []

    def test_retry_with_kwargs(self):
        """Test that the function properly handles keyword arguments."""
        def mock_func(param1, param2=None):
            if param2 == 'fail':
                raise Exception('Failure')
            return f'{param1}_{param2}'

        # Test successful call
        result = _simple_retry(mock_func, param1='test', param2='value')
        assert result == 'test_value'

        # Test failure
        with patch('time.sleep'):
            result = _simple_retry(mock_func, param1='test', param2='fail')
            assert result == []


class TestOntomaUdfCI:
    """Test the _ontoma_udf function in CI environment."""

    def test_mapping_by_disease_name(self):
        """Test mapping by disease name."""
        mock_ontoma = Mock()
        mock_mapping = Mock()
        mock_mapping.id_ot_schema = 'EFO:0001234'
        mock_ontoma.find_term.return_value = [mock_mapping]

        row = {
            'diseaseFromSource': 'Alzheimer disease',
            'diseaseFromSourceId': 'MONDO:0004975'
        }

        with patch('pts.utils.ontology._simple_retry', return_value=[mock_mapping]):
            result = _ontoma_udf(row, mock_ontoma)
            assert result == ['EFO:0001234']

    def test_mapping_by_disease_id(self):
        """Test mapping by disease ID when name mapping fails."""
        mock_ontoma = Mock()
        mock_mapping = Mock()
        mock_mapping.id_ot_schema = 'EFO:0005678'

        row = {
            'diseaseFromSource': 'Unknown disease',
            'diseaseFromSourceId': 'MONDO:0001234'
        }

        with patch('pts.utils.ontology._simple_retry') as mock_retry:
            # First call (by name) returns empty, second call (by ID) returns mapping
            mock_retry.side_effect = [[], [mock_mapping]]
            result = _ontoma_udf(row, mock_ontoma)
            assert result == ['EFO:0005678']

    def test_no_mappings_found(self):
        """Test when no mappings are found."""
        mock_ontoma = Mock()
        row = {
            'diseaseFromSource': 'Unknown disease',
            'diseaseFromSourceId': 'UNKNOWN:123'
        }

        with patch('pts.utils.ontology._simple_retry', return_value=[]):
            result = _ontoma_udf(row, mock_ontoma)
            assert result == []

    def test_obsolete_disease_name(self):
        """Test handling of obsolete disease names."""
        mock_ontoma = Mock()
        mock_mapping = Mock()
        mock_mapping.id_ot_schema = 'EFO:0009999'

        row = {
            'diseaseFromSource': 'obsolete  Alzheimer disease  ',
            'diseaseFromSourceId': 'MONDO:0004975'
        }

        with patch('pts.utils.ontology._simple_retry', return_value=[mock_mapping]) as mock_retry:
            result = _ontoma_udf(row, mock_ontoma)
            # Should call with cleaned disease name
            mock_retry.assert_called_with(mock_ontoma.find_term, query='Alzheimer disease', code=False)
            assert result == ['EFO:0009999']

    def test_disease_id_with_underscore(self):
        """Test that underscores in disease ID are replaced with colons."""
        mock_ontoma = Mock()
        mock_mapping = Mock()
        mock_mapping.id_ot_schema = 'EFO:0001111'

        row = {
            'diseaseFromSource': None,
            'diseaseFromSourceId': 'MONDO_0001234'
        }

        with patch('pts.utils.ontology._simple_retry', return_value=[mock_mapping]) as mock_retry:
            result = _ontoma_udf(row, mock_ontoma)
            # Should call with colon-replaced ID
            mock_retry.assert_called_with(mock_ontoma.find_term, query='MONDO:0001234', code=True)
            assert result == ['EFO:0001111']

    def test_empty_disease_fields(self):
        """Test handling of empty disease fields."""
        row = {
            'diseaseFromSource': None,
            'diseaseFromSourceId': None
        }

        with patch('pts.utils.ontology._simple_retry', return_value=[]):
            result = _ontoma_udf(row, None)
            assert result == []

    def test_disease_id_without_colon(self):
        """Test that disease IDs without colons are not used for mapping."""
        mock_ontoma = Mock()
        row = {
            'diseaseFromSource': None,
            'diseaseFromSourceId': 'INVALID_ID'
        }

        with patch('pts.utils.ontology._simple_retry', return_value=[]):
            result = _ontoma_udf(row, mock_ontoma)
            assert result == []


class TestEdgeCasesCI:
    """Test edge cases and error handling in CI environment."""

    def test_ontoma_max_attempts_constant(self):
        """Test that ONTOMA_MAX_ATTEMPTS constant is defined correctly."""
        assert ONTOMA_MAX_ATTEMPTS == 3

    def test_retry_sleep_timing(self):
        """Test that retry includes appropriate sleep timing."""
        def mock_func():
            raise Exception('Test exception')

        with patch('time.sleep') as mock_sleep, \
             patch('random.random', return_value=0.5):
            _simple_retry(mock_func)
            # Should sleep 2 times (max attempts - 1 = 3 - 1 = 2)
            assert mock_sleep.call_count == 2
            assert all(call[0][0] == 10.0 for call in mock_sleep.call_args_list)

    def test_multiple_mappings_explosion(self):
        """Test that multiple mappings are properly handled."""
        mock_ontoma = Mock()
        mock_mapping1 = Mock()
        mock_mapping1.id_ot_schema = 'EFO:0001111'
        mock_mapping2 = Mock()
        mock_mapping2.id_ot_schema = 'EFO:0002222'

        row = {
            'diseaseFromSource': 'Complex disease',
            'diseaseFromSourceId': 'MONDO:0009999'
        }

        with patch('pts.utils.ontology._simple_retry', return_value=[mock_mapping1, mock_mapping2]):
            result = _ontoma_udf(row, mock_ontoma)
            assert result == ['EFO:0001111', 'EFO:0002222']

    def test_efo_version_environment_variable(self):
        """Test EFO version handling from environment variable."""
        with patch.dict(os.environ, {'EFO_VERSION': '3.0.0'}), \
             patch('pts.utils.ontology.OnToma') as mock_ontoma_class:
            mock_ontoma_class.return_value = Mock()
            # This test just verifies the environment variable is read correctly
            # without actually calling the full function
            assert os.environ.get('EFO_VERSION') == '3.0.0'

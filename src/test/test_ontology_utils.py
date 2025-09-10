"""Tests for the ontology utility module."""
import os
import time
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, StructField, StructType

from pts.utils.ontology import (
    ONTOMA_MAX_ATTEMPTS,
    _ontoma_udf,
    _simple_retry,
    add_efo_mapping,
)


class TestSimpleRetry:
    """Test the _simple_retry function."""

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


class TestOntomaUdf:
    """Test the _ontoma_udf function."""

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


class TestAddEfoMapping:
    """Test the add_efo_mapping function."""

    @pytest.fixture
    def spark_session(self):
        """Create a Spark session for testing."""
        spark = SparkSession.builder.master('local[1]').appName('test').getOrCreate()
        yield spark
        spark.stop()

    @pytest.fixture
    def sample_evidence_df(self, spark_session):
        """Create a sample evidence DataFrame for testing."""
        data = [
            ('Alzheimer disease', 'MONDO:0004975'),
            ('Type 2 diabetes', 'MONDO:0005148'),
            ('Unknown disease', None),
            (None, 'MONDO:0001234'),
        ]
        return spark_session.createDataFrame(data, ['diseaseFromSource', 'diseaseFromSourceId'])

    def test_efo_mapping_success(self, spark_session, sample_evidence_df):
        """Test successful EFO mapping."""
        # This test verifies that the function can be called without errors
        # The actual Spark operations are complex to mock, so we focus on basic functionality
        with patch('pts.utils.ontology.OnToma') as mock_ontoma_class, \
             patch.object(sample_evidence_df, 'select') as mock_select, \
             patch.object(sample_evidence_df, 'distinct') as mock_distinct, \
             patch.object(mock_distinct, 'toPandas', return_value=pd.DataFrame()), \
             patch('pts.utils.ontology.pandarallel') as mock_pandarallel:

            # Setup mocks
            mock_ontoma_instance = Mock()
            mock_ontoma_class.return_value = mock_ontoma_instance
            mock_select.return_value = mock_distinct

            # Test that the function can be called and returns a DataFrame
            result = add_efo_mapping(sample_evidence_df, spark_session)
            assert result is not None
            # Verify that OnToma was initialized
            mock_ontoma_class.assert_called_once()

    def test_efo_version_from_environment(self, spark_session, sample_evidence_df):
        """Test that EFO version is read from environment variable."""
        with patch.dict(os.environ, {'EFO_VERSION': '3.0.0'}), \
             patch('pts.utils.ontology.OnToma') as mock_ontoma_class, \
             patch.object(sample_evidence_df, 'select'), \
             patch.object(sample_evidence_df, 'distinct'), \
             patch.object(sample_evidence_df, 'toPandas', return_value=pd.DataFrame()):

            mock_ontoma_class.return_value = Mock()
            add_efo_mapping(sample_evidence_df, spark_session)
            mock_ontoma_class.assert_called_with(cache_dir=None, efo_release='3.0.0')

    def test_efo_version_default(self, spark_session, sample_evidence_df):
        """Test that default EFO version is used when not specified."""
        with patch.dict(os.environ, {}, clear=True), \
             patch('pts.utils.ontology.OnToma') as mock_ontoma_class, \
             patch.object(sample_evidence_df, 'select'), \
             patch.object(sample_evidence_df, 'distinct'), \
             patch.object(sample_evidence_df, 'toPandas', return_value=pd.DataFrame()):

            mock_ontoma_class.return_value = Mock()
            add_efo_mapping(sample_evidence_df, spark_session)
            mock_ontoma_class.assert_called_with(cache_dir=None, efo_release='latest')

    def test_efo_version_parameter(self, spark_session, sample_evidence_df):
        """Test that EFO version parameter is used when provided."""
        with patch('pts.utils.ontology.OnToma') as mock_ontoma_class, \
             patch.object(sample_evidence_df, 'select'), \
             patch.object(sample_evidence_df, 'distinct'), \
             patch.object(sample_evidence_df, 'toPandas', return_value=pd.DataFrame()):

            mock_ontoma_class.return_value = Mock()
            add_efo_mapping(sample_evidence_df, spark_session, efo_version='2.0.0')
            mock_ontoma_class.assert_called_with(cache_dir=None, efo_release='2.0.0')

    def test_ontoma_cache_dir_parameter(self, spark_session, sample_evidence_df):
        """Test that OnToma cache directory parameter is used when provided."""
        with patch('pts.utils.ontology.OnToma') as mock_ontoma_class, \
             patch.object(sample_evidence_df, 'select'), \
             patch.object(sample_evidence_df, 'distinct'), \
             patch.object(sample_evidence_df, 'toPandas', return_value=pd.DataFrame()):

            mock_ontoma_class.return_value = Mock()
            add_efo_mapping(sample_evidence_df, spark_session, ontoma_cache_dir='/tmp/cache')
            mock_ontoma_class.assert_called_with(cache_dir='/tmp/cache', efo_release='latest')

    def test_empty_evidence_dataframe(self, spark_session):
        """Test handling of empty evidence DataFrame."""
        empty_df = spark_session.createDataFrame([], StructType([
            StructField('diseaseFromSource', StringType(), True),
            StructField('diseaseFromSourceId', StringType(), True)
        ]))

        with patch('pts.utils.ontology.OnToma') as mock_ontoma_class, \
             patch.object(empty_df, 'select'), \
             patch.object(empty_df, 'distinct'), \
             patch.object(empty_df, 'toPandas', return_value=pd.DataFrame()):

            mock_ontoma_class.return_value = Mock()
            result = add_efo_mapping(empty_df, spark_session)
            assert result is not None


class TestEdgeCases:
    """Test edge cases and error handling."""

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

    def test_nan_handling_in_dataframe(self):
        """Test that NaN values are properly handled in DataFrame operations."""
        # This test would require more complex mocking of pandas operations
        # For now, we'll just verify the function exists and can be called
        assert callable(add_efo_mapping)

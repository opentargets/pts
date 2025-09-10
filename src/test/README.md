# PTS Test Suite

This directory contains tests for the PTS (Pipeline Transformation Stage) project.

## Test Files

### `test_ontology_utils.py`
Comprehensive test suite for the ontology utility module (`pts.utils.ontology`). This includes:

- **TestSimpleRetry**: Tests for the `_simple_retry` function including:
  - Successful function calls
  - Retry logic on failures
  - Maximum attempts handling
  - Keyword argument handling

- **TestOntomaUdf**: Tests for the `_ontoma_udf` function including:
  - Mapping by disease name
  - Mapping by disease ID when name mapping fails
  - Handling of obsolete disease names
  - Empty field handling
  - Multiple mappings

- **TestAddEfoMapping**: Tests for the `add_efo_mapping` function including:
  - Successful EFO mapping
  - Environment variable handling for EFO version
  - Parameter handling (cache directory, EFO version)
  - Empty DataFrame handling

- **TestEdgeCases**: Tests for edge cases and error handling including:
  - Constant definitions
  - Sleep timing in retry logic
  - Multiple mappings explosion
  - NaN handling

### `test_ontology_utils_simple.py`
Simple test suite that can run without external dependencies. This includes basic import tests and functionality verification.

### `test_stub.py`
Basic test stub for the test framework.

## Running Tests

### Prerequisites
Before running the full test suite, ensure all dependencies are installed:

```bash
# Install the project in development mode
pip install -e .

# Or install dependencies directly
pip install numpy ontoma pandarallel pyspark pandas pytest
```

### Running Tests

#### Using pytest (recommended)
```bash
# Run all tests
pytest

# Run only ontology utility tests
pytest src/test/test_ontology_utils.py -v

# Run with coverage
pytest --cov=pts.utils.ontology src/test/test_ontology_utils.py
```

#### Using simple test runner
```bash
# Run simple tests (no external dependencies required)
python3 src/test/test_ontology_utils_simple.py
```

## Test Coverage

The test suite covers:

- ✅ Function imports and basic structure
- ✅ Retry logic with success and failure scenarios
- ✅ Disease name and ID mapping
- ✅ EFO version handling (environment variables and parameters)
- ✅ OnToma cache directory configuration
- ✅ Empty DataFrame handling
- ✅ Edge cases and error conditions
- ✅ Multiple mapping scenarios
- ✅ NaN value handling

## Mocking Strategy

The tests use extensive mocking to isolate the functionality being tested:

- **OnToma library**: Mocked to avoid external API calls
- **Pandas operations**: Mocked to test DataFrame handling
- **Spark operations**: Mocked to test DataFrame transformations
- **Time operations**: Mocked to speed up retry tests
- **Environment variables**: Mocked to test configuration handling

## Test Data

The tests use realistic test data including:
- Disease names: "Alzheimer disease", "Type 2 diabetes"
- Disease IDs: "MONDO:0004975", "MONDO:0005148"
- EFO IDs: "EFO:0001234", "EFO:0005678"
- Edge cases: empty strings, None values, invalid IDs

## Continuous Integration

The test suite is designed to work in CI environments:
- No external network dependencies (all external calls are mocked)
- Deterministic test results
- Fast execution (mocked time operations)
- Clear error messages and assertions

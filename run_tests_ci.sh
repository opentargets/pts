#!/bin/bash
# Script to run tests in CI environment without Spark dependencies

echo "Running PTS tests in CI environment..."
echo "======================================"

# Set environment variable to skip Spark tests
export SKIP_SPARK_TESTS=true

# Run CI-friendly tests
echo "Running CI-friendly ontology tests..."
uv run pytest src/test/test_ontology_utils_ci.py -v

# Run simple test runner
echo "Running simple test runner..."
uv run python3 src/test/test_ontology_utils_simple.py

# Run other tests excluding Spark tests
echo "Running other tests (excluding Spark tests)..."
uv run pytest src/test/ -v -m "not spark"

echo "======================================"
echo "CI test run completed."

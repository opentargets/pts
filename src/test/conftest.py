"""Pytest configuration and shared fixtures for PTS tests."""

import pytest
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession


def get_spark_testing_conf() -> SparkConf:
    """Get SparkConf optimized for testing purposes.

    This configuration is designed to:
    - Minimize resource usage for CI environments
    - Disable UI components that aren't needed for tests
    - Set conservative memory limits
    - Use minimal partitions for test data

    Returns:
        SparkConf: SparkConf with settings optimized for testing.
    """
    return (
        SparkConf()
        .set('spark.driver.bindAddress', '127.0.0.1')
        # Minimal shuffling for tests
        .set('spark.sql.shuffle.partitions', '1')
        .set('spark.default.parallelism', '1')
        # Disable UI components to save resources
        .set('spark.ui.showConsoleProgress', 'false')
        .set('spark.ui.enabled', 'false')
        .set('spark.ui.dagGraph.retainedRootRDDs', '1')
        .set('spark.ui.retainedJobs', '1')
        .set('spark.ui.retainedStages', '1')
        .set('spark.ui.retainedTasks', '1')
        .set('spark.sql.ui.retainedExecutions', '1')
        .set('spark.worker.ui.retainedExecutors', '1')
        .set('spark.worker.ui.retainedDrivers', '1')
        # Conservative memory settings for CI
        .set('spark.driver.memory', '1g')
        .set('spark.executor.memory', '1g')
        # Use Kryo serializer for better performance
        .set('spark.serializer', 'org.apache.spark.serializer.KryoSerializer')
        # Reduce logging noise in tests
        .set('spark.sql.adaptive.enabled', 'false')
        .set('spark.sql.adaptive.coalescePartitions.enabled', 'false')
        # Disable unnecessary features for tests
        .set('spark.sql.execution.arrow.pyspark.enabled', 'false')
    )


@pytest.fixture(scope='session')
def spark():
    """Create a Spark session for testing.

    This fixture is session-scoped to avoid recreating Spark sessions
    for multiple tests, which improves test performance.
    """
    spark_session = (
        SparkSession.builder
        .config(conf=get_spark_testing_conf())
        .master('local[1]')
        .appName('pts_test_suite')
        .getOrCreate()
    )
    spark_session.sparkContext.setLogLevel('ERROR')
    yield spark_session
    spark_session.stop()

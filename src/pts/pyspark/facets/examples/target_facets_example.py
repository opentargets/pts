"""Example script demonstrating how to use the target_facets module.

This script shows both local and GCS usage patterns for computing target facets.
"""

from pts.pyspark.facets import target_facets


# Example 1: Local filesystem usage
def example_local():
    """Example using local filesystem paths."""
    print('Example 1: Local filesystem usage')

    source = {
        'targets': 'work/output/targets/targets.parquet',
        'go': 'work/output/go/go.parquet'
    }

    destination = {
        'targets': 'work/output/facets/target_facets.parquet'
    }

    # Optional: Custom Spark properties
    properties = {
        'spark.sql.shuffle.partitions': '8',
    }

    # Optional: Custom category names
    category_config = {
        'SM': 'Small Molecule',
        'AB': 'Antibody',
        # ... other custom category names
    }

    target_facets(
        source=source,
        destination=destination,
        properties=properties,
        category_config=category_config
    )

    print('Target facets computed successfully (local)')


# Example 2: Google Cloud Storage usage
def example_gcs():
    """Example using Google Cloud Storage paths."""
    print('Example 2: Google Cloud Storage usage')

    source = {
        'targets': 'gs://your-bucket/output/targets/targets.parquet',
        'go': 'gs://your-bucket/output/go/go.parquet'
    }

    destination = {
        'targets': 'gs://your-bucket/output/facets/target_facets.parquet'
    }

    # The Session class automatically configures GCS support
    target_facets(
        source=source,
        destination=destination,
    )

    print('Target facets computed successfully (GCS)')


# Example 3: Using individual facet functions for fine-grained control
def example_individual_facets():
    """Example using individual facet computation functions."""
    print('Example 3: Individual facet computation')

    from pts.pyspark.common.session import Session
    from pts.pyspark.facets import (
        FacetSearchCategories,
        compute_go_facets,
        compute_tractability_facets,
    )

    # Initialize Spark session
    session = Session(app_name='facets_example')
    spark = session.spark

    try:
        # Load data
        targets_df = spark.read.parquet('work/output/targets/targets.parquet')
        go_df = spark.read.parquet('work/output/go/go.parquet')

        # Initialize category configuration
        categories = FacetSearchCategories()

        # Compute specific facets
        tractability_facets = compute_tractability_facets(targets_df, categories, spark)
        go_facets = compute_go_facets(targets_df, go_df, categories, spark)

        # Display results
        print(f'Tractability facets count: {tractability_facets.count()}')
        print(f'GO facets count: {go_facets.count()}')

        # Show sample data
        print('\nSample tractability facets:')
        tractability_facets.show(5, truncate=False)

        print('\nSample GO facets:')
        go_facets.show(5, truncate=False)

        # Write specific facets
        tractability_facets.write.mode('overwrite').parquet('work/output/facets/tractability_facets.parquet')

    finally:
        session.stop()

    print('Individual facets computed successfully')


# Example 4: Testing with sample data
def example_with_test_data():
    """Example creating minimal test data and computing facets."""
    print('Example 4: Testing with sample data')

    from pyspark.sql import Row

    from pts.pyspark.common.session import Session
    from pts.pyspark.facets import FacetSearchCategories, compute_all_target_facets

    session = Session(app_name='facets_test')
    spark = session.spark

    try:
        # Create minimal test target data
        test_targets = [
            Row(
                id='ENSG00000001',
                approvedSymbol='TP53',
                approvedName='Tumor protein p53',
                tractability=[
                    Row(modality='SM', id='Clinical Precedence', value=True),
                    Row(modality='AB', id='Predicted Tractable', value=False),
                ],
                go=[
                    Row(id='GO:0006915', aspect='P'),  # apoptotic process
                    Row(id='GO:0003700', aspect='F'),  # DNA-binding transcription factor activity
                ]
            ),
            Row(
                id='ENSG00000002',
                approvedSymbol='BRCA1',
                approvedName='BRCA1 DNA repair associated',
                tractability=[
                    Row(modality='SM', id='Discovery Precedence', value=True),
                ],
                go=[
                    Row(id='GO:0006281', aspect='P'),  # DNA repair
                ]
            ),
        ]
        targets_df = spark.createDataFrame(test_targets)

        # Create minimal test GO data
        test_go = [
            Row(id='GO:0006915', name='apoptotic process', namespace='biological_process'),
            Row(id='GO:0003700', name='DNA-binding transcription factor activity', namespace='molecular_function'),
            Row(id='GO:0006281', name='DNA repair', namespace='biological_process'),
        ]
        go_df = spark.createDataFrame(test_go)

        # Compute facets
        categories = FacetSearchCategories()
        facets_df = compute_all_target_facets(targets_df, go_df, categories, spark)

        # Display results
        print(f'Total facets computed: {facets_df.count()}')
        print('\nAll facets:')
        facets_df.show(truncate=False)

    finally:
        session.stop()

    print('Test data example completed successfully')


if __name__ == '__main__':
    print('Target Facets Examples\n' + '=' * 50)

    # Uncomment the example you want to run:

    # example_local()
    # example_gcs()
    # example_individual_facets()
    example_with_test_data()


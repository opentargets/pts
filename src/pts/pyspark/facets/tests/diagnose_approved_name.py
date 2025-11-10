"""Diagnostic script to investigate Approved Name facet count differences.

This script compares the Python and Scala implementations to identify
why Approved Name facets have different counts.
"""
from pyspark.sql import functions as F

from pts.pyspark.common.session import Session

print('=' * 80)
print('APPROVED NAME FACETS DIAGNOSIS')
print('=' * 80)

session = Session(app_name='diagnose_approved_name')
spark = session.spark

try:
    # Load both datasets
    print('\n📥 Loading datasets...')
    python_facets = spark.read.option('pathGlobFilter', '*.parquet').parquet(
        'work/output/facets/real_target_facets.parquet'
    )
    scala_facets = spark.read.option('pathGlobFilter', '*.parquet').parquet(
        'work/input/search_target_facet'
    )

    # Filter to Approved Name category
    python_approved = python_facets.filter(F.col('category') == 'Approved Name')
    scala_approved = scala_facets.filter(F.col('category') == 'Approved Name')

    print(f'\n✅ Python Approved Name facets: {python_approved.count()}')
    print(f'✅ Scala Approved Name facets:  {scala_approved.count()}')

    # Check for null labels
    print('\n' + '=' * 80)
    print('NULL LABEL ANALYSIS')
    print('=' * 80)

    python_null_labels = python_approved.filter(F.col('label').isNull()).count()
    scala_null_labels = scala_approved.filter(F.col('label').isNull()).count()

    print(f'\nPython facets with null labels: {python_null_labels}')
    print(f'Scala facets with null labels:  {scala_null_labels}')

    # Check for empty string labels
    python_empty_labels = python_approved.filter(F.col('label') == '').count()
    scala_empty_labels = scala_approved.filter(F.col('label') == '').count()

    print(f'\nPython facets with empty string labels: {python_empty_labels}')
    print(f'Scala facets with empty string labels:  {scala_empty_labels}')

    # Check for whitespace-only labels
    python_whitespace_labels = python_approved.filter(
        F.trim(F.col('label')) == ''
    ).filter(F.col('label').isNotNull()).count()
    scala_whitespace_labels = scala_approved.filter(
        F.trim(F.col('label')) == ''
    ).filter(F.col('label').isNotNull()).count()

    print(f'\nPython facets with whitespace-only labels: {python_whitespace_labels}')
    print(f'Scala facets with whitespace-only labels:  {scala_whitespace_labels}')

    # Find labels in Scala but not in Python
    print('\n' + '=' * 80)
    print('LABEL DIFFERENCES')
    print('=' * 80)

    python_labels = python_approved.select('label').distinct()
    scala_labels = scala_approved.select('label').distinct()

    python_label_set = set([row['label'] for row in python_labels.collect()])
    scala_label_set = set([row['label'] for row in scala_labels.collect()])

    only_in_scala = scala_label_set - python_label_set
    only_in_python = python_label_set - scala_label_set

    print(f'\nLabels only in Scala: {len(only_in_scala)}')
    if len(only_in_scala) > 0:
        print(f'  Showing first 30 examples:')
        for i, label in enumerate(sorted(list(only_in_scala))[:30], 1):
            # Get entity IDs for this label in Scala
            scala_example = scala_approved.filter(F.col('label') == label).first()
            entity_ids = scala_example['entityIds'] if scala_example else []
            print(f'  {i}. "{label}" -> {len(entity_ids)} entity IDs: {entity_ids[:3]}...')

    print(f'\nLabels only in Python: {len(only_in_python)}')
    if len(only_in_python) > 0:
        print(f'  Showing first 30 examples:')
        for i, label in enumerate(sorted(list(only_in_python))[:30], 1):
            # Get entity IDs for this label in Python
            python_example = python_approved.filter(F.col('label') == label).first()
            entity_ids = python_example['entityIds'] if python_example else []
            print(f'  {i}. "{label}" -> {len(entity_ids)} entity IDs: {entity_ids[:3]}...')
    
    # Check if there are case sensitivity differences
    print('\n' + '=' * 80)
    print('CASE SENSITIVITY CHECK')
    print('=' * 80)
    
    python_lower = set([str(l).lower() if l else '' for l in python_label_set])
    scala_lower = set([str(l).lower() if l else '' for l in scala_label_set])
    
    only_in_scala_lower = scala_lower - python_lower
    only_in_python_lower = python_lower - scala_lower
    
    print(f'\nLabels only in Scala (case-insensitive): {len(only_in_scala_lower)}')
    print(f'Labels only in Python (case-insensitive): {len(only_in_python_lower)}')
    
    if len(only_in_scala_lower) < len(only_in_scala):
        print('⚠️  Some differences may be due to case sensitivity!')

    # Sample data comparison
    print('\n' + '=' * 80)
    print('SAMPLE DATA COMPARISON')
    print('=' * 80)

    print('\n📋 Python sample (first 10):')
    python_approved.select('label', 'entityIds').show(10, truncate=False)

    print('\n📋 Scala sample (first 10):')
    scala_approved.select('label', 'entityIds').show(10, truncate=False)

    # Check entityIds counts
    print('\n' + '=' * 80)
    print('ENTITY IDS ANALYSIS')
    print('=' * 80)

    python_with_entity_count = python_approved.withColumn(
        'entity_count', F.size(F.col('entityIds'))
    )
    scala_with_entity_count = scala_approved.withColumn(
        'entity_count', F.size(F.col('entityIds'))
    )

    print('\nPython entityIds size distribution:')
    python_with_entity_count.groupBy('entity_count').count().orderBy('entity_count').show(20)

    print('\nScala entityIds size distribution:')
    scala_with_entity_count.groupBy('entity_count').count().orderBy('entity_count').show(20)

    # Check for duplicate labels with different entity sets
    print('\n' + '=' * 80)
    print('DUPLICATE LABEL ANALYSIS')
    print('=' * 80)

    python_label_counts = python_approved.groupBy('label').count()
    scala_label_counts = scala_approved.groupBy('label').count()

    python_duplicates = python_label_counts.filter(F.col('count') > 1)
    scala_duplicates = scala_label_counts.filter(F.col('count') > 1)

    print(f'\nPython labels appearing multiple times: {python_duplicates.count()}')
    if python_duplicates.count() > 0:
        python_duplicates.show(10, truncate=False)

    print(f'\nScala labels appearing multiple times: {scala_duplicates.count()}')
    if scala_duplicates.count() > 0:
        scala_duplicates.show(10, truncate=False)

finally:
    session.stop()

print('\n✅ Diagnosis complete!')


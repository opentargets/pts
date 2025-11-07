"""Compare Python-generated facets with Scala-generated facets.

This script compares the output from the new Python implementation with the
existing Scala implementation to verify correctness.
"""
from pyspark.sql import functions as F

from pts.pyspark.common.session import Session

print('=' * 80)
print('FACETS COMPARISON: Python vs Scala')
print('=' * 80)


session = Session(app_name='compare_facets')
spark = session.spark

try:
    # Load both datasets
    print('\n📥 Loading datasets...')
    
    # Use pathGlobFilter to ignore _SUCCESS, .gstmp, and other non-parquet files
    python_facets = spark.read.option('pathGlobFilter', '*.parquet').parquet('/Users/polina/ot_pipelines/pts/work/output/facets/real_target_facets.parquet')
    scala_facets = spark.read.option('pathGlobFilter', '*.parquet').parquet('/Users/polina/ot_pipelines/pts/work/input/search_target_facet')
    
    print(f'✅ Python facets loaded')
    print(f'✅ Scala facets loaded')
    
    # Overall counts
    print('\n' + '=' * 80)
    print('OVERALL COUNTS')
    print('=' * 80)
    
    python_total = python_facets.count()
    scala_total = scala_facets.count()
    
    print(f'Python facets total: {python_total:,}')
    print(f'Scala facets total:  {scala_total:,}')
    print(f'Difference:          {abs(python_total - scala_total):,}')
    
    if python_total == scala_total:
        print('✅ Total counts match!')
    else:
        pct_diff = abs(python_total - scala_total) / scala_total * 100
        print(f'⚠️  Difference: {pct_diff:.2f}%')
    
    # Counts by category
    print('\n' + '=' * 80)
    print('COUNTS BY CATEGORY')
    print('=' * 80)
    
    python_by_cat = python_facets.groupBy('category').count().orderBy('category')
    scala_by_cat = scala_facets.groupBy('category').count().orderBy('category')
    
    print('\n📊 Python facets by category:')
    python_by_cat.show(20, truncate=False)
    
    print('\n📊 Scala facets by category:')
    scala_by_cat.show(20, truncate=False)
    
    # Side-by-side comparison
    print('\n' + '=' * 80)
    print('SIDE-BY-SIDE COMPARISON')
    print('=' * 80)
    
    python_counts = python_by_cat.withColumnRenamed('count', 'python_count')
    scala_counts = scala_by_cat.withColumnRenamed('count', 'scala_count')
    
    comparison = python_counts.join(scala_counts, on='category', how='full_outer').orderBy('category')
    comparison = comparison.withColumn(
        'difference',
        F.coalesce(F.col('python_count'), F.lit(0)) - F.coalesce(F.col('scala_count'), F.lit(0))
    )
    comparison = comparison.withColumn(
        'match',
        F.when(F.col('difference') == 0, '✅').otherwise('❌')
    )
    
    print('\n')
    comparison.show(100, truncate=False)
    
    # Check for matching categories
    print('\n' + '=' * 80)
    print('CATEGORY ANALYSIS')
    print('=' * 80)
    
    python_categories = set([row['category'] for row in python_by_cat.collect()])
    scala_categories = set([row['category'] for row in scala_by_cat.collect()])
    
    print(f'\nPython categories: {len(python_categories)}')
    print(f'Scala categories:  {len(scala_categories)}')
    
    only_in_python = python_categories - scala_categories
    only_in_scala = scala_categories - python_categories
    in_both = python_categories & scala_categories
    
    print(f'In both:           {len(in_both)}')
    
    if only_in_python:
        print(f'\n⚠️  Only in Python ({len(only_in_python)}):')
        for cat in sorted(only_in_python):
            print(f'   - {cat}')
    
    if only_in_scala:
        print(f'\n⚠️  Only in Scala ({len(only_in_scala)}):')
        for cat in sorted(only_in_scala):
            print(f'   - {cat}')
    
    if not only_in_python and not only_in_scala:
        print('✅ All categories match!')
    
    # Schema comparison
    print('\n' + '=' * 80)
    print('SCHEMA COMPARISON')
    print('=' * 80)
    
    print('\n📋 Python schema:')
    python_facets.printSchema()
    
    print('\n📋 Scala schema:')
    scala_facets.printSchema()
    
    # Check if schemas match
    python_cols = set(python_facets.columns)
    scala_cols = set(scala_facets.columns)
    
    if python_cols == scala_cols:
        print('✅ Column names match!')
    else:
        print('⚠️  Column names differ:')
        if python_cols - scala_cols:
            print(f'   Only in Python: {python_cols - scala_cols}')
        if scala_cols - python_cols:
            print(f'   Only in Scala: {scala_cols - python_cols}')
    
    # Sample comparison
    print('\n' + '=' * 80)
    print('SAMPLE DATA COMPARISON')
    print('=' * 80)
    
    # Pick a common label to compare
    common_label = 'TP53'  # Common gene symbol
    
    print(f'\n🔍 Comparing facets for label: "{common_label}"')
    
    python_sample = python_facets.filter(F.col('label') == common_label)
    scala_sample = scala_facets.filter(F.col('label') == common_label)
    
    print(f'\nPython ({python_sample.count()} rows):')
    python_sample.show(10, truncate=False)
    
    print(f'\nScala ({scala_sample.count()} rows):')
    scala_sample.show(10, truncate=False)
    
    # Summary
    print('\n' + '=' * 80)
    print('SUMMARY')
    print('=' * 80)
    
    all_match = (python_total == scala_total and 
                 python_categories == scala_categories and
                 python_cols == scala_cols)
    
    if all_match:
        print('\n✅ PERFECT MATCH! Python implementation matches Scala exactly.')
    else:
        print('\n⚠️  DIFFERENCES FOUND - Review details above')
        print('\nPossible reasons for differences:')
        print('  - Different input data (different target file versions)')
        print('  - Different GO reference data versions')
        print('  - Implementation differences (e.g., handling of null values)')
        print('  - Expected if using different data sources')
    
    print('\n' + '=' * 80)

finally:
    session.stop()
    print('\n✅ Comparison complete!')



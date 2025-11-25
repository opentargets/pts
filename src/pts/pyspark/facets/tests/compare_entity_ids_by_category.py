"""Compare entityIds vs entityIdsPropagated by category.

This script analyzes the output parquet file and shows statistics comparing
the original entityIds with the propagated entityIdsPropagated for each category.
"""

from pyspark.sql import functions as F

from pts.pyspark.common.session import Session


def main() -> None:
    """Compare entityIds and entityIdsPropagated by category."""
    session = Session(app_name='compare_entity_ids_by_category')
    spark = session.spark

    try:
        # Read the output parquet file
        input_path = 'work/output/facets/facets_propagated_merged'
        print(f'📥 Reading facets from: {input_path}\n')
        
        facets_df = spark.read.option('pathGlobFilter', '*.parquet').parquet(input_path)
        
        # Check if entityIdsPropagated column exists
        if 'entityIdsPropagated' not in facets_df.columns:
            print('❌ ERROR: entityIdsPropagated column not found in the data!')
            print(f'Available columns: {facets_df.columns}')
            return
        
        total_facets = facets_df.count()
        print(f'✅ Loaded {total_facets:,} total facets\n')
        
        # Calculate sizes for comparison
        facets_with_sizes = facets_df.withColumn(
            'entityIds_size', F.size('entityIds')
        ).withColumn(
            'entityIdsPropagated_size', F.size('entityIdsPropagated')
        ).withColumn(
            'size_difference', F.col('entityIdsPropagated_size') - F.col('entityIds_size')
        ).withColumn(
            'has_propagation', F.col('entityIdsPropagated_size') > F.col('entityIds_size')
        )
        
        # Statistics by category
        print('=' * 100)
        print('STATISTICS BY CATEGORY')
        print('=' * 100)
        
        category_stats = (
            facets_with_sizes.groupBy('category')
            .agg(
                F.count('*').alias('total_facets'),
                
                # Average sizes
                F.avg('entityIds_size').alias('avg_entityIds_size'),
                F.avg('entityIdsPropagated_size').alias('avg_entityIdsPropagated_size'),
                F.avg('size_difference').alias('avg_size_difference'),
                
                # Median sizes (using percentile_approx)
                F.expr('percentile_approx(entityIds_size, 0.5)').alias('median_entityIds_size'),
                F.expr('percentile_approx(entityIdsPropagated_size, 0.5)').alias('median_entityIdsPropagated_size'),
                
                # Min/Max sizes
                F.min('entityIds_size').alias('min_entityIds_size'),
                F.max('entityIds_size').alias('max_entityIds_size'),
                F.min('entityIdsPropagated_size').alias('min_entityIdsPropagated_size'),
                F.max('entityIdsPropagated_size').alias('max_entityIdsPropagated_size'),
                
                # Count facets with propagation
                F.sum(F.when(F.col('has_propagation'), 1).otherwise(0)).alias('facets_with_propagation'),
                F.sum(F.when(F.col('entityIdsPropagated_size') == F.col('entityIds_size'), 1).otherwise(0)).alias('facets_unchanged'),
                F.sum(F.when(F.col('entityIdsPropagated_size') < F.col('entityIds_size'), 1).otherwise(0)).alias('facets_decreased'),
                
                # Total unique entityIds across all facets in category
                F.expr('size(array_distinct(flatten(collect_list(entityIds))))').alias('total_unique_entityIds'),
                F.expr('size(array_distinct(flatten(collect_list(entityIdsPropagated))))').alias('total_unique_entityIdsPropagated'),
            )
            .orderBy('category')
        )
        
        # Format and display results
        print('\n📊 Detailed Statistics:\n')
        category_stats.select(
            'category',
            'total_facets',
            F.round('avg_entityIds_size', 2).alias('avg_entityIds'),
            F.round('avg_entityIdsPropagated_size', 2).alias('avg_entityIdsPropagated'),
            F.round('avg_size_difference', 2).alias('avg_diff'),
            'median_entityIds_size',
            'median_entityIdsPropagated_size',
            'facets_with_propagation',
            'facets_unchanged',
            'total_unique_entityIds',
            'total_unique_entityIdsPropagated',
        ).show(100, truncate=False)
        
        # Summary table
        print('\n' + '=' * 100)
        print('SUMMARY BY CATEGORY')
        print('=' * 100)
        
        summary = category_stats.select(
            'category',
            'total_facets',
            F.round('avg_entityIds_size', 1).alias('avg_original'),
            F.round('avg_entityIdsPropagated_size', 1).alias('avg_propagated'),
            F.round(
                (F.col('facets_with_propagation') / F.col('total_facets') * 100), 
                1
            ).alias('pct_with_propagation'),
            'facets_with_propagation',
            F.round(
                ((F.col('total_unique_entityIdsPropagated') - F.col('total_unique_entityIds')) / 
                 F.col('total_unique_entityIds') * 100), 
                1
            ).alias('pct_more_unique_ids'),
        ).orderBy('total_facets', ascending=False)
        
        print('\n📈 Summary (sorted by total facets):\n')
        summary.show(100, truncate=False)
        
        # Overall statistics
        print('\n' + '=' * 100)
        print('OVERALL STATISTICS')
        print('=' * 100)
        
        overall = facets_with_sizes.agg(
            F.count('*').alias('total_facets'),
            F.sum(F.when(F.col('has_propagation'), 1).otherwise(0)).alias('facets_with_propagation'),
            F.avg('entityIds_size').alias('avg_entityIds_size'),
            F.avg('entityIdsPropagated_size').alias('avg_entityIdsPropagated_size'),
            F.avg('size_difference').alias('avg_size_difference'),
        ).collect()[0]
        
        print(f'\nTotal facets: {overall.total_facets:,}')
        print(f'Facets with propagation (entityIdsPropagated > entityIds): {overall.facets_with_propagation:,}')
        print(f'  Percentage: {overall.facets_with_propagation / overall.total_facets * 100:.2f}%')
        print(f'\nAverage entityIds size: {overall.avg_entityIds_size:.2f}')
        print(f'Average entityIdsPropagated size: {overall.avg_entityIdsPropagated_size:.2f}')
        print(f'Average difference: {overall.avg_size_difference:.2f}')
        
        # Examples of facets with significant propagation
        print('\n' + '=' * 100)
        print('EXAMPLES: Facets with Significant Propagation')
        print('=' * 100)
        
        significant_propagation = (
            facets_with_sizes
            .filter(F.col('size_difference') >= 5)  # At least 5 more entityIds after propagation
            .select(
                'category',
                'label',
                'entityIds_size',
                'entityIdsPropagated_size',
                'size_difference',
            )
            .orderBy('size_difference', ascending=False)
            .limit(20)
        )
        
        print(f'\n📋 Top 20 facets with largest propagation (showing first 10):\n')
        significant_propagation.show(10, truncate=False)
        
        # Category breakdown of propagation
        print('\n' + '=' * 100)
        print('PROPAGATION IMPACT BY CATEGORY')
        print('=' * 100)
        
        propagation_impact = (
            category_stats.select(
                'category',
                'total_facets',
                'facets_with_propagation',
                F.round(
                    (F.col('facets_with_propagation') / F.col('total_facets') * 100), 
                    1
                ).alias('pct_with_propagation'),
                F.round('avg_size_difference', 2).alias('avg_increase'),
                F.round(
                    ((F.col('total_unique_entityIdsPropagated') - F.col('total_unique_entityIds')) / 
                     F.col('total_unique_entityIds') * 100), 
                    1
                ).alias('pct_more_unique'),
            )
            .orderBy('avg_increase', ascending=False)
        )
        
        print('\n📊 Categories sorted by average increase in entityIds:\n')
        propagation_impact.show(100, truncate=False)
        
        print('\n' + '=' * 100)
        print('✅ Analysis complete!')
        print('=' * 100)
        
    finally:
        session.stop()


if __name__ == '__main__':
    main()


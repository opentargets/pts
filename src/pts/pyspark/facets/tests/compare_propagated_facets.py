"""Compare original and propagated facets outputs.

This script compares two facet Parquet outputs and shows:
- category: Facet category name
- total_rows: Total number of rows per category
- changed_rows: Number of rows with different entityIds between the two dataframes
"""

from pyspark.sql import functions as F

from pts.pyspark.common.session import Session


def main() -> None:
    session = Session(app_name='compare_propagated_facets')
    spark = session.spark

    try:
        base_path = 'work/output/facets'
        before_path = f'{base_path}/real_target_facets_chembl'
        after_path = f'{base_path}/real_target_facets_propagated'

        # Read Parquet outputs (ignore non-parquet files like _SUCCESS)
        before_df = spark.read.option('pathGlobFilter', '*.parquet').parquet(before_path)
        after_df = spark.read.option('pathGlobFilter', '*.parquet').parquet(after_path)

        # Prepare data for comparison
        before_with_size = before_df.select(
            'label',
            'category',
            'datasourceId',
            'parentId',
            F.size('entityIds').alias('before_size'),
            F.array_sort('entityIds').alias('before_sorted'),
        )
        after_with_size = after_df.select(
            'label',
            'category',
            'datasourceId',
            'parentId',
            F.size('entityIds').alias('after_size'),
            F.array_sort('entityIds').alias('after_sorted'),
        )

        # Join on all identifying columns to match the same facet row
        comparison = (
            before_with_size.alias('b')
            .join(after_with_size.alias('a'), on=['label', 'category', 'datasourceId', 'parentId'], how='outer')
            .withColumn(
                'changed',
                F.when(
                    (F.col('b.before_size').isNull()) | (F.col('a.after_size').isNull()),
                    F.lit(True),  # Row missing in one dataset
                )
                .when(
                    F.col('b.before_size') != F.col('a.after_size'),
                    F.lit(True),  # Size changed
                )
                .otherwise(
                    # Even if size is same, check if content changed
                    # Use array_except to check if arrays differ
                    (F.size(F.array_except(F.col('b.before_sorted'), F.col('a.after_sorted'))) > 0)
                    | (F.size(F.array_except(F.col('a.after_sorted'), F.col('b.before_sorted'))) > 0)
                ),
            )
        )

        # Show results: category, total_rows, changed_rows
        result = (
            comparison.groupBy('category')
            .agg(
                F.count('*').alias('total_rows'),
                F.sum(F.when(F.col('changed'), 1).otherwise(0)).alias('changed_rows'),
            )
            .select('category', 'total_rows', 'changed_rows')
            .orderBy('changed_rows', ascending=False)
        )

        result.show(100, truncate=False)

    finally:
        session.stop()


if __name__ == '__main__':
    main()

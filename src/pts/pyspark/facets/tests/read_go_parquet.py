"""Simple script to read and inspect the GO parquet file."""
from pts.pyspark.common.session import Session

print('=' * 80)
print('READING GO PARQUET FILE')
print('=' * 80)

session = Session(app_name='read_go_parquet')
spark = session.spark

try:
    # Read the parquet file
    print('\n📥 Loading GO parquet file...')
    go_df = spark.read.parquet('/Users/polina/ot_pipelines/pts/work/output/go/go.parquet')
    
    print(f'✅ File loaded successfully')
    
    # Show basic info
    print(f'\n📊 Row count: {go_df.count()}')
    print(f'\n📋 Schema:')
    go_df.printSchema()
    
    # Show sample data
    print('\n📋 Sample data (first 10 rows):')
    go_df.show(10, truncate=False)
    
    # Show column statistics if useful
    print('\n📊 Column names:')
    for col_name in go_df.columns:
        print(f'  - {col_name}')
    
finally:
    session.stop()

print('\n✅ Done!')



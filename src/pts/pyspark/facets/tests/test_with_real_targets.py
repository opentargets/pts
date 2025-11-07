"""Test facets with real downloaded targets data."""

from pts.pyspark.common.session import Session
from pts.pyspark.facets import target_facets

print('🚀 Testing facets with real targets data\n')

# Paths - using the single parquet file you downloaded
source = {
    'targets': 'work/input/target',  # Will read all parquet files in this dir
    'go': 'work/output/go/go.parquet',
}

destination = {
    'targets': 'work/output/facets/real_target_facets.parquet'
}

print(f'📥 Reading targets from: {source["targets"]}')
print(f'📥 Reading GO from: {source["go"]}')
print(f'💾 Writing facets to: {destination["targets"]}\n')

try:
    # Compute facets
    target_facets(source=source, destination=destination)
    
    print('\n✅ SUCCESS! Facets computed from real data.\n')
    
    # Inspect results
    print('📊 Inspecting results...\n')
    session = Session(app_name='inspect')
    
    # Show input stats
    targets_df = session.spark.read.parquet(source['targets'])
    print(f'Input targets: {targets_df.count()} targets')
    
    # Show output stats
    facets_df = session.spark.read.parquet(destination['targets'])
    total = facets_df.count()
    print(f'Output facets: {total} facets\n')
    
    print('Facets by category:')
    facets_df.groupBy('category').count().orderBy('count', ascending=False).show(20, truncate=False)
    
    print('\nSample facets (first 10):')
    facets_df.show(10, truncate=False)
    
    print(f'\n✅ Results saved to: {destination["targets"]}')
    
    session.stop()
    
except Exception as e:
    print(f'\n❌ ERROR: {e}')
    import traceback
    traceback.print_exc()
"""Test facets with real downloaded targets data."""

from pyspark.sql import functions as F

from pts.pyspark.common.session import Session
from pts.pyspark.facets import target_facets

print('🚀 Testing facets with real targets data\n')

# Paths - using the single parquet file you downloaded
source = {
    'targets': 'work/input/target',  # Will read all parquet files in this dir
    'go': 'work/output/go/go.parquet',
    'reactome': 'work/input/reactome',  # Reactome reference data
}

destination = {'targets': 'work/output/facets/real_target_facets_propagated'}

print(f'📥 Reading targets from: {source["targets"]}')
print(f'📥 Reading GO from: {source["go"]}')
print(f'📥 Reading Reactome from: {source["reactome"]}')
print(f'💾 Writing facets to: {destination["targets"]}\n')

# Check GO file before filtering
print('🔍 Checking GO file for obsolete terms...')
session_check = Session(app_name='check_go')
go_df_check = session_check.spark.read.parquet(source['go'])
initial_count = go_df_check.count()
obsolete_count = go_df_check.filter(F.col('isObsolete') == True).count()
null_count = go_df_check.filter(F.col('isObsolete').isNull()).count()
false_count = go_df_check.filter(F.col('isObsolete') == False).count()
print(f'   Total GO terms: {initial_count}')
print(f'   Obsolete (True): {obsolete_count}')
print(f'   Not obsolete (False): {false_count}')
print(f'   Not obsolete (null): {null_count}')
print(f'   Will keep: {false_count + null_count} terms\n')
session_check.stop()

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

    # print('Facets by category:')
    # facets_df.groupBy('category').count().orderBy('count', ascending=False).show(20, truncate=False)

    # print('\nSample facets (first 10):')
    # facets_df.show(10, truncate=False)

    # print('\nSample facets with parentId (first 10):')
    # facets_with_parents = facets_df.filter(F.col('parentId').isNotNull() & (F.size(F.col('parentId')) > 0))
    # print(f'Total facets with parentId: {facets_with_parents.count()}')
    # facets_with_parents.show(10, truncate=False)

    # # Check Reactome facets specifically
    # print('\n🔍 Reactome facets with parentId:')
    # reactome_facets = facets_df.filter(F.col('category') == 'Reactome')
    # reactome_with_parents = reactome_facets.filter(F.col('parentId').isNotNull() & (F.size(F.col('parentId')) > 0))
    # reactome_without_parents = reactome_facets.filter(F.col('parentId').isNull() | (F.size(F.col('parentId')) == 0))
    # reactome_total = reactome_facets.count()
    # reactome_with_parents_count = reactome_with_parents.count()
    # print(f'   Total Reactome facets: {reactome_total}')
    # print(f'   Reactome facets with parentId: {reactome_with_parents_count}')
    # print(f'   Reactome facets without parentId: {reactome_total - reactome_with_parents_count}')
    # reactome_without_parents.show(10, truncate=False)

    # Check ChEMBL Target Class facets specifically
    # print('\n🔍 ChEMBL Target Class facets with parentId:')
    # chembl_facets = facets_df.filter(F.col('category') == 'ChEMBL Target Class')
    # chembl_with_parents = chembl_facets.filter(F.col('parentId').isNotNull() & (F.size(F.col('parentId')) > 0))
    # chembl_without_parents = chembl_facets.filter(F.col('parentId').isNull() | (F.size(F.col('parentId')) == 0))
    # chembl_total = chembl_facets.count()
    # chembl_with_parents_count = chembl_with_parents.count()
    # print(f'   Total ChEMBL Target Class facets: {chembl_total}')
    # print(f'   ChEMBL facets with parentId: {chembl_with_parents_count}')
    # print(f'   ChEMBL facets without parentId: {chembl_total - chembl_with_parents_count}')
    # if chembl_total - chembl_with_parents_count > 0:
    #     print('\n   ChEMBL labels without parentId:')
    #     chembl_without_parents.select('label').distinct().orderBy('label').show(100, truncate=False)
    # else:
    #     print('\n   ✅ All ChEMBL facets have parentId values')

    print(f'\n✅ Results saved to: {destination["targets"]}')

    session.stop()

except Exception as e:
    print(f'\n❌ ERROR: {e}')
    import traceback

    traceback.print_exc()

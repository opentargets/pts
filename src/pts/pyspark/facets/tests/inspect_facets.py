from pts.pyspark.common.session import Session

session = Session(app_name='inspect_facets')
spark = session.spark

# Load the actual facets output
facets_df = spark.read.parquet('work/output/facets/real_target_facets.parquet')

print("=" * 80)
print("FACETS WITH MULTIPLE ENTITY IDs")
print("=" * 80)

# Find facets with more than 1 entity ID
from pyspark.sql.functions import col, size

multi_entity_facets = facets_df.filter(size(col('entityIds')) > 1)

print(f"\nTotal facets: {facets_df.count()}")
print(f"Facets with multiple entities: {multi_entity_facets.count()}\n")

# Show breakdown by category
print("Facets with multiple entities by category:")
multi_entity_facets.groupBy('category').count().orderBy('count', ascending=False).show(20, truncate=False)

# Show examples from each category
categories = ['GO:BP', 'GO:MF', 'GO:CC', 
              'Reactome', 'Subcellular Location', 'ChEMBL Target Class', 
              'Tractability Small Molecule', 'Tractability Antibody', 
              'Tractability PROTAC', 'Tractability Other Modalities']

for cat in categories:
    print(f"\n{'=' * 80}")
    print(f"EXAMPLE: {cat}")
    print("=" * 80)
    
    cat_facets = multi_entity_facets.filter(col('category') == cat)
    
    if cat_facets.count() > 0:
        # Show one example with entity count
        example = cat_facets.withColumn('entity_count', size(col('entityIds'))).orderBy('entity_count', ascending=False).first()
        
        if example:
            print(f"Label: {example['label']}")
            print(f"Category: {example['category']}")
            print(f"Number of entities: {len(example['entityIds'])}")
            print(f"DatasourceId: {example['datasourceId']}")
            print(f"Entity IDs: {example['entityIds'][:10]}...")  # Show first 10
            if len(example['entityIds']) > 10:
                print(f"  ... and {len(example['entityIds']) - 10} more")

session.stop()
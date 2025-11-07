"""Quick test script for GO facets only."""

from pts.pyspark.common.session import Session
from pts.pyspark.facets import compute_go_facets, FacetSearchCategories
from pyspark.sql import Row

print('Testing GO Facets only...\n')

# Initialize Spark
session = Session(app_name='test_go_facets')
spark = session.spark

try:
    # Create minimal test target data (only needs id and go fields)
    test_targets = [
        Row(
            id='ENSG00000141510',  # TP53
            go=[
                Row(id='GO:0006915', aspect='P'),  # apoptotic process
                Row(id='GO:0003700', aspect='F'),  # DNA-binding transcription factor activity
                Row(id='GO:0005634', aspect='C'),  # nucleus
            ]
        ),
        Row(
            id='ENSG00000012048',  # BRCA1
            go=[
                Row(id='GO:0006281', aspect='P'),  # DNA repair
                Row(id='GO:0003677', aspect='F'),  # DNA binding
                Row(id='GO:0005654', aspect='C'),  # nucleoplasm
            ]
        ),
        Row(
            id='ENSG00000185345',  # EGFR
            go=[
                Row(id='GO:0007165', aspect='P'),  # signal transduction
                Row(id='GO:0004714', aspect='F'),  # transmembrane receptor protein tyrosine kinase activity
                Row(id='GO:0005886', aspect='C'),  # plasma membrane
            ]
        ),
    ]
    
    targets_df = spark.createDataFrame(test_targets)
    print(f'✅ Created test targets: {targets_df.count()} targets\n')
    
    # Load real GO reference data
    go_df = spark.read.parquet('work/output/go/go.parquet')
    print(f'✅ Loaded GO reference: {go_df.count()} terms\n')
    
    # Initialize categories
    categories = FacetSearchCategories()
    
    # Compute GO facets only
    print('Computing GO facets...')
    go_facets = compute_go_facets(targets_df, go_df, categories, spark)
    
    facet_count = go_facets.count()
    print(f'\n✅ GO facets computed: {facet_count} facets\n')
    
    # Show sample results
    print('Sample GO facets:')
    go_facets.show(10, truncate=False)
    
    # Show breakdown by category
    print('\nFacets by category:')
    go_facets.groupBy('category').count().show(truncate=False)
    
    # Save to go folder
    output_path = 'work/output/go/go_facets.parquet'
    print(f'\nSaving to: {output_path}')
    go_facets.write.mode('overwrite').parquet(output_path)
    
    print(f'\n✅ Success! GO facets saved to {output_path}')
    
finally:
    session.stop()


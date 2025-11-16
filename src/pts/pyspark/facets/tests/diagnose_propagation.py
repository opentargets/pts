"""Diagnose why Reactome and GO:CC show 0 changed rows after propagation.

The issue is likely that parentId contains IDs, but propagation matches on label (names).
"""

from pyspark.sql import functions as F

from pts.pyspark.common.session import Session


def main() -> None:
    session = Session(app_name='diagnose_propagation')
    spark = session.spark

    try:
        base_path = 'work/output/facets'
        before_path = f'{base_path}/real_target_facets_chembl'
        after_path = f'{base_path}/real_target_facets_propagated'

        before_df = spark.read.option('pathGlobFilter', '*.parquet').parquet(before_path)
        after_df = spark.read.option('pathGlobFilter', '*.parquet').parquet(after_path)

        # Check Reactome facets
        print('=' * 80)
        print('REACTOME FACETS (BEFORE PROPAGATION)')
        print('=' * 80)
        reactome_before = before_df.filter(F.col('category') == 'Reactome')
        print(f'Total Reactome facets: {reactome_before.count()}')
        
        # Check how many have non-empty parentId
        with_parents = reactome_before.filter(
            F.col('parentId').isNotNull() & (F.size(F.col('parentId')) > 0)
        )
        print(f'Reactome facets with non-empty parentId: {with_parents.count()}')
        
        if with_parents.count() > 0:
            print('\nSample Reactome facets with parentId:')
            with_parents.select('label', 'datasourceId', 'parentId', F.size('entityIds').alias('entityIds_size')).show(10, truncate=False)
            
            # Check if parentId values match any label or datasourceId
            sample_parent_id = with_parents.select('parentId').first()['parentId'][0] if with_parents.count() > 0 else None
            if sample_parent_id:
                print(f'\nChecking if parentId value "{sample_parent_id}" matches any label or datasourceId:')
                matching_label = reactome_before.filter(F.col('label') == sample_parent_id)
                matching_datasource = reactome_before.filter(F.col('datasourceId') == sample_parent_id)
                print(f'  Matches by label: {matching_label.count()} rows')
                print(f'  Matches by datasourceId: {matching_datasource.count()} rows')
                if matching_datasource.count() > 0:
                    matching_datasource.select('label', 'datasourceId').show(5, truncate=False)

        # Check GO:CC facets
        print('\n' + '=' * 80)
        print('GO:CC FACETS (BEFORE PROPAGATION)')
        print('=' * 80)
        go_cc_before = before_df.filter(F.col('category') == 'GO:CC')
        print(f'Total GO:CC facets: {go_cc_before.count()}')
        
        with_parents_cc = go_cc_before.filter(
            F.col('parentId').isNotNull() & (F.size(F.col('parentId')) > 0)
        )
        print(f'GO:CC facets with non-empty parentId: {with_parents_cc.count()}')
        
        if with_parents_cc.count() > 0:
            print('\nSample GO:CC facets with parentId:')
            with_parents_cc.select('label', 'datasourceId', 'parentId', F.size('entityIds').alias('entityIds_size')).show(10, truncate=False)
            
            # Check if parentId values match any label or datasourceId
            sample_parent_id_cc = with_parents_cc.select('parentId').first()['parentId'][0] if with_parents_cc.count() > 0 else None
            if sample_parent_id_cc:
                print(f'\nChecking if parentId value "{sample_parent_id_cc}" matches any label or datasourceId:')
                matching_label_cc = go_cc_before.filter(F.col('label') == sample_parent_id_cc)
                matching_datasource_cc = go_cc_before.filter(F.col('datasourceId') == sample_parent_id_cc)
                print(f'  Matches by label: {matching_label_cc.count()} rows')
                print(f'  Matches by datasourceId: {matching_datasource_cc.count()} rows')
                if matching_datasource_cc.count() > 0:
                    matching_datasource_cc.select('label', 'datasourceId').show(5, truncate=False)

    finally:
        session.stop()


if __name__ == '__main__':
    main()


"""Step-by-step test to debug the propagation algorithm."""

from pyspark.sql import Row
from pyspark.sql import functions as F

from pts.pyspark.common.session import Session


def test_step_a0_only():
    """Test only step a0 to see what children are found."""
    session = Session(app_name='test_step_a0')
    spark = session.spark

    try:
        # Input data
        input_data = [
            {'id': 'child1', 'parent_id': 'parent1', 'entityIds': ['entity1', 'entity2']},
            {'id': 'parent1', 'parent_id': 'grandparent1', 'entityIds': ['entity3']},
            {'id': 'grandparent1', 'parent_id': None, 'entityIds': ['entity4']},
        ]

        rows = [Row(**row) for row in input_data]
        df = spark.createDataFrame(rows)

        print('\n=== Input DataFrame ===')
        df.show(truncate=False)

        # Step a0: Get children table
        existing_ids = [row['id'] for row in df.select('id').distinct().collect()]
        print(f'\nExisting IDs: {existing_ids}')

        children_df = df.filter(F.col('parent_id').isNotNull() & F.col('parent_id').isin(existing_ids)).select(
            F.col('id').alias('child_id'),
            F.col('parent_id').alias('child_parent_id'),
            F.col('entityIds').alias('child_entityIds'),
        )

        print('\n=== Children DataFrame (step a0 output) ===')
        children_df.show(truncate=False)

        children_count = children_df.count()
        print(f'\nChildren count: {children_count}')

        # Verify expected children
        children_list = children_df.collect()
        assert children_count == 2
        child_ids = {row['child_id'] for row in children_list}
        assert child_ids == {'child1', 'parent1'}

    finally:
        session.stop()


def test_step_a_only():
    """Test step a (join) after step a0."""
    session = Session(app_name='test_step_a')
    spark = session.spark

    try:
        # Input data
        input_data = [
            {'id': 'child1', 'parent_id': 'parent1', 'entityIds': ['entity1', 'entity2']},
            {'id': 'parent1', 'parent_id': 'grandparent1', 'entityIds': ['entity3']},
            {'id': 'grandparent1', 'parent_id': None, 'entityIds': ['entity4']},
        ]

        rows = [Row(**row) for row in input_data]
        current_df = spark.createDataFrame(rows)

        # Step a0: Get children
        existing_ids = [row['id'] for row in current_df.select('id').distinct().collect()]
        children_df = current_df.filter(F.col('parent_id').isNotNull() & F.col('parent_id').isin(existing_ids)).select(
            F.col('id').alias('child_id'),
            F.col('parent_id').alias('child_parent_id'),
            F.col('entityIds').alias('child_entityIds'),
        )

        print('\n=== Current DataFrame ===')
        current_df.show(truncate=False)
        print('\n=== Children DataFrame ===')
        children_df.show(truncate=False)

        # Step a: Left join
        joined = (
            current_df.alias('df')
            .join(
                children_df.alias('children'),
                on=F.col('df.id') == F.col('children.child_parent_id'),
                how='left',
            )
            .select(
                F.col('df.id'),
                F.col('df.parent_id'),
                F.col('df.entityIds'),
                F.col('children.child_id'),
                F.col('children.child_entityIds').alias('child_entity_ids'),
            )
        )

        print('\n=== Joined DataFrame (step a output) ===')
        joined.show(truncate=False)

        # Verify join results
        joined_list = joined.collect()
        assert len(joined_list) == 3

        # Check that parent1 got child1's entityIds
        parent1_row = next(r for r in joined_list if r['id'] == 'parent1')
        assert parent1_row['child_id'] == 'child1'
        assert parent1_row['child_entity_ids'] == ['entity1', 'entity2']

        # Check that grandparent1 got parent1's entityIds
        grandparent1_row = next(r for r in joined_list if r['id'] == 'grandparent1')
        assert grandparent1_row['child_id'] == 'parent1'
        assert grandparent1_row['child_entity_ids'] == ['entity3']

    finally:
        session.stop()


def test_step_b_only():
    """Test step b (group and union) after step a."""
    session = Session(app_name='test_step_b')
    spark = session.spark

    try:
        # Input data
        input_data = [
            {'id': 'child1', 'parent_id': 'parent1', 'entityIds': ['entity1', 'entity2']},
            {'id': 'parent1', 'parent_id': 'grandparent1', 'entityIds': ['entity3']},
            {'id': 'grandparent1', 'parent_id': None, 'entityIds': ['entity4']},
        ]

        rows = [Row(**row) for row in input_data]
        current_df = spark.createDataFrame(rows)

        # Step a0: Get children
        existing_ids = [row['id'] for row in current_df.select('id').distinct().collect()]
        children_df = current_df.filter(F.col('parent_id').isNotNull() & F.col('parent_id').isin(existing_ids)).select(
            F.col('id').alias('child_id'),
            F.col('parent_id').alias('child_parent_id'),
            F.col('entityIds').alias('child_entityIds'),
        )

        # Step a: Join
        joined = (
            current_df.alias('df')
            .join(
                children_df.alias('children'),
                on=F.col('df.id') == F.col('children.child_parent_id'),
                how='left',
            )
            .select(
                F.col('df.id'),
                F.col('df.parent_id'),
                F.col('df.entityIds'),
                F.col('children.child_id'),
                F.col('children.child_entityIds').alias('child_entity_ids'),
            )
        )

        print('\n=== Joined DataFrame (input to step b) ===')
        joined.show(truncate=False)

        # Step b: Group and union
        result = (
            joined.groupBy('id', 'parent_id')
            .agg(
                F.first('entityIds').alias('original_entityIds'),
                F.collect_list('child_entity_ids').alias('child_entity_ids_list'),
            )
            .withColumn(
                'child_entityIds_flat',
                F.when(
                    F.col('child_entity_ids_list').isNotNull(),
                    F.array_distinct(
                        F.flatten(
                            F.filter(
                                F.col('child_entity_ids_list'),
                                lambda x: x.isNotNull(),
                            )
                        )
                    ),
                ).otherwise(F.array().cast('array<string>')),
            )
            .withColumn(
                'entityIds',
                F.when(
                    F.col('child_entityIds_flat').isNotNull() & (F.size(F.col('child_entityIds_flat')) > 0),
                    F.array_distinct(
                        F.concat(
                            F.coalesce(F.col('original_entityIds'), F.array().cast('array<string>')),
                            F.col('child_entityIds_flat'),
                        )
                    ),
                ).otherwise(F.coalesce(F.col('original_entityIds'), F.array().cast('array<string>'))),
            )
            .select('id', 'parent_id', 'entityIds')
        )

        print('\n=== Result DataFrame (step b output) ===')
        result.show(truncate=False)

        # Verify results
        result_list = result.collect()
        assert len(result_list) == 3

        parent1_row = next(r for r in result_list if r['id'] == 'parent1')
        assert set(parent1_row['entityIds']) == {'entity1', 'entity2', 'entity3'}

        grandparent1_row = next(r for r in result_list if r['id'] == 'grandparent1')
        assert set(grandparent1_row['entityIds']) == {'entity3', 'entity4'}

    finally:
        session.stop()


def test_one_iteration():
    """Test one complete iteration."""
    session = Session(app_name='test_one_iteration')
    spark = session.spark

    try:
        # Input data
        input_data = [
            {'id': 'child1', 'parent_id': 'parent1', 'entityIds': ['entity1', 'entity2']},
            {'id': 'parent1', 'parent_id': 'grandparent1', 'entityIds': ['entity3']},
            {'id': 'grandparent1', 'parent_id': None, 'entityIds': ['entity4']},
        ]

        rows = [Row(**row) for row in input_data]
        current_df = spark.createDataFrame(rows)

        print('\n=== Before iteration ===')
        current_df.show(truncate=False)

        # One iteration
        # Step a0
        existing_ids = [row['id'] for row in current_df.select('id').distinct().collect()]
        children_df = current_df.filter(F.col('parent_id').isNotNull() & F.col('parent_id').isin(existing_ids)).select(
            F.col('id').alias('child_id'),
            F.col('parent_id').alias('child_parent_id'),
            F.col('entityIds').alias('child_entityIds'),
        )

        if children_df.isEmpty():
            print('No children found')
            return

        # Step a
        joined = (
            current_df.alias('df')
            .join(
                children_df.alias('children'),
                on=F.col('df.id') == F.col('children.child_parent_id'),
                how='left',
            )
            .select(
                F.col('df.id'),
                F.col('df.parent_id'),
                F.col('df.entityIds'),
                F.col('children.child_id'),
                F.col('children.child_entityIds').alias('child_entity_ids'),
            )
        )

        # Step b
        result = (
            joined.groupBy('id', 'parent_id')
            .agg(
                F.first('entityIds').alias('original_entityIds'),
                F.collect_list('child_entity_ids').alias('child_entity_ids_list'),
            )
            .withColumn(
                'child_entityIds_flat',
                F.when(
                    F.col('child_entity_ids_list').isNotNull(),
                    F.array_distinct(
                        F.flatten(
                            F.filter(
                                F.col('child_entity_ids_list'),
                                lambda x: x.isNotNull(),
                            )
                        )
                    ),
                ).otherwise(F.array().cast('array<string>')),
            )
            .withColumn(
                'entityIds',
                F.when(
                    F.col('child_entityIds_flat').isNotNull() & (F.size(F.col('child_entityIds_flat')) > 0),
                    F.array_distinct(
                        F.concat(
                            F.coalesce(F.col('original_entityIds'), F.array().cast('array<string>')),
                            F.col('child_entityIds_flat'),
                        )
                    ),
                ).otherwise(F.coalesce(F.col('original_entityIds'), F.array().cast('array<string>'))),
            )
            .select('id', 'parent_id', 'entityIds')
        )

        print('\n=== After one iteration ===')
        result.show(truncate=False)

        # Check what children exist after this iteration
        existing_ids_after = [row['id'] for row in result.select('id').distinct().collect()]
        children_df_after = result.filter(
            F.col('parent_id').isNotNull() & F.col('parent_id').isin(existing_ids_after)
        ).select(
            F.col('id').alias('child_id'),
            F.col('parent_id').alias('child_parent_id'),
            F.col('entityIds').alias('child_entityIds'),
        )

        print('\n=== Children after iteration ===')
        children_df_after.show(truncate=False)
        print(f'Children count after: {children_df_after.count()}')

    finally:
        session.stop()

"""Test suite for PySpark propagate_entity_ids_pyspark function.

Example tests that can be easily copied and edited.
"""

import pytest
from pyspark.sql import Row

from pts.pyspark.common.session import Session
from pts.pyspark.facets.pandasprop import propagate_entity_ids_pyspark


def test_propagate_from_dict():
    """Example test using dictionary input - easy to copy and edit.

    This test demonstrates how to create tests from dictionaries.
    Each dictionary represents a row with id, parent_id, and entityIds.
    """
    session = Session(app_name='test_propagate_pyspark')
    spark = session.spark

    try:
        # Input data as list of dictionaries - easy to edit
        input_data = [
            {'id': 'child1', 'parent_id': 'parent1', 'entityIds': ['entity1', 'entity2']},
            {'id': 'parent1', 'parent_id': 'grandparent1', 'entityIds': ['entity3']},
            {
                'id': 'grandparent1',
                'parent_id': None,  # Root node with no parent
                'entityIds': ['entity4'],
            },
        ]

        # Convert to PySpark DataFrame
        rows = [Row(**row) for row in input_data]
        df = spark.createDataFrame(rows)

        # Run propagation
        result, iterations = propagate_entity_ids_pyspark(df, return_iterations=True)

        # Verify convergence happened before max iterations
        max_iterations = 40
        assert iterations < max_iterations, (
            f'Algorithm should converge before {max_iterations} iterations, but took {iterations}'
        )

        # Convert result to list of dicts for easier assertion
        result_list = result.collect()

        # Find rows by id for assertions
        child1_row = next(r for r in result_list if r.id == 'child1')
        parent1_row = next(r for r in result_list if r.id == 'parent1')
        grandparent1_row = next(r for r in result_list if r.id == 'grandparent1')

        # Child should remain unchanged
        assert set(child1_row.entityIds) == {'entity1', 'entity2'}
        assert child1_row.parent_id == 'parent1'

        # Parent should have its own entityIds plus child's entityIds
        assert set(parent1_row.entityIds) == {'entity1', 'entity2', 'entity3'}
        assert parent1_row.parent_id == 'grandparent1'

        # Grandparent should have all descendant entityIds transitively
        assert set(grandparent1_row.entityIds) == {'entity1', 'entity2', 'entity3', 'entity4'}
        assert grandparent1_row.parent_id is None

    finally:
        session.stop()


def test_propagate_multiple_parents():
    """Test propagation when one id has multiple parents.

    This test demonstrates that one id can have multiple parent_id values
    (multiple rows with same id but different parent_id), and entityIds
    should propagate to all parents.
    """
    session = Session(app_name='test_propagate_pyspark')
    spark = session.spark

    try:
        # Input data as list of dictionaries - easy to edit
        input_data = [
            {'id': 'child1', 'parent_id': 'parent1', 'entityIds': ['entity1', 'entity2']},
            {
                'id': 'child1',
                'parent_id': 'parent2',
                'entityIds': ['entity1', 'entity2'],  # Same id, different parent
            },
            {
                'id': 'parent1',
                'parent_id': None,  # Root node
                'entityIds': ['entity3'],
            },
            {
                'id': 'parent2',
                'parent_id': None,  # Root node
                'entityIds': ['entity4'],
            },
        ]

        # Convert to PySpark DataFrame
        rows = [Row(**row) for row in input_data]
        df = spark.createDataFrame(rows)

        # Run propagation
        result, iterations = propagate_entity_ids_pyspark(df, return_iterations=True)

        # Verify convergence happened before max iterations
        max_iterations = 40
        assert iterations < max_iterations, (
            f'Algorithm should converge before {max_iterations} iterations, but took {iterations}'
        )

        # Convert result to list of dicts for easier assertion
        result_list = result.collect()

        # Find rows by (id, parent_id) for assertions
        child1_parent1_row = next(r for r in result_list if r.id == 'child1' and r.parent_id == 'parent1')
        child1_parent2_row = next(r for r in result_list if r.id == 'child1' and r.parent_id == 'parent2')
        parent1_row = next(r for r in result_list if r.id == 'parent1' and r.parent_id is None)
        parent2_row = next(r for r in result_list if r.id == 'parent2' and r.parent_id is None)

        # Child rows should remain unchanged (same entityIds for both parent relationships)
        assert set(child1_parent1_row.entityIds) == {'entity1', 'entity2'}
        assert set(child1_parent2_row.entityIds) == {'entity1', 'entity2'}

        # Parent1 should have its own entityIds plus child's entityIds
        assert set(parent1_row.entityIds) == {'entity1', 'entity2', 'entity3'}

        # Parent2 should have its own entityIds plus child's entityIds
        assert set(parent2_row.entityIds) == {'entity1', 'entity2', 'entity4'}

    finally:
        session.stop()


def test_propagate_multiple_children():
    """Test propagation when one parent has multiple children.

    This test demonstrates that one parent_id can have multiple children
    (multiple rows with different id but same parent_id), and entityIds
    from all children should propagate to the parent.
    """
    session = Session(app_name='test_propagate_pyspark')
    spark = session.spark

    try:
        # Input data as list of dictionaries - easy to edit
        input_data = [
            {'id': 'child1', 'parent_id': 'parent1', 'entityIds': ['entity1', 'entity2']},
            {'id': 'child2', 'parent_id': 'parent1', 'entityIds': ['entity3', 'entity4']},
            {'id': 'child3', 'parent_id': 'parent1', 'entityIds': ['entity5']},
            {
                'id': 'parent1',
                'parent_id': None,  # Root node
                'entityIds': ['entity6'],
            },
        ]

        # Convert to PySpark DataFrame
        rows = [Row(**row) for row in input_data]
        df = spark.createDataFrame(rows)

        # Run propagation
        result, iterations = propagate_entity_ids_pyspark(df, return_iterations=True)

        # Verify convergence happened before max iterations
        max_iterations = 40
        assert iterations < max_iterations, (
            f'Algorithm should converge before {max_iterations} iterations, but took {iterations}'
        )

        # Convert result to list of dicts for easier assertion
        result_list = result.collect()

        # Find rows by id for assertions
        child1_row = next(r for r in result_list if r.id == 'child1')
        child2_row = next(r for r in result_list if r.id == 'child2')
        child3_row = next(r for r in result_list if r.id == 'child3')
        parent1_row = next(r for r in result_list if r.id == 'parent1' and r.parent_id is None)

        # Children should remain unchanged
        assert set(child1_row.entityIds) == {'entity1', 'entity2'}
        assert set(child2_row.entityIds) == {'entity3', 'entity4'}
        assert set(child3_row.entityIds) == {'entity5'}

        # Parent should have its own entityIds plus all children's entityIds
        assert set(parent1_row.entityIds) == {'entity1', 'entity2', 'entity3', 'entity4', 'entity5', 'entity6'}

    finally:
        session.stop()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

"""Test suite for propagate_entity_ids_to_parents function."""


# from pts.pyspark.facets.target_facets import propagate_entity_ids_to_parents


# COMMENTED OUT: Tests for old propagate_entity_ids_to_parents function
# def test_propagate_single_child_to_parent():
#     """Test basic propagation: one child with one parent."""
#     session = Session(app_name='test_propagate')
#     spark = session.spark
#
#     try:
#         # Create test facets DataFrame
#         test_facets = [
#             Row(
#                 label='Child1',
#                 category='ChEMBL Target Class',
#                 entityIds=['Target1'],
#                 datasourceId=None,
#                 parentId=['Parent1'],
#             ),
#             Row(
#                 label='Parent1',
#                 category='ChEMBL Target Class',
#                 entityIds=['Target2'],
#                 datasourceId=None,
#                 parentId=[],
#             ),
#         ]
#         facets_df = spark.createDataFrame(test_facets, schema=facet_schema)
#
#         # Run propagation
#         result = propagate_entity_ids_to_parents(facets_df)
#
#         # Verify results
#         result_list = result.collect()
#         child_row = next(r for r in result_list if r.label == 'Child1')
#         parent_row = next(r for r in result_list if r.label == 'Parent1')
#
#         # Child should remain unchanged
#         assert set(child_row.entityIds) == {'Target1'}
#         assert child_row.parentId == ['Parent1']
#
#         # Parent should have both its own and child's entityIds
#         assert set(parent_row.entityIds) == {'Target1', 'Target2'}
#         assert parent_row.parentId == []
#
#     finally:
#         session.stop()
#
#
# def test_propagate_multiple_children_to_one_parent():
#     """Test propagation with multiple children to one parent."""
#     session = Session(app_name='test_propagate')
#     spark = session.spark
#
#     try:
#         test_facets = [
#             Row(
#                 label='Child1',
#                 category='ChEMBL Target Class',
#                 entityIds=['Target1'],
#                 datasourceId=None,
#                 parentId=['Parent1'],
#             ),
#             Row(
#                 label='Child2',
#                 category='ChEMBL Target Class',
#                 entityIds=['Target2', 'Target3'],
#                 datasourceId=None,
#                 parentId=['Parent1'],
#             ),
#             Row(
#                 label='Parent1',
#                 category='ChEMBL Target Class',
#                 entityIds=['Target4'],
#                 datasourceId=None,
#                 parentId=[],
#             ),
#         ]
#         facets_df = spark.createDataFrame(test_facets, schema=facet_schema)
#
#         result = propagate_entity_ids_to_parents(facets_df)
#
#         result_list = result.collect()
#         parent_row = next(r for r in result_list if r.label == 'Parent1')
#
#         # Parent should have all entityIds from both children plus its own
#         assert set(parent_row.entityIds) == {'Target1', 'Target2', 'Target3', 'Target4'}
#
#     finally:
#         session.stop()
#
#
# def test_propagate_multi_level_hierarchy():
#     """Test transitive propagation in a multi-level hierarchy (child -> parent -> grandparent).
#
#     The function iteratively propagates entityIds through all hierarchy levels,
#     so grandparents receive entityIds from all descendants (children and grandchildren).
#     """
#     session = Session(app_name='test_propagate')
#     spark = session.spark
#
#     try:
#         test_facets = [
#             Row(
#                 label='Child1',
#                 category='ChEMBL Target Class',
#                 entityIds=['Target1'],
#                 datasourceId=None,
#                 parentId=['Parent1'],
#             ),
#             Row(
#                 label='Parent1',
#                 category='ChEMBL Target Class',
#                 entityIds=['Target2'],
#                 datasourceId=None,
#                 parentId=['Grandparent1'],
#             ),
#             Row(
#                 label='Grandparent1',
#                 category='ChEMBL Target Class',
#                 entityIds=['Target3'],
#                 datasourceId=None,
#                 parentId=[],
#             ),
#         ]
#         facets_df = spark.createDataFrame(test_facets, schema=facet_schema)
#
#         result = propagate_entity_ids_to_parents(facets_df)
#
#         result_list = result.collect()
#         child_row = next(r for r in result_list if r.label == 'Child1')
#         parent_row = next(r for r in result_list if r.label == 'Parent1')
#         grandparent_row = next(r for r in result_list if r.label == 'Grandparent1')
#
#         # Child unchanged
#         assert set(child_row.entityIds) == {'Target1'}
#
#         # Parent gets child's entityIds (first iteration)
#         assert set(parent_row.entityIds) == {'Target1', 'Target2'}
#
#         # Grandparent gets ALL descendant entityIds transitively:
#         # - From Parent1: Target2 (original) + Target1 (from Child1, propagated in iteration 1)
#         # - From itself: Target3
#         # After iteration 1: Parent1 has [Target1, Target2]
#         # After iteration 2: Grandparent1 gets [Target1, Target2] from Parent1 + [Target3] from itself
#         assert set(grandparent_row.entityIds) == {'Target1', 'Target2', 'Target3'}
#
#     finally:
#         session.stop()
#
#
# def test_propagate_respects_category():
#     """Test that propagation only happens within the same category."""
#     session = Session(app_name='test_propagate')
#     spark = session.spark
#
#     try:
#         test_facets = [
#             Row(
#                 label='Child1',
#                 category='ChEMBL Target Class',
#                 entityIds=['Target1'],
#                 datasourceId=None,
#                 parentId=['Parent1'],
#             ),
#             Row(
#                 label='Parent1',
#                 category='ChEMBL Target Class',
#                 entityIds=['Target2'],
#                 datasourceId=None,
#                 parentId=[],
#             ),
#             # Same label but different category - should NOT propagate
#             Row(
#                 label='Parent1',
#                 category='Reactome',
#                 entityIds=['Target3'],
#                 datasourceId=None,
#                 parentId=[],
#             ),
#         ]
#         facets_df = spark.createDataFrame(test_facets, schema=facet_schema)
#
#         result = propagate_entity_ids_to_parents(facets_df)
#
#         result_list = result.collect()
#         chembl_parent = next(r for r in result_list if r.label == 'Parent1' and r.category == 'ChEMBL Target Class')
#         reactome_parent = next(r for r in result_list if r.label == 'Parent1' and r.category == 'Reactome')
#
#         # ChEMBL parent should have propagated entityIds
#         assert set(chembl_parent.entityIds) == {'Target1', 'Target2'}
#
#         # Reactome parent should remain unchanged (different category)
#         assert set(reactome_parent.entityIds) == {'Target3'}
#
#     finally:
#         session.stop()
#
#
# def test_propagate_handles_empty_parentid():
#     """Test that facets with empty parentId arrays are not affected."""
#     session = Session(app_name='test_propagate')
#     spark = session.spark
#
#     try:
#         test_facets = [
#             Row(
#                 label='Child1',
#                 category='ChEMBL Target Class',
#                 entityIds=['Target1'],
#                 datasourceId=None,
#                 parentId=['Parent1'],
#             ),
#             Row(
#                 label='Parent1',
#                 category='ChEMBL Target Class',
#                 entityIds=['Target2'],
#                 datasourceId=None,
#                 parentId=[],
#             ),
#             # Facet with empty parentId - should not propagate
#             Row(
#                 label='Standalone',
#                 category='Approved Symbol',
#                 entityIds=['Target3'],
#                 datasourceId=None,
#                 parentId=[],
#             ),
#         ]
#         facets_df = spark.createDataFrame(test_facets, schema=facet_schema)
#
#         result = propagate_entity_ids_to_parents(facets_df)
#
#         result_list = result.collect()
#         standalone_row = next(r for r in result_list if r.label == 'Standalone')
#
#         # Standalone facet should remain unchanged
#         assert set(standalone_row.entityIds) == {'Target3'}
#         assert standalone_row.parentId == []
#
#     finally:
#         session.stop()
#
#
# def test_propagate_handles_no_hierarchical_facets():
#     """Test that function returns original DataFrame when no hierarchical facets exist."""
#     session = Session(app_name='test_propagate')
#     spark = session.spark
#
#     try:
#         test_facets = [
#             Row(
#                 label='Facet1',
#                 category='Approved Symbol',
#                 entityIds=['Target1'],
#                 datasourceId=None,
#                 parentId=[],
#             ),
#             Row(
#                 label='Facet2',
#                 category='Approved Name',
#                 entityIds=['Target2'],
#                 datasourceId=None,
#                 parentId=[],
#             ),
#         ]
#         facets_df = spark.createDataFrame(test_facets, schema=facet_schema)
#
#         result = propagate_entity_ids_to_parents(facets_df)
#
#         # Should return unchanged DataFrame
#         result_list = result.collect()
#         assert len(result_list) == 2
#         assert all(r.parentId == [] for r in result_list)
#
#     finally:
#         session.stop()
#
#
# def test_propagate_handles_multiple_parents():
#     """Test that a child with multiple parents propagates to all of them."""
#     session = Session(app_name='test_propagate')
#     spark = session.spark
#
#     try:
#         test_facets = [
#             Row(
#                 label='Child1',
#                 category='ChEMBL Target Class',
#                 entityIds=['Target1'],
#                 datasourceId=None,
#                 parentId=['Parent1', 'Parent2'],
#             ),
#             Row(
#                 label='Parent1',
#                 category='ChEMBL Target Class',
#                 entityIds=['Target2'],
#                 datasourceId=None,
#                 parentId=[],
#             ),
#             Row(
#                 label='Parent2',
#                 category='ChEMBL Target Class',
#                 entityIds=['Target3'],
#                 datasourceId=None,
#                 parentId=[],
#             ),
#         ]
#         facets_df = spark.createDataFrame(test_facets, schema=facet_schema)
#
#         result = propagate_entity_ids_to_parents(facets_df)
#
#         result_list = result.collect()
#         parent1_row = next(r for r in result_list if r.label == 'Parent1')
#         parent2_row = next(r for r in result_list if r.label == 'Parent2')
#
#         # Both parents should receive child's entityIds
#         assert set(parent1_row.entityIds) == {'Target1', 'Target2'}
#         assert set(parent2_row.entityIds) == {'Target1', 'Target3'}
#
#     finally:
#         session.stop()
#
#
# def test_propagate_removes_duplicates():
#     """Test that duplicate entityIds are removed when merging."""
#     session = Session(app_name='test_propagate')
#     spark = session.spark
#
#     try:
#         test_facets = [
#             Row(
#                 label='Child1',
#                 category='ChEMBL Target Class',
#                 entityIds=['Target1', 'Target2'],
#                 datasourceId=None,
#                 parentId=['Parent1'],
#             ),
#             Row(
#                 label='Parent1',
#                 category='ChEMBL Target Class',
#                 entityIds=['Target2', 'Target3'],  # Target2 is duplicate
#                 datasourceId=None,
#                 parentId=[],
#             ),
#         ]
#         facets_df = spark.createDataFrame(test_facets, schema=facet_schema)
#
#         result = propagate_entity_ids_to_parents(facets_df)
#
#         result_list = result.collect()
#         parent_row = next(r for r in result_list if r.label == 'Parent1')
#
#         # Should have unique entityIds only
#         assert set(parent_row.entityIds) == {'Target1', 'Target2', 'Target3'}
#         assert len(parent_row.entityIds) == 3  # No duplicates
#
#     finally:
#         session.stop()

if __name__ == '__main__':
    import pytest

    pytest.main([__file__, '-v'])

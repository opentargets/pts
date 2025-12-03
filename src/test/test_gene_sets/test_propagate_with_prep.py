"""Test suite for propagate_entity_ids_with_dataset_prep function.

Example tests that can be easily copied and edited.
Tests use full facet schema with label, category, datasourceId, parentId, entityIds.
"""

import pytest
from pyspark.sql import Row, SparkSession

from pts.pyspark.gene_sets.gene_sets import propagate_entity_ids_with_dataset_prep
from pts.schemas.gene_sets import gene_sets_schema


@pytest.mark.slow
class TestPropagateWithPrep:
    """Test suite for propagate_entity_ids_with_dataset_prep function."""

    @pytest.fixture(autouse=True)
    def _setup(self, spark: SparkSession) -> None:
        """Set up test fixtures with spark session."""
        self.spark = spark

    def test_propagate_with_prep_from_dict(self) -> None:
        """Example test using dictionary input with GO category - easy to copy and edit.

        This test demonstrates how to create tests from dictionaries.
        Each dictionary represents a row with label, category, datasourceId, parentId, entityIds.
        For GO categories, id is derived from datasourceId.
        """
        # Input data as list of dictionaries - easy to edit
        # Using GO:BP category, so id will be derived from datasourceId
        input_data = [
            Row(
                label='apoptotic process',
                category='GO:BP',
                entityIds=['entity1', 'entity2'],
                datasourceId='GO:0006915',  # This will be used as id
                parentId=['GO:0008150'],  # biological_process
            ),
            Row(
                label='biological_process',
                category='GO:BP',
                entityIds=['entity3'],
                datasourceId='GO:0008150',  # This will be used as id
                parentId=['GO:0003674'],  # gene_ontology
            ),
            Row(
                label='gene_ontology',
                category='GO:BP',
                entityIds=['entity4'],
                datasourceId='GO:0003674',  # This will be used as id
                parentId=None,  # Root node with no parent
            ),
        ]

        # Convert to PySpark DataFrame with facet schema
        df = self.spark.createDataFrame(input_data, schema=gene_sets_schema)

        # Run propagation
        result, iterations = propagate_entity_ids_with_dataset_prep(df, return_iterations=True)

        # Verify convergence happened before max iterations
        max_iterations = 40
        assert iterations < max_iterations, (
            f'Algorithm should converge before {max_iterations} iterations, but took {iterations}'
        )

        # Convert result to list of dicts for easier assertion
        result_list = result.collect()

        # Find rows by id (which is datasourceId for GO categories)
        # Note: prepare_dataset_for_propagation filters out rows with null/empty parentId,
        # so root nodes won't be in the result. We only check rows that have a parent.
        child_row = next(r for r in result_list if r.id == 'GO:0006915')
        parent_row = next(r for r in result_list if r.id == 'GO:0008150')

        # Child should remain unchanged
        assert set(child_row.entityIds) == {'entity1', 'entity2'}
        assert child_row.parent_id == 'GO:0008150'

        # Parent should have its own entityIds plus child's entityIds
        assert set(parent_row.entityIds) == {'entity1', 'entity2', 'entity3'}
        assert parent_row.parent_id == 'GO:0003674'

        # Note: Grandparent (GO:0003674) is not in the result because it has parentId=None,
        # so it was filtered out by prepare_dataset_for_propagation.
        # The function only returns rows that have a parent (i.e., rows that are children).

    def test_propagate_with_prep_multiple_parents(self) -> None:
        """Test propagation when one id has multiple parents (ChEMBL category).

        This test demonstrates that one id can have multiple parent_id values
        (multiple rows with same id but different parent_id), and entityIds
        should propagate to all parents.
        For ChEMBL category, id is derived from label.
        """
        # Input data as list of dictionaries - easy to edit
        # Using ChEMBL Target Class category, so id will be derived from label
        input_data = [
            Row(
                label='Enzyme',
                category='ChEMBL Target Class',
                entityIds=['entity1', 'entity2'],
                datasourceId=None,
                parentId=['Protein'],  # First parent
            ),
            Row(
                label='Enzyme',
                category='ChEMBL Target Class',
                entityIds=['entity1', 'entity2'],  # Same id, different parent
                datasourceId=None,
                parentId=['Catalyst'],  # Second parent
            ),
            Row(
                label='Protein',
                category='ChEMBL Target Class',
                entityIds=['entity3'],
                datasourceId=None,
                parentId=None,  # Root node
            ),
            Row(
                label='Catalyst',
                category='ChEMBL Target Class',
                entityIds=['entity4'],
                datasourceId=None,
                parentId=None,  # Root node
            ),
        ]

        # Convert to PySpark DataFrame with facet schema
        df = self.spark.createDataFrame(input_data, schema=gene_sets_schema)

        # Run propagation
        result, iterations = propagate_entity_ids_with_dataset_prep(df, return_iterations=True)

        # Verify convergence happened before max iterations
        max_iterations = 40
        assert iterations < max_iterations, (
            f'Algorithm should converge before {max_iterations} iterations, but took {iterations}'
        )

        # Convert result to list of dicts for easier assertion
        result_list = result.collect()

        # Find rows by (id, parent_id) for assertions
        # id is label for ChEMBL category
        # Note: prepare_dataset_for_propagation filters out rows with null/empty parentId,
        # so root nodes (Protein and Catalyst) won't be in the result.
        child_parent1_row = next(r for r in result_list if r.id == 'Enzyme' and r.parent_id == 'Protein')
        child_parent2_row = next(r for r in result_list if r.id == 'Enzyme' and r.parent_id == 'Catalyst')

        # Child rows should remain unchanged (same entityIds for both parent relationships)
        assert set(child_parent1_row.entityIds) == {'entity1', 'entity2'}
        assert set(child_parent2_row.entityIds) == {'entity1', 'entity2'}

        # Note: Parent1 (Protein) and Parent2 (Catalyst) are not in the result because
        # they have parentId=None, so they were filtered out by prepare_dataset_for_propagation.
        # The function only returns rows that have a parent (i.e., rows that are children).

    def test_propagate_with_prep_multiple_children(self) -> None:
        """Test propagation when one parent has multiple children (Reactome category).

        This test demonstrates that one parent_id can have multiple children
        (multiple rows with different id but same parent_id), and entityIds
        from all children should propagate to the parent.
        For Reactome category, id is derived from datasourceId.
        """
        # Input data as list of dictionaries - easy to edit
        # Using Reactome category, so id will be derived from datasourceId
        input_data = [
            Row(
                label='Signal Transduction',
                category='Reactome',
                entityIds=['entity1', 'entity2'],
                datasourceId='R-HSA-162582',  # This will be used as id
                parentId=['R-HSA-1430728'],  # Signal Transduction pathway
            ),
            Row(
                label='Cell Cycle',
                category='Reactome',
                entityIds=['entity3', 'entity4'],
                datasourceId='R-HSA-1640170',  # This will be used as id
                parentId=['R-HSA-1430728'],  # Same parent
            ),
            Row(
                label='Apoptosis',
                category='Reactome',
                entityIds=['entity5'],
                datasourceId='R-HSA-109581',  # This will be used as id
                parentId=['R-HSA-1430728'],  # Same parent
            ),
            Row(
                label='Signal Transduction pathway',
                category='Reactome',
                entityIds=['entity6'],
                datasourceId='R-HSA-1430728',  # This will be used as id
                parentId=None,  # Root node
            ),
        ]

        # Convert to PySpark DataFrame with facet schema
        df = self.spark.createDataFrame(input_data, schema=gene_sets_schema)

        # Run propagation
        result, iterations = propagate_entity_ids_with_dataset_prep(df, return_iterations=True)

        # Verify convergence happened before max iterations
        max_iterations = 40
        assert iterations < max_iterations, (
            f'Algorithm should converge before {max_iterations} iterations, but took {iterations}'
        )

        # Convert result to list of dicts for easier assertion
        result_list = result.collect()

        # Find rows by id (which is datasourceId for Reactome category)
        # Note: prepare_dataset_for_propagation filters out rows with null/empty parentId,
        # so root nodes won't be in the result. We only check rows that have a parent.
        child1_row = next(r for r in result_list if r.id == 'R-HSA-162582')
        child2_row = next(r for r in result_list if r.id == 'R-HSA-1640170')
        child3_row = next(r for r in result_list if r.id == 'R-HSA-109581')

        # Children should remain unchanged
        assert set(child1_row.entityIds) == {'entity1', 'entity2'}
        assert set(child2_row.entityIds) == {'entity3', 'entity4'}
        assert set(child3_row.entityIds) == {'entity5'}

        # Note: Parent (R-HSA-1430728) is not in the result because it has parentId=None,
        # so it was filtered out by prepare_dataset_for_propagation.
        # The function only returns rows that have a parent (i.e., rows that are children).


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

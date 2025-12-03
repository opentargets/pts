"""Test suite for merge_propagated_entity_ids function."""

import pytest
from pyspark.sql import Row, SparkSession
from pyspark.sql.types import ArrayType, StringType, StructField, StructType

from pts.pyspark.gene_sets.gene_sets import merge_propagated_entity_ids
from pts.schemas.gene_sets import gene_sets_schema


@pytest.mark.slow
class TestMergePropagated:
    """Test suite for merge_propagated_entity_ids function."""

    @pytest.fixture(autouse=True)
    def _setup(self, spark: SparkSession) -> None:
        """Set up test fixtures with spark session."""
        self.spark = spark
        self.propagated_schema = StructType([
            StructField('id', StringType(), False),
            StructField('parent_id', StringType(), True),
            StructField('entityIds', ArrayType(StringType(), containsNull=False), False),
        ])

    def test_merge_propagated_go_category(self) -> None:
        """Test merging propagated entityIds for GO category (uses datasourceId for id)."""
        # Original facets DataFrame
        original_facets = [
            Row(
                label='apoptotic process',
                category='GO:BP',
                entityIds=['ENSG00000141510'],  # Original entityIds
                datasourceId='GO:0006915',
                parentId=['GO:0008150'],
            ),
        ]
        facets_df = self.spark.createDataFrame(original_facets, schema=gene_sets_schema)

        # Propagated DataFrame (after propagation, parent got child's entityIds)
        propagated_data = [
            Row(id='GO:0006915', parent_id='GO:0008150', entityIds=['ENSG00000141510']),  # Child unchanged
            Row(
                id='GO:0008150', parent_id=None, entityIds=['ENSG00000141510', 'ENSG00000012048']
            ),  # Parent got propagated
        ]
        propagated_df = self.spark.createDataFrame(propagated_data, schema=self.propagated_schema)

        # Merge
        result = merge_propagated_entity_ids(facets_df, propagated_df)

        # Verify
        result_list = result.collect()
        assert len(result_list) == 1
        row = result_list[0]

        # Original entityIds should be preserved
        assert row.entityIds == ['ENSG00000141510']
        # entityIdsPropagated should be added
        assert 'entityIdsPropagated' in row.asDict()
        assert set(row.entityIdsPropagated) == {'ENSG00000141510'}
        # All other columns should be preserved
        assert row.label == 'apoptotic process'
        assert row.category == 'GO:BP'
        assert row.datasourceId == 'GO:0006915'

    def test_merge_propagated_chembl_category(self) -> None:
        """Test merging propagated entityIds for ChEMBL category (uses label for id)."""
        # Original facets DataFrame
        original_facets = [
            Row(
                label='Enzyme',
                category='ChEMBL Target Class',
                entityIds=['ENSG00000141510'],  # Original entityIds
                datasourceId=None,
                parentId=['Protein'],
            ),
        ]
        facets_df = self.spark.createDataFrame(original_facets, schema=gene_sets_schema)

        # Propagated DataFrame
        propagated_data = [
            Row(id='Enzyme', parent_id='Protein', entityIds=['ENSG00000141510']),  # Child unchanged
            Row(
                id='Protein', parent_id=None, entityIds=['ENSG00000141510', 'ENSG00000012048']
            ),  # Parent got propagated
        ]
        propagated_df = self.spark.createDataFrame(propagated_data, schema=self.propagated_schema)

        # Merge
        result = merge_propagated_entity_ids(facets_df, propagated_df)

        # Verify
        result_list = result.collect()
        assert len(result_list) == 1
        row = result_list[0]

        # Original entityIds should be preserved
        assert row.entityIds == ['ENSG00000141510']
        # entityIdsPropagated should match propagated value
        assert set(row.entityIdsPropagated) == {'ENSG00000141510'}
        # All other columns should be preserved
        assert row.label == 'Enzyme'
        assert row.category == 'ChEMBL Target Class'

    def test_merge_propagated_multiple_parents(self) -> None:
        """Test merging when a node has multiple parents (multiple rows per id in propagated_df)."""
        # Original facets DataFrame
        original_facets = [
            Row(
                label='Child',
                category='ChEMBL Target Class',
                entityIds=['ENSG00000141510'],
                datasourceId=None,
                parentId=['Parent1', 'Parent2'],  # Multiple parents
            ),
        ]
        facets_df = self.spark.createDataFrame(original_facets, schema=gene_sets_schema)

        # Propagated DataFrame - same id appears multiple times (one per parent)
        propagated_data = [
            Row(id='Child', parent_id='Parent1', entityIds=['ENSG00000141510']),  # Same id, different parent
            Row(id='Child', parent_id='Parent2', entityIds=['ENSG00000141510']),  # Same id, different parent
        ]
        propagated_df = self.spark.createDataFrame(propagated_data, schema=self.propagated_schema)

        # Merge
        result = merge_propagated_entity_ids(facets_df, propagated_df)

        # Verify
        result_list = result.collect()
        assert len(result_list) == 1
        row = result_list[0]

        # Should aggregate correctly (using first() since they're all the same)
        assert set(row.entityIdsPropagated) == {'ENSG00000141510'}
        assert row.entityIds == ['ENSG00000141510']

    def test_merge_propagated_missing_id(self) -> None:
        """Test merging when an id doesn't exist in propagated result (should get empty array)."""
        # Original facets DataFrame
        original_facets = [
            Row(
                label='MissingTerm',
                category='ChEMBL Target Class',
                entityIds=['ENSG00000141510'],
                datasourceId=None,
                parentId=['Parent'],
            ),
        ]
        facets_df = self.spark.createDataFrame(original_facets, schema=gene_sets_schema)

        # Propagated DataFrame - doesn't contain 'MissingTerm'
        propagated_data = [
            Row(id='OtherTerm', parent_id='Parent', entityIds=['ENSG00000012048']),
        ]
        propagated_df = self.spark.createDataFrame(propagated_data, schema=self.propagated_schema)

        # Merge
        result = merge_propagated_entity_ids(facets_df, propagated_df)

        # Verify
        result_list = result.collect()
        assert len(result_list) == 1
        row = result_list[0]

        # Should have empty array for entityIdsPropagated since id wasn't found
        assert row.entityIdsPropagated == []
        # Original entityIds should still be preserved
        assert row.entityIds == ['ENSG00000141510']

    def test_merge_propagated_mixed_categories(self) -> None:
        """Test merging with multiple facets of different categories."""
        # Original facets DataFrame with GO and ChEMBL
        original_facets = [
            Row(
                label='apoptotic process',
                category='GO:BP',
                entityIds=['ENSG00000141510'],
                datasourceId='GO:0006915',
                parentId=['GO:0008150'],
            ),
            Row(
                label='Enzyme',
                category='ChEMBL Target Class',
                entityIds=['ENSG00000012048'],
                datasourceId=None,
                parentId=['Protein'],
            ),
        ]
        facets_df = self.spark.createDataFrame(original_facets, schema=gene_sets_schema)

        # Propagated DataFrame
        propagated_data = [
            Row(id='GO:0006915', parent_id='GO:0008150', entityIds=['ENSG00000141510']),
            Row(id='Enzyme', parent_id='Protein', entityIds=['ENSG00000012048', 'ENSG00000141510']),  # Got propagated
        ]
        propagated_df = self.spark.createDataFrame(propagated_data, schema=self.propagated_schema)

        # Merge
        result = merge_propagated_entity_ids(facets_df, propagated_df)

        # Verify
        result_list = result.collect()
        assert len(result_list) == 2

        # Find rows by category
        go_row = next(r for r in result_list if r.category == 'GO:BP')
        chembl_row = next(r for r in result_list if r.category == 'ChEMBL Target Class')

        # GO row
        assert go_row.entityIds == ['ENSG00000141510']
        assert set(go_row.entityIdsPropagated) == {'ENSG00000141510'}

        # ChEMBL row
        assert chembl_row.entityIds == ['ENSG00000012048']
        assert set(chembl_row.entityIdsPropagated) == {'ENSG00000012048', 'ENSG00000141510'}

    def test_merge_propagated_preserves_all_columns(self) -> None:
        """Test that all original columns are preserved in the result."""
        # Original facets DataFrame
        original_facets = [
            Row(
                label='TestTerm',
                category='GO:MF',
                entityIds=['ENSG00000141510'],
                datasourceId='GO:0003700',
                parentId=['GO:0003674'],
            ),
        ]
        facets_df = self.spark.createDataFrame(original_facets, schema=gene_sets_schema)

        # Propagated DataFrame
        propagated_data = [
            Row(id='GO:0003700', parent_id='GO:0003674', entityIds=['ENSG00000141510']),
        ]
        propagated_df = self.spark.createDataFrame(propagated_data, schema=self.propagated_schema)

        # Merge
        result = merge_propagated_entity_ids(facets_df, propagated_df)

        # Verify schema includes all original columns plus entityIdsPropagated
        schema = result.schema
        field_names = {field.name for field in schema.fields}

        assert 'label' in field_names
        assert 'category' in field_names
        assert 'entityIds' in field_names
        assert 'datasourceId' in field_names
        assert 'parentId' in field_names
        assert 'entityIdsPropagated' in field_names
        assert 'id' not in field_names  # Temporary id column should be dropped

        # Verify data
        result_list = result.collect()
        assert len(result_list) == 1
        row = result_list[0]

        assert row.label == 'TestTerm'
        assert row.category == 'GO:MF'
        assert row.datasourceId == 'GO:0003700'
        assert row.parentId == ['GO:0003674']
        assert row.entityIds == ['ENSG00000141510']
        assert set(row.entityIdsPropagated) == {'ENSG00000141510'}

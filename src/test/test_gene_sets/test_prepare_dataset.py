"""Test suite for prepare_dataset_for_propagation function."""

import pytest
from pyspark.sql import Row, SparkSession

from pts.pyspark.gene_sets.gene_sets import prepare_dataset_for_propagation
from pts.schemas.gene_sets import gene_sets_schema


@pytest.mark.slow
class TestPrepareDataset:
    """Test suite for prepare_dataset_for_propagation function."""

    @pytest.fixture(autouse=True)
    def _setup(self, spark: SparkSession) -> None:
        """Set up test fixtures with spark session."""
        self.spark = spark

    def test_prepare_dataset_go_category(self) -> None:
        """Test that GO categories use datasourceId for id column."""
        test_facets = [
            Row(
                label='apoptotic process',
                category='GO:BP',
                entityIds=['ENSG00000141510'],
                datasourceId='GO:0006915',
                parentId=['GO:0008150'],  # biological_process
            ),
        ]
        facets_df = self.spark.createDataFrame(test_facets, schema=gene_sets_schema)

        result = prepare_dataset_for_propagation(facets_df)

        result_list = result.collect()
        assert len(result_list) == 1
        row = result_list[0]

        # GO category should use datasourceId for id
        assert row.id == 'GO:0006915'
        assert row.parent_id == 'GO:0008150'
        assert row.entityIds == ['ENSG00000141510']

    def test_prepare_dataset_reactome_category(self) -> None:
        """Test that Reactome category uses datasourceId for id column."""
        test_facets = [
            Row(
                label='Signal Transduction',
                category='Reactome',
                entityIds=['ENSG00000141510', 'ENSG00000012048'],
                datasourceId='R-HSA-162582',
                parentId=['R-HSA-1430728'],
            ),
        ]
        facets_df = self.spark.createDataFrame(test_facets, schema=gene_sets_schema)

        result = prepare_dataset_for_propagation(facets_df)

        result_list = result.collect()
        assert len(result_list) == 1
        row = result_list[0]

        # Reactome category should use datasourceId for id
        assert row.id == 'R-HSA-162582'
        assert row.parent_id == 'R-HSA-1430728'
        assert row.entityIds == ['ENSG00000141510', 'ENSG00000012048']

    def test_prepare_dataset_chembl_category(self) -> None:
        """Test that ChEMBL category uses label for id column."""
        test_facets = [
            Row(
                label='Enzyme',
                category='ChEMBL Target Class',
                entityIds=['ENSG00000141510'],
                datasourceId=None,
                parentId=['Protein'],
            ),
        ]
        facets_df = self.spark.createDataFrame(test_facets, schema=gene_sets_schema)

        result = prepare_dataset_for_propagation(facets_df)

        result_list = result.collect()
        assert len(result_list) == 1
        row = result_list[0]

        # ChEMBL category should use label for id
        assert row.id == 'Enzyme'
        assert row.parent_id == 'Protein'
        assert row.entityIds == ['ENSG00000141510']

    def test_prepare_dataset_multiple_parent_ids(self) -> None:
        """Test that multiple parentIds are exploded correctly."""
        test_facets = [
            Row(
                label='Child',
                category='ChEMBL Target Class',
                entityIds=['ENSG00000141510'],
                datasourceId=None,
                parentId=['Parent1', 'Parent2'],
            ),
        ]
        facets_df = self.spark.createDataFrame(test_facets, schema=gene_sets_schema)

        result = prepare_dataset_for_propagation(facets_df)

        result_list = result.collect()
        assert len(result_list) == 2

        # Should have two rows, one for each parent
        parent_ids = {row.parent_id for row in result_list}
        assert parent_ids == {'Parent1', 'Parent2'}

        # Both rows should have same id and entityIds
        for row in result_list:
            assert row.id == 'Child'
            assert row.entityIds == ['ENSG00000141510']

    def test_prepare_dataset_empty_parent_id(self) -> None:
        """Test that rows with empty parentId arrays are filtered out."""
        test_facets = [
            Row(
                label='Root',
                category='ChEMBL Target Class',
                entityIds=['ENSG00000141510'],
                datasourceId=None,
                parentId=[],
            ),
            Row(
                label='Child',
                category='ChEMBL Target Class',
                entityIds=['ENSG00000012048'],
                datasourceId=None,
                parentId=['Root'],
            ),
        ]
        facets_df = self.spark.createDataFrame(test_facets, schema=gene_sets_schema)

        result = prepare_dataset_for_propagation(facets_df)

        result_list = result.collect()
        # Only the row with non-empty parentId should be in result
        assert len(result_list) == 1
        row = result_list[0]
        assert row.id == 'Child'
        assert row.parent_id == 'Root'

    def test_prepare_dataset_null_parent_id(self) -> None:
        """Test that rows with null parentId are filtered out."""
        test_facets = [
            Row(
                label='Root',
                category='ChEMBL Target Class',
                entityIds=['ENSG00000141510'],
                datasourceId=None,
                parentId=None,
            ),
            Row(
                label='Child',
                category='ChEMBL Target Class',
                entityIds=['ENSG00000012048'],
                datasourceId=None,
                parentId=['Root'],
            ),
        ]
        facets_df = self.spark.createDataFrame(test_facets, schema=gene_sets_schema)

        result = prepare_dataset_for_propagation(facets_df)

        result_list = result.collect()
        # Only the row with non-null parentId should be in result
        assert len(result_list) == 1
        row = result_list[0]
        assert row.id == 'Child'
        assert row.parent_id == 'Root'

    def test_prepare_dataset_mixed_categories(self) -> None:
        """Test with mixed categories (GO and ChEMBL)."""
        test_facets = [
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
        facets_df = self.spark.createDataFrame(test_facets, schema=gene_sets_schema)

        result = prepare_dataset_for_propagation(facets_df)

        result_list = result.collect()
        assert len(result_list) == 2

        # Find rows by parent_id
        go_row = next(r for r in result_list if r.parent_id == 'GO:0008150')
        chembl_row = next(r for r in result_list if r.parent_id == 'Protein')

        # GO should use datasourceId
        assert go_row.id == 'GO:0006915'
        assert go_row.entityIds == ['ENSG00000141510']

        # ChEMBL should use label
        assert chembl_row.id == 'Enzyme'
        assert chembl_row.entityIds == ['ENSG00000012048']

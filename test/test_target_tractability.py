"""Tests for the target_tractability PySpark module."""

from pyspark.sql import Row
from pyspark.sql.types import StringType, StructField, StructType

from pts.pyspark.target_tractability import _build_tractability

# ---------------------------------------------------------------------------
# Shared schemas and helpers
# ---------------------------------------------------------------------------

TARGET_SCHEMA = StructType([StructField('id', StringType())])

# Minimal tractability TSV row: ensembl_gene_id + two bucket columns
TRACT_SCHEMA = StructType([
    StructField('ensembl_gene_id', StringType()),
    StructField('SM_B1_existence_sm_tractability_bucket', StringType()),
    StructField('AB_B1_existence_ab_tractability_bucket', StringType()),
])


def _target(ensg_id):
    return Row(id=ensg_id)


def _tract_row(ensg_id, sm_b1='0', ab_b1='0'):
    return Row(
        ensembl_gene_id=ensg_id,
        SM_B1_existence_sm_tractability_bucket=sm_b1,
        AB_B1_existence_ab_tractability_bucket=ab_b1,
    )


# ---------------------------------------------------------------------------
# _build_tractability
# ---------------------------------------------------------------------------


def test_build_tractability_output_columns(spark):
    """Output has exactly targetId and tractability columns."""
    rows = [_tract_row('ENSG00000001')]
    targets = spark.createDataFrame([_target('ENSG00000001')], TARGET_SCHEMA)
    result = _build_tractability(spark.createDataFrame(rows, TRACT_SCHEMA), targets)
    assert set(result.columns) == {'targetId', 'tractability'}


def test_build_tractability_struct_fields(spark):
    """Each tractability element has modality, id, and value fields."""
    rows = [_tract_row('ENSG00000001', sm_b1='1')]
    targets = spark.createDataFrame([_target('ENSG00000001')], TARGET_SCHEMA)
    result = _build_tractability(spark.createDataFrame(rows, TRACT_SCHEMA), targets)
    elem = result.first().tractability[0]
    assert hasattr(elem, 'modality')
    assert hasattr(elem, 'id')
    assert hasattr(elem, 'value')


def test_build_tractability_value_true_when_bucket_is_1(spark):
    """value is True when the bucket column contains '1'."""
    rows = [_tract_row('ENSG00000001', sm_b1='1')]
    targets = spark.createDataFrame([_target('ENSG00000001')], TARGET_SCHEMA)
    result = _build_tractability(spark.createDataFrame(rows, TRACT_SCHEMA), targets)
    sm_elem = next(e for e in result.first().tractability if e.modality == 'SM')
    assert sm_elem.value is True


def test_build_tractability_value_false_when_bucket_is_0(spark):
    """value is False when the bucket column contains '0'."""
    rows = [_tract_row('ENSG00000001', sm_b1='0')]
    targets = spark.createDataFrame([_target('ENSG00000001')], TARGET_SCHEMA)
    result = _build_tractability(spark.createDataFrame(rows, TRACT_SCHEMA), targets)
    sm_elem = next(e for e in result.first().tractability if e.modality == 'SM')
    assert sm_elem.value is False


def test_build_tractability_drops_unknown_target(spark):
    """Rows whose ensembl_gene_id is not in the target universe are dropped."""
    rows = [_tract_row('ENSG99999999')]
    targets = spark.createDataFrame([_target('ENSG00000001')], TARGET_SCHEMA)
    result = _build_tractability(spark.createDataFrame(rows, TRACT_SCHEMA), targets)
    assert result.count() == 0


def test_build_tractability_keeps_known_target(spark):
    """Rows whose ensembl_gene_id is in the target universe are retained."""
    rows = [_tract_row('ENSG00000001'), _tract_row('ENSG99999999')]
    targets = spark.createDataFrame([_target('ENSG00000001')], TARGET_SCHEMA)
    result = _build_tractability(spark.createDataFrame(rows, TRACT_SCHEMA), targets)
    assert result.count() == 1
    assert result.first().targetId == 'ENSG00000001'


def test_build_tractability_modality_extracted_from_column_name(spark):
    """Modality label is the first underscore-delimited part of the column name."""
    rows = [_tract_row('ENSG00000001')]
    targets = spark.createDataFrame([_target('ENSG00000001')], TARGET_SCHEMA)
    result = _build_tractability(spark.createDataFrame(rows, TRACT_SCHEMA), targets)
    modalities = {e.modality for e in result.first().tractability}
    assert modalities == {'SM', 'AB'}


def test_build_tractability_bucket_id_extracted_from_column_name(spark):
    """Bucket id label is the last underscore-delimited part of the column name."""
    rows = [_tract_row('ENSG00000001')]
    targets = spark.createDataFrame([_target('ENSG00000001')], TARGET_SCHEMA)
    result = _build_tractability(spark.createDataFrame(rows, TRACT_SCHEMA), targets)
    ids = {e.id for e in result.first().tractability}
    assert 'bucket' in ids

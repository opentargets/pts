"""Tests for the chembl_molecule module."""

import pytest
from pyspark.sql import Row
from pyspark.sql.types import (
    ArrayType,
    StringType,
    StructField,
    StructType,
)

from pts.pyspark.chembl_molecule import _molecule_preprocess, process_molecules

# --- Schemas matching the raw ChEMBL molecule input ---

MOLECULE_STRUCTURES = StructType([
    StructField('canonical_smiles', StringType()),
    StructField('standard_inchi_key', StringType()),
    StructField('molfile', StringType()),
])

MOLECULE_HIERARCHY = StructType([
    StructField('parent_chembl_id', StringType()),
])

CROSS_REFERENCE = StructType([
    StructField('xref_id', StringType()),
    StructField('xref_src', StringType()),
])

MOLECULE_SYNONYM = StructType([
    StructField('molecule_synonym', StringType()),
    StructField('syn_type', StringType()),
])

RAW_MOLECULE_SCHEMA = StructType([
    StructField('molecule_chembl_id', StringType()),
    StructField('molecule_structures', MOLECULE_STRUCTURES),
    StructField('molecule_type', StringType()),
    StructField('pref_name', StringType()),
    StructField('cross_references', ArrayType(CROSS_REFERENCE)),
    StructField('molecule_hierarchy', MOLECULE_HIERARCHY),
    StructField('molecule_synonyms', ArrayType(MOLECULE_SYNONYM)),
])

# drugbank lookup as already renamed inside process_molecules
DRUGBANK_SCHEMA = StructType([
    StructField('id', StringType()),
    StructField('drugbank_id', StringType()),
])

# raw drugbank lookup with ChEMBL's source column names
RAW_DRUGBANK_SCHEMA = StructType([
    StructField("From src:'1'", StringType()),
    StructField("To src:'2'", StringType()),
])

# A short but structurally valid MDL molfile (single carbon atom).
SAMPLE_MOLFILE = (
    '\n     RDKit          2D\n\n'
    '  1  0  0  0  0  0  0  0  0  0999 V2000\n'
    '    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n'
    'M  END\n'
)


# --- Fixtures ---


@pytest.fixture(scope='module')
def raw_molecule_df(spark):
    """Raw ChEMBL molecule rows: one with a molfile, one without."""
    data = [
        Row(
            molecule_chembl_id='CHEMBL1',
            molecule_structures=Row(
                canonical_smiles='C',
                standard_inchi_key='INCHI1',
                molfile=SAMPLE_MOLFILE,
            ),
            molecule_type='Small molecule',
            pref_name='Drug A',
            cross_references=[],
            molecule_hierarchy=Row(parent_chembl_id='CHEMBL1'),
            molecule_synonyms=[],
        ),
        Row(
            molecule_chembl_id='CHEMBL2',
            molecule_structures=Row(
                canonical_smiles=None,
                standard_inchi_key=None,
                molfile=None,
            ),
            molecule_type='Antibody',
            pref_name='Drug B',
            cross_references=[],
            molecule_hierarchy=Row(parent_chembl_id='CHEMBL2'),
            molecule_synonyms=[],
        ),
    ]
    return spark.createDataFrame(data, schema=RAW_MOLECULE_SCHEMA)


@pytest.fixture(scope='module')
def drugbank_df(spark):
    """Renamed drugbank lookup as consumed by _molecule_preprocess."""
    return spark.createDataFrame([], schema=DRUGBANK_SCHEMA)


@pytest.fixture(scope='module')
def raw_drugbank_df(spark):
    """Raw drugbank lookup with ChEMBL's source column names."""
    return spark.createDataFrame([], schema=RAW_DRUGBANK_SCHEMA)


# --- Tests for _molecule_preprocess ---


class TestMoleculePreprocess:
    def test_molfile_extracted(self, raw_molecule_df, drugbank_df):
        """molfile is pulled verbatim from molecule_structures.molfile."""
        result = _molecule_preprocess(raw_molecule_df, drugbank_df)
        rows = {r['id']: r['molfile'] for r in result.collect()}
        assert rows['CHEMBL1'] == SAMPLE_MOLFILE

    def test_molfile_null_when_absent(self, raw_molecule_df, drugbank_df):
        """molfile is null when the source molecule has no molfile."""
        result = _molecule_preprocess(raw_molecule_df, drugbank_df)
        rows = {r['id']: r['molfile'] for r in result.collect()}
        assert rows['CHEMBL2'] is None

    def test_molfile_is_string_column(self, raw_molecule_df, drugbank_df):
        """molfile is exposed as a string column."""
        result = _molecule_preprocess(raw_molecule_df, drugbank_df)
        assert result.schema['molfile'].dataType == StringType()


# --- Tests for process_molecules ---


class TestProcessMolecules:
    def test_molfile_preserved(self, raw_molecule_df, raw_drugbank_df):
        """molfile survives process_molecules into the output."""
        result = process_molecules(raw_molecule_df, raw_drugbank_df)
        rows = {r['id']: r['molfile'] for r in result.collect()}
        assert rows['CHEMBL1'] == SAMPLE_MOLFILE
        assert rows['CHEMBL2'] is None

    def test_row_count_unchanged(self, raw_molecule_df, raw_drugbank_df):
        """Adding molfile does not change the row count."""
        result = process_molecules(raw_molecule_df, raw_drugbank_df)
        assert result.count() == raw_molecule_df.count()

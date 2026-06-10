"""Tests for the chembl_molecule module."""

import json

import pyspark.sql.functions as f
import pytest
from pyspark.sql import Row
from pyspark.sql.types import (
    ArrayType,
    StringType,
    StructField,
    StructType,
)

from pts.pyspark.chembl_molecule import _molecule_preprocess, process_molecules

LABEL_SOURCE_SCHEMA_T = ArrayType(StructType([
    StructField('label', StringType()),
    StructField('source', StringType()),
]))

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

# A short but structurally valid MDL molblock (single carbon atom), terminated
# by the `M  END` line. This is what PTS should emit.
SAMPLE_MOLBLOCK = (
    '\n     RDKit          2D\n\n'
    '  1  0  0  0  0  0  0  0  0  0999 V2000\n'
    '    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n'
    'M  END\n'
)

# ChEMBL ships `molfile` as a full SD-file record: the molblock plus appended
# SDF property tags. PTS truncates this back to the bare molblock.
SAMPLE_MOLFILE = SAMPLE_MOLBLOCK + (
    '> <chembl_id>\nCHEMBL1\n\n'
    '> <chembl_pref_name>\nDRUG A\n\n'
    '$$$$\n'
)

# A molfile-shaped string with no `M  END` terminator. PTS has nothing to
# truncate here, so it must pass through unchanged.
MOLFILE_NO_TERMINATOR = 'malformed molfile content\nwith no terminator line\n'


# --- Fixtures ---


@pytest.fixture(scope='module')
def raw_molecule_df(spark):
    """Raw ChEMBL molecule rows: an SD-file molfile, a missing one, a malformed one."""
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
        Row(
            molecule_chembl_id='CHEMBL3',
            molecule_structures=Row(
                canonical_smiles='CC',
                standard_inchi_key='INCHI3',
                molfile=MOLFILE_NO_TERMINATOR,
            ),
            molecule_type='Small molecule',
            pref_name='Drug C',
            cross_references=[],
            molecule_hierarchy=Row(parent_chembl_id='CHEMBL3'),
            molecule_synonyms=[],
        ),
    ]
    return spark.createDataFrame(data, schema=RAW_MOLECULE_SCHEMA)


# Two drugbank fixtures because the two entry points expect different shapes:
# _molecule_preprocess takes the already-renamed lookup (id, drugbank_id), while
# process_molecules takes the raw lookup and renames the columns itself.
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
    def test_molblock_truncated_at_m_end(self, raw_molecule_df, drugbank_df):
        """molblock is the source molfile truncated at `M  END`."""
        result = _molecule_preprocess(raw_molecule_df, drugbank_df)
        rows = {r['id']: r['molblock'] for r in result.collect()}
        assert rows['CHEMBL1'] == SAMPLE_MOLBLOCK

    def test_molblock_sdf_tags_stripped(self, raw_molecule_df, drugbank_df):
        """The SDF property tags appended after `M  END` are removed."""
        result = _molecule_preprocess(raw_molecule_df, drugbank_df)
        molblock = {r['id']: r['molblock'] for r in result.collect()}['CHEMBL1']
        assert molblock.endswith('M  END\n')
        assert '> <chembl_id>' not in molblock
        assert '$$$$' not in molblock

    def test_molblock_null_when_molfile_absent(self, raw_molecule_df, drugbank_df):
        """molblock is null when the source molecule has no molfile."""
        result = _molecule_preprocess(raw_molecule_df, drugbank_df)
        rows = {r['id']: r['molblock'] for r in result.collect()}
        assert rows['CHEMBL2'] is None

    def test_molfile_without_terminator_passed_through(self, raw_molecule_df, drugbank_df):
        """A source molfile with no `M  END` terminator is left unchanged."""
        result = _molecule_preprocess(raw_molecule_df, drugbank_df)
        rows = {r['id']: r['molblock'] for r in result.collect()}
        assert rows['CHEMBL3'] == MOLFILE_NO_TERMINATOR

    def test_molblock_is_string_column(self, raw_molecule_df, drugbank_df):
        """molblock is exposed as a string column."""
        result = _molecule_preprocess(raw_molecule_df, drugbank_df)
        assert result.schema['molblock'].dataType == StringType()


# --- Tests for process_molecules ---


class TestProcessMolecules:
    def test_molblock_preserved(self, raw_molecule_df, raw_drugbank_df):
        """The truncated molblock survives process_molecules into the output."""
        result = process_molecules(raw_molecule_df, raw_drugbank_df)
        rows = {r['id']: r['molblock'] for r in result.collect()}
        assert rows['CHEMBL1'] == SAMPLE_MOLBLOCK
        assert rows['CHEMBL2'] is None

    def test_row_count_unchanged(self, raw_molecule_df, raw_drugbank_df):
        """Adding molblock does not change the row count."""
        result = process_molecules(raw_molecule_df, raw_drugbank_df)
        assert result.count() == raw_molecule_df.count()


class TestSynonymStructs:
    def test_synonyms_are_label_source_structs(self, spark, raw_drugbank_df):
        """ChEMBL synonyms become {label, source:'ChEMBL'} structs, sorted."""
        data = [
            Row(
                molecule_chembl_id='CHEMBL10',
                molecule_structures=Row(canonical_smiles=None, standard_inchi_key=None, molfile=None),
                molecule_type='Small molecule',
                pref_name='Aspirin',
                cross_references=[],
                molecule_hierarchy=Row(parent_chembl_id='CHEMBL10'),
                molecule_synonyms=[
                    Row(molecule_synonym='ASA', syn_type='OTHER'),
                    Row(molecule_synonym='Bayer', syn_type='TRADE_NAME'),
                ],
            ),
        ]
        df = spark.createDataFrame(data, schema=RAW_MOLECULE_SCHEMA)
        row = {r['id']: r for r in process_molecules(df, raw_drugbank_df).collect()}['CHEMBL10']
        assert [(s['label'], s['source']) for s in row['synonyms']] == [('ASA', 'ChEMBL')]
        assert [(t['label'], t['source']) for t in row['tradeNames']] == [('Bayer', 'ChEMBL')]

    def test_empty_synonyms_are_empty_struct_array(self, raw_molecule_df, raw_drugbank_df):
        """Molecules with no synonyms get an empty (not null) struct array."""
        row = {r['id']: r for r in process_molecules(raw_molecule_df, raw_drugbank_df).collect()}['CHEMBL1']
        assert row['synonyms'] == []
        assert row['tradeNames'] == []

    def test_synonyms_schema_is_struct(self, raw_molecule_df, raw_drugbank_df):
        """synonyms column type is array<struct<label,source>>."""
        result = process_molecules(raw_molecule_df, raw_drugbank_df)
        field = result.schema['synonyms'].dataType
        assert isinstance(field, ArrayType)
        assert {f.name for f in field.elementType.fields} == {'label', 'source'}


class TestNormalizeName:
    def test_normalization(self, spark):
        from pts.pyspark.chembl_molecule import _normalize_name
        data = [
            Row(raw='  Revlimid®  '),
            Row(raw='G  CSF'),
            Row(raw='Aspirin™'),
        ]
        df = spark.createDataFrame(data, StructType([StructField('raw', StringType())]))
        out = {r['raw']: r['norm'] for r in df.withColumn('norm', _normalize_name(f.col('raw'))).collect()}
        assert out['  Revlimid®  '] == 'revlimid'
        assert out['G  CSF'] == 'g csf'
        assert out['Aspirin™'] == 'aspirin'


class TestParseAactBatch:
    def _batch_df(self, spark, text_payload, custom_id='NCT01'):
        outer_schema = StructType([
            StructField('custom_id', StringType()),
            StructField('response', StructType([
                StructField('body', StructType([
                    StructField('output', ArrayType(StructType([
                        StructField('type', StringType()),
                        StructField('content', ArrayType(StructType([
                            StructField('text', StringType()),
                        ]))),
                    ]))),
                ])),
            ])),
        ])
        data = [Row(
            custom_id=custom_id,
            response=Row(body=Row(output=[
                Row(type='message', content=[Row(text=text_payload)]),
            ])),
        )]
        return spark.createDataFrame(data, outer_schema)

    def test_parse_extracts_all_roles(self, spark):
        from pts.pyspark.chembl_molecule import _parse_aact_batch
        payload = json.dumps({
            'investigated_drugs': [{'drug': 'Lenalidomide', 'synonyms': ['Revlimid', 'CC-5013']}],
            'comparator_drugs': [{'drug': 'Dexamethasone', 'synonyms': []}],
            'supportive_drugs': [{'drug': 'Filgrastim', 'synonyms': ['G-CSF']}],
        })
        out = _parse_aact_batch(self._batch_df(spark, payload)).collect()
        member_sets = [set(r['members']) for r in out]
        assert {'cc-5013', 'lenalidomide', 'revlimid'} in member_sets
        assert {'filgrastim', 'g-csf'} in member_sets
        assert {'dexamethasone'} in member_sets
        assert all(r['nct_id'] == 'NCT01' for r in out)

    def test_malformed_json_dropped(self, spark):
        from pts.pyspark.chembl_molecule import _parse_aact_batch
        out = _parse_aact_batch(self._batch_df(spark, 'not-valid-json')).collect()
        assert out == []


class TestChemblIndexes:
    def _mol_df(self, spark):
        schema = StructType([
            StructField('id', StringType()),
            StructField('name', StringType()),
            StructField('synonyms', LABEL_SOURCE_SCHEMA_T),
            StructField('tradeNames', LABEL_SOURCE_SCHEMA_T),
            StructField('parentId', StringType()),
            StructField('childChemblIds', ArrayType(StringType())),
        ])
        data = [
            Row(id='CHEMBL1', name='Filgrastim',
                synonyms=[Row(label='Neupogen-syn', source='ChEMBL')],
                tradeNames=[Row(label='Neupogen', source='ChEMBL')],
                parentId=None, childChemblIds=['CHEMBL2']),
            Row(id='CHEMBL9', name='Aspirin component of FOLFOX',
                synonyms=[Row(label='ingredient X COMPONENT OF FOLFOX', source='ChEMBL')],
                tradeNames=[], parentId=None, childChemblIds=[]),
            Row(id='CHEMBL2', name='Sub', synonyms=[], tradeNames=[],
                parentId='CHEMBL1', childChemblIds=[]),
        ]
        return spark.createDataFrame(data, schema)

    def test_name_index_covers_name_syn_trade(self, spark):
        from pts.pyspark.chembl_molecule import _build_chembl_indexes
        name_idx, _regimen, _pc = _build_chembl_indexes(self._mol_df(spark))
        got = {r['name_norm']: set(r['ids']) for r in name_idx.collect()}
        assert got['filgrastim'] == {'CHEMBL1'}
        assert got['neupogen'] == {'CHEMBL1'}
        assert got['neupogen-syn'] == {'CHEMBL1'}

    def test_regimen_index_extracts_regimen(self, spark):
        from pts.pyspark.chembl_molecule import _build_chembl_indexes
        _name, regimen_idx, _pc = _build_chembl_indexes(self._mol_df(spark))
        got = {r['regimen_norm']: set(r['ids']) for r in regimen_idx.collect()}
        assert got['folfox'] == {'CHEMBL9'}

    def test_parent_child_includes_children(self, spark):
        from pts.pyspark.chembl_molecule import _build_chembl_indexes
        _name, _regimen, pc = _build_chembl_indexes(self._mol_df(spark))
        got = {r['id']: set(r['related']) for r in pc.collect()}
        assert 'CHEMBL2' in got['CHEMBL1']
        assert 'CHEMBL1' in got['CHEMBL2']


class TestAnchorCandidates:
    def test_synonym_anchors_novel_candidate(self, spark):
        from pts.pyspark.chembl_molecule import _anchor_candidates
        entries = spark.createDataFrame(
            [Row(nct_id='NCT1', members=['filgrastim', 'g-csf'])],
            StructType([StructField('nct_id', StringType()), StructField('members', ArrayType(StringType()))]),
        )
        name_index = spark.createDataFrame(
            [Row(name_norm='filgrastim', ids=['CHEMBL1'])],
            StructType([StructField('name_norm', StringType()), StructField('ids', ArrayType(StringType()))]),
        )
        pc = spark.createDataFrame(
            [Row(id='CHEMBL1', related=[])],
            StructType([StructField('id', StringType()), StructField('related', ArrayType(StringType()))]),
        )
        out = _anchor_candidates(entries, name_index, pc).collect()
        rows = {(r['id'], r['candidate'], r['status']) for r in out}
        assert ('CHEMBL1', 'g-csf', 'NOVEL') in rows

    def test_over_ambiguous_member_skipped(self, spark):
        from pts.pyspark.chembl_molecule import _anchor_candidates
        entries = spark.createDataFrame(
            [Row(nct_id='NCT1', members=['ssri', 'fluoxetine'])],
            StructType([StructField('nct_id', StringType()), StructField('members', ArrayType(StringType()))]),
        )
        # 'ssri' resolves to 11 molecules -> entry must not anchor through it
        name_index = spark.createDataFrame(
            [Row(name_norm='ssri', ids=[f'CHEMBL{i}' for i in range(11)])],
            StructType([StructField('name_norm', StringType()), StructField('ids', ArrayType(StringType()))]),
        )
        pc = spark.createDataFrame(
            [], StructType([StructField('id', StringType()), StructField('related', ArrayType(StringType()))]),
        )
        out = _anchor_candidates(entries, name_index, pc).collect()
        assert out == []

    def test_conflict_status(self, spark):
        from pts.pyspark.chembl_molecule import _anchor_candidates
        # entry anchors CHEMBL1 (via 'filgrastim'); 'aspirin' resolves to unrelated CHEMBL5 -> CONFLICT for CHEMBL1
        entries = spark.createDataFrame(
            [Row(nct_id='NCT1', members=['filgrastim', 'aspirin'])],
            StructType([StructField('nct_id', StringType()), StructField('members', ArrayType(StringType()))]),
        )
        name_index = spark.createDataFrame(
            [Row(name_norm='filgrastim', ids=['CHEMBL1']), Row(name_norm='aspirin', ids=['CHEMBL5'])],
            StructType([StructField('name_norm', StringType()), StructField('ids', ArrayType(StringType()))]),
        )
        pc = spark.createDataFrame(
            [Row(id='CHEMBL1', related=[])],
            StructType([StructField('id', StringType()), StructField('related', ArrayType(StringType()))]),
        )
        out = {(r['id'], r['candidate'], r['status']) for r in _anchor_candidates(entries, name_index, pc).collect()}
        assert ('CHEMBL1', 'aspirin', 'CONFLICT') in out

    def test_parent_child_status(self, spark):
        from pts.pyspark.chembl_molecule import _anchor_candidates
        # entry anchors CHEMBL1; 'pegfilgrastim' resolves to CHEMBL2 which is a child of CHEMBL1 -> PARENT_CHILD
        entries = spark.createDataFrame(
            [Row(nct_id='NCT1', members=['filgrastim', 'pegfilgrastim'])],
            StructType([StructField('nct_id', StringType()), StructField('members', ArrayType(StringType()))]),
        )
        name_index = spark.createDataFrame(
            [Row(name_norm='filgrastim', ids=['CHEMBL1']), Row(name_norm='pegfilgrastim', ids=['CHEMBL2'])],
            StructType([StructField('name_norm', StringType()), StructField('ids', ArrayType(StringType()))]),
        )
        pc = spark.createDataFrame(
            [Row(id='CHEMBL1', related=['CHEMBL2'])],
            StructType([StructField('id', StringType()), StructField('related', ArrayType(StringType()))]),
        )
        out = {(r['id'], r['candidate'], r['status']) for r in _anchor_candidates(entries, name_index, pc).collect()}
        assert ('CHEMBL1', 'pegfilgrastim', 'PARENT_CHILD') in out

    def test_exactly_cap_is_allowed(self, spark):
        from pts.pyspark.chembl_molecule import _anchor_candidates
        entries = spark.createDataFrame(
            [Row(nct_id='NCT1', members=['generic', 'g-csf'])],
            StructType([StructField('nct_id', StringType()), StructField('members', ArrayType(StringType()))]),
        )
        name_index = spark.createDataFrame(
            [Row(name_norm='generic', ids=[f'CHEMBL{i}' for i in range(10)])],
            StructType([StructField('name_norm', StringType()), StructField('ids', ArrayType(StringType()))]),
        )
        pc = spark.createDataFrame(
            [], StructType([StructField('id', StringType()), StructField('related', ArrayType(StringType()))]),
        )
        # 10 == cap -> entry NOT poisoned; 'g-csf' (unresolved) is a NOVEL candidate for each of the 10
        out = _anchor_candidates(entries, name_index, pc).collect()
        assert out != []

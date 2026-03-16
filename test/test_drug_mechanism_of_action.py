"""Tests for the drug_mechanism_of_action module."""

import pytest
from pyspark.sql import Row
from pyspark.sql.types import ArrayType, StringType, StructField, StructType

from pts.pyspark.drug_mechanism_of_action import _chembl_target, process_mechanism_of_action


TARGET_SCHEMA = StructType([
    StructField('target_chembl_id', StringType()),
    StructField('pref_name', StringType()),
    StructField('target_type', StringType()),
    StructField('target_components', ArrayType(StructType([
        StructField('accession', StringType()),
    ]))),
])

GENE_SCHEMA = StructType([
    StructField('id', StringType()),
    StructField('uniprot_trembl', ArrayType(StringType())),
    StructField('uniprot_swissprot', ArrayType(StringType())),
])


def test_chembl_target_resolves_swissprot(spark):
    """A single-protein target with a matching swissprot ID produces a non-empty targets list."""
    target_df = spark.createDataFrame(
        [Row(
            target_chembl_id='CHEMBL2111394',
            pref_name='Glucagon-like peptide 1 receptor',
            target_type='SINGLE PROTEIN',
            target_components=[Row(accession='P43220')],
        )],
        schema=TARGET_SCHEMA,
    )
    gene_df = spark.createDataFrame(
        [Row(id='ENSG00000112164', uniprot_trembl=[], uniprot_swissprot=['P43220'])],
        schema=GENE_SCHEMA,
    )

    result = _chembl_target(target_df, gene_df).collect()

    assert len(result) == 1
    row = result[0]
    assert row['targets'] == ['ENSG00000112164']
    assert row['targetName'] == 'Glucagon-like peptide 1 receptor'


def test_chembl_target_resolves_trembl(spark):
    """A single-protein target with a matching trembl ID produces a non-empty targets list."""
    target_df = spark.createDataFrame(
        [Row(
            target_chembl_id='CHEMBL9999',
            pref_name='Some Protein',
            target_type='SINGLE PROTEIN',
            target_components=[Row(accession='A0A001')],
        )],
        schema=TARGET_SCHEMA,
    )
    gene_df = spark.createDataFrame(
        [Row(id='ENSG00000000001', uniprot_trembl=['A0A001'], uniprot_swissprot=[])],
        schema=GENE_SCHEMA,
    )

    result = _chembl_target(target_df, gene_df).collect()

    assert len(result) == 1
    assert result[0]['targets'] == ['ENSG00000000001']


def test_chembl_target_no_matching_gene_yields_empty_targets(spark):
    """When no gene maps to the UniProt accession, targets is an empty list."""
    target_df = spark.createDataFrame(
        [Row(
            target_chembl_id='CHEMBL_UNMATCHED',
            pref_name='Unknown Protein',
            target_type='SINGLE PROTEIN',
            target_components=[Row(accession='ZZZZZZ')],
        )],
        schema=TARGET_SCHEMA,
    )
    gene_df = spark.createDataFrame(
        [Row(id='ENSG00000000001', uniprot_trembl=['A0A001'], uniprot_swissprot=['P43220'])],
        schema=GENE_SCHEMA,
    )

    result = _chembl_target(target_df, gene_df).collect()

    assert len(result) == 1
    assert result[0]['targets'] == []


def test_chembl_target_null_accession_is_dropped(spark):
    """Target components with null accession are excluded from the join."""
    target_df = spark.createDataFrame(
        [Row(
            target_chembl_id='CHEMBL_NULL',
            pref_name='Protein Complex',
            target_type='PROTEIN COMPLEX',
            target_components=[Row(accession=None)],
        )],
        schema=TARGET_SCHEMA,
    )
    gene_df = spark.createDataFrame(
        [Row(id='ENSG00000000001', uniprot_trembl=[], uniprot_swissprot=['P43220'])],
        schema=GENE_SCHEMA,
    )

    result = _chembl_target(target_df, gene_df).collect()

    # No accession to join on, so the target chembl id is dropped entirely
    assert len(result) == 0

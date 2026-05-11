"""Tests for the uniprot_literature pyspark task."""

from __future__ import annotations

from pyspark.sql import Row
from pyspark.sql.types import (
    ArrayType,
    StringType,
    StructField,
    StructType,
)

_DISEASE_SCHEMA = StructType([
    StructField('omimId', StringType(), True),
    StructField('name', StringType(), True),
    StructField('acronym', StringType(), True),
    StructField('description', StringType(), True),
    StructField('evidencePmids', ArrayType(StringType()), True),
])

_VARIANT_SCHEMA = StructType([
    StructField('ftId', StringType(), True),
    StructField('dbSnpRsId', StringType(), True),
    StructField('aminoacidChange', StringType(), True),
    StructField('description', StringType(), True),
    StructField('evidencePmids', ArrayType(StringType()), True),
    StructField('linkedOmimIds', ArrayType(StringType()), True),
])

_PARSED_SCHEMA = StructType([
    StructField('id', StringType(), True),
    StructField('accession', StringType(), True),
    StructField('geneNames', ArrayType(StringType()), True),
    StructField('diseases', ArrayType(_DISEASE_SCHEMA), True),
    StructField('variants', ArrayType(_VARIANT_SCHEMA), True),
])


def _make_parsed_row(
    accession='P38398',
    diseases=None,
    variants=None,
):
    return Row(
        id='BRCA1_HUMAN',
        accession=accession,
        geneNames=['BRCA1'],
        diseases=diseases or [],
        variants=variants or [],
    )


def test_uniprot_literature_projects_expected_columns(spark, tmp_path, monkeypatch):
    from pts.pyspark import uniprot_literature as mod

    parsed = [
        _make_parsed_row(
            diseases=[
                Row(
                    omimId='604370',
                    name='Breast-ovarian cancer 1',
                    acronym='BROVCA1',
                    description='Cancer.',
                    evidencePmids=['7545954', '9145676'],
                ),
                Row(
                    omimId='999999',
                    name='Empty-evidence disease',
                    acronym='EMPTY',
                    description='No PMIDs.',
                    evidencePmids=[],
                ),
            ],
        ),
    ]
    parsed_path = tmp_path / 'parsed.parquet'
    spark.createDataFrame(parsed, schema=_PARSED_SCHEMA).write.parquet(str(parsed_path))

    # Stub add_efo_mapping: behave as a no-op that adds a null column
    def _fake_add_efo_mapping(spark, evidence_df, **kwargs):
        from pyspark.sql import functions as f
        return evidence_df.withColumn('diseaseFromSourceMappedId', f.lit(None).cast('string'))

    monkeypatch.setattr(mod, 'add_efo_mapping', _fake_add_efo_mapping)

    out_path = tmp_path / 'literature.parquet'
    mod._compute_literature(
        spark=spark,
        parsed_path=str(parsed_path),
        disease_label_lut_path='unused',
        disease_id_lut_path='unused',
    ).write.parquet(str(out_path))

    df = spark.read.parquet(str(out_path))
    rows = df.collect()

    assert len(rows) == 1  # the empty-evidence disease is filtered out
    r = rows[0]
    assert r['datasourceId'] == 'uniprot_literature'
    assert r['datatypeId'] == 'genetic_association'
    assert r['targetFromSourceId'] == 'P38398'
    assert r['diseaseFromSource'] == 'Breast-ovarian cancer 1'
    assert r['diseaseFromSourceId'] == 'OMIM:604370'
    assert r['literature'] == ['7545954', '9145676']
    assert r['confidence'] == 'high'
    assert r['urls'][0]['niceName'] == 'UniProt'
    assert r['urls'][0]['url'] == 'https://www.uniprot.org/uniprotkb/P38398'

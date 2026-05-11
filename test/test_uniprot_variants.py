"""Tests for the uniprot_variants pyspark task."""

from __future__ import annotations

from pyspark.sql import Row
from pyspark.sql.types import ArrayType, StringType, StructField, StructType

_DISEASE_SCHEMA = StructType([
    StructField('omimId', StringType(), True),
    StructField('name', StringType(), True),
    StructField('acronym', StringType(), True),
    StructField('description', StringType(), True),
    StructField('evidencePmids', ArrayType(StringType()), True),
])

_VARIANT_SCHEMA = StructType([
    StructField('ftId', StringType(), True),
    StructField('description', StringType(), True),
    StructField('aminoacidChange', StringType(), True),
    StructField('dbSnpRsId', StringType(), True),
    StructField('linkedOmimIds', ArrayType(StringType()), True),
    StructField('evidencePmids', ArrayType(StringType()), True),
])

_PARSED_SCHEMA = StructType([
    StructField('id', StringType(), True),
    StructField('accession', StringType(), True),
    StructField('geneNames', ArrayType(StringType()), True),
    StructField('diseases', ArrayType(_DISEASE_SCHEMA), True),
    StructField('variants', ArrayType(_VARIANT_SCHEMA), True),
])


def _parsed_row(accession='P38398', diseases=None, variants=None):
    return Row(
        id='BRCA1_HUMAN',
        accession=accession,
        geneNames=['BRCA1'],
        diseases=diseases or [],
        variants=variants or [],
    )


def _variant(
    ft_id='VAR_007800',
    description='in BROVCA1; dbSNP:rs28897696',
    aa='p.Arg1699Gln',
    rsid='rs28897696',
    linked_omim=('604370',),
    pmids=('9145676',),
):
    return Row(
        ftId=ft_id,
        description=description,
        aminoacidChange=aa,
        dbSnpRsId=rsid,
        linkedOmimIds=list(linked_omim),
        evidencePmids=list(pmids),
    )


def _disease(
    omim='604370',
    name='Breast-ovarian cancer 1',
    acronym='BROVCA1',
    description='A cancer predisposition syndrome.',
):
    return Row(
        omimId=omim,
        name=name,
        acronym=acronym,
        description=description,
        evidencePmids=['7545954'],
    )


def test_uniprot_variants_projection(spark, tmp_path, monkeypatch):
    from pts.pyspark import uniprot_variants as mod

    parsed = [
        _parsed_row(
            diseases=[_disease()],
            variants=[_variant()],
        ),
    ]
    parsed_path = tmp_path / 'parsed.parquet'
    spark.createDataFrame(parsed, schema=_PARSED_SCHEMA).write.parquet(str(parsed_path))

    def _fake_add_efo_mapping(spark, evidence_df, **kwargs):
        from pyspark.sql import functions as f
        return evidence_df.withColumn('diseaseFromSourceMappedId', f.lit(None).cast('string'))

    monkeypatch.setattr(mod, 'add_efo_mapping', _fake_add_efo_mapping)

    out_path = tmp_path / 'variants.parquet'
    mod._compute_variants(
        spark=spark,
        parsed_path=str(parsed_path),
        disease_label_lut_path='unused',
        disease_id_lut_path='unused',
    ).write.parquet(str(out_path))

    df = spark.read.parquet(str(out_path))
    rows = df.collect()

    assert len(rows) == 1
    r = rows[0]
    assert r['datasourceId'] == 'uniprot_variants'
    assert r['datatypeId'] == 'genetic_association'
    assert r['targetFromSourceId'] == 'P38398'
    assert r['diseaseFromSource'] == 'Breast-ovarian cancer 1'
    assert r['diseaseFromSourceId'] == 'OMIM:604370'
    assert r['variantRsId'] == 'rs28897696'
    assert r['literature'] == ['9145676']
    assert r['confidence'] == 'high'
    assert r['targetModulation'] == 'up_or_down'

    # Fields that previously existed but are NO LONGER part of the schema:
    assert 'urls' not in df.columns
    assert 'alleleOrigins' not in df.columns
    assert 'variantAminoacidDescriptions' not in df.columns


def test_uniprot_variants_drops_unlinked(spark, tmp_path, monkeypatch):
    from pts.pyspark import uniprot_variants as mod

    parsed = [
        _parsed_row(
            diseases=[_disease()],
            variants=[
                _variant(ft_id='VAR_lonely', linked_omim=(), rsid='rs1'),
            ],
        ),
    ]
    parsed_path = tmp_path / 'parsed.parquet'
    spark.createDataFrame(parsed, schema=_PARSED_SCHEMA).write.parquet(str(parsed_path))

    def _fake_add_efo_mapping(spark, evidence_df, **kwargs):
        from pyspark.sql import functions as f
        return evidence_df.withColumn('diseaseFromSourceMappedId', f.lit(None).cast('string'))

    monkeypatch.setattr(mod, 'add_efo_mapping', _fake_add_efo_mapping)

    out_path = tmp_path / 'variants.parquet'
    mod._compute_variants(
        spark=spark,
        parsed_path=str(parsed_path),
        disease_label_lut_path='unused',
        disease_id_lut_path='unused',
    ).write.parquet(str(out_path))

    df = spark.read.parquet(str(out_path))
    assert df.count() == 0


def test_uniprot_variants_medium_confidence_for_indefinite_description(spark, tmp_path, monkeypatch):
    from pts.pyspark import uniprot_variants as mod

    parsed = [
        _parsed_row(
            diseases=[
                _disease(
                    description='The disease may be caused by mutations affecting the gene represented in this entry.',
                ),
            ],
            variants=[_variant()],
        ),
    ]
    parsed_path = tmp_path / 'parsed.parquet'
    spark.createDataFrame(parsed, schema=_PARSED_SCHEMA).write.parquet(str(parsed_path))

    def _fake_add_efo_mapping(spark, evidence_df, **kwargs):
        from pyspark.sql import functions as f
        return evidence_df.withColumn('diseaseFromSourceMappedId', f.lit(None).cast('string'))

    monkeypatch.setattr(mod, 'add_efo_mapping', _fake_add_efo_mapping)

    out_path = tmp_path / 'variants.parquet'
    mod._compute_variants(
        spark=spark,
        parsed_path=str(parsed_path),
        disease_label_lut_path='unused',
        disease_id_lut_path='unused',
    ).write.parquet(str(out_path))

    df = spark.read.parquet(str(out_path))
    rows = df.collect()
    assert len(rows) == 1
    assert rows[0]['confidence'] == 'medium'

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


def _disease(omim='604370', name='Breast-ovarian cancer 1', acronym='BROVCA1'):
    return Row(
        omimId=omim,
        name=name,
        acronym=acronym,
        description='Cancer.',
        evidencePmids=['7545954'],
    )


def test_uniprot_variants_projection_and_origin(spark, tmp_path, monkeypatch):
    from pts.pyspark import uniprot_variants as mod

    parsed = [
        _parsed_row(
            diseases=[_disease()],
            variants=[
                _variant(),  # germline (rsid not in census)
                _variant(ft_id='VAR_999', rsid='rs99999', linked_omim=('604370',)),
            ],
        ),
    ]
    parsed_path = tmp_path / 'parsed.parquet'
    spark.createDataFrame(parsed, schema=_PARSED_SCHEMA).write.parquet(str(parsed_path))

    census_path = tmp_path / 'census.txt'
    census_path.write_text('rs99999\n')

    def _fake_add_efo_mapping(spark, evidence_df, **kwargs):
        from pyspark.sql import functions as f
        return evidence_df.withColumn('diseaseFromSourceMappedId', f.lit(None).cast('string'))

    monkeypatch.setattr(mod, 'add_efo_mapping', _fake_add_efo_mapping)

    out_path = tmp_path / 'variants.parquet'
    mod._compute_variants(
        spark=spark,
        parsed_path=str(parsed_path),
        somatic_census_path=str(census_path),
        disease_label_lut_path='unused',
        disease_id_lut_path='unused',
    ).write.parquet(str(out_path))

    df = spark.read.parquet(str(out_path))
    rows = df.collect()
    by_rsid = {r['variantRsId']: r for r in rows}

    assert by_rsid['rs28897696']['datatypeId'] == 'genetic_association'
    assert by_rsid['rs28897696']['alleleOrigins'] == ['germline']
    assert by_rsid['rs28897696']['datasourceId'] == 'uniprot_variants'
    assert by_rsid['rs28897696']['variantAminoacidDescriptions'] == ['p.Arg1699Gln']
    assert by_rsid['rs28897696']['diseaseFromSourceId'] == 'OMIM:604370'
    assert by_rsid['rs28897696']['literature'] == ['9145676']

    assert by_rsid['rs99999']['datatypeId'] == 'somatic_mutations'
    assert by_rsid['rs99999']['alleleOrigins'] == ['somatic']


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

    census_path = tmp_path / 'census.txt'
    census_path.write_text('')

    def _fake_add_efo_mapping(spark, evidence_df, **kwargs):
        from pyspark.sql import functions as f
        return evidence_df.withColumn('diseaseFromSourceMappedId', f.lit(None).cast('string'))

    monkeypatch.setattr(mod, 'add_efo_mapping', _fake_add_efo_mapping)

    out_path = tmp_path / 'variants.parquet'
    mod._compute_variants(
        spark=spark,
        parsed_path=str(parsed_path),
        somatic_census_path=str(census_path),
        disease_label_lut_path='unused',
        disease_id_lut_path='unused',
    ).write.parquet(str(out_path))

    df = spark.read.parquet(str(out_path))
    assert df.count() == 0

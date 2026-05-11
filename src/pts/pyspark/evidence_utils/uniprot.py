"""Shared helpers for the uniprot_variants and uniprot_literature evidence tasks."""

from __future__ import annotations

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as f

UNIPROT_BASE_URL = 'https://www.uniprot.org/uniprotkb/'

DATASOURCE_VARIANTS = 'uniprot_variants'
DATASOURCE_LITERATURE = 'uniprot_literature'

DATATYPE_GENETIC_ASSOCIATION = 'genetic_association'
DATATYPE_GENETIC_LITERATURE = 'genetic_literature'
DATATYPE_SOMATIC_MUTATION = 'somatic_mutation'


def uniprot_url(accession_col: Column) -> Column:
    """Build the canonical UniProt URL for an accession column."""
    return f.concat(f.lit(UNIPROT_BASE_URL), accession_col)


def uniprot_urls_struct_array(accession_col: Column) -> Column:
    """Return an array(struct(niceName, url)) shaped like other PTS evidence pipelines."""
    return f.array(
        f.struct(
            f.lit('UniProt').alias('niceName'),
            uniprot_url(accession_col).alias('url'),
        )
    )


def confidence_from_literature(literature_col: Column) -> Column:
    """High when at least one citing PMID is present, medium otherwise.

    Used to drive the score_expression mapping in evidence_postprocess
    (high -> 1.0, medium -> 0.5).
    """
    return f.when(f.size(literature_col) > 0, f.lit('high')).otherwise(f.lit('medium'))


def load_somatic_rsids(spark: SparkSession, path: str) -> DataFrame:
    r"""Read the static somatic dbSNP census into a single-column DataFrame.

    The census file is tab-separated with columns `rsID\tUniProtAcc\tEnsemblId\tEfo\tflag`.
    Only the first column (rsID) is meaningful for somatic classification.
    Empty lines and comment lines (starting with `#`) are skipped.

    Returns a DataFrame with one column `dbSnpRsId` of unique rsIDs, suitable for a
    broadcast join.
    """
    raw = spark.read.text(path).withColumnRenamed('value', 'line')
    return (
        raw.select(f.trim(f.col('line')).alias('line'))
        .filter((f.length('line') > 0) & (~f.col('line').startswith('#')))
        .select(f.split(f.col('line'), '\\t').getItem(0).alias('dbSnpRsId'))
        .filter(f.col('dbSnpRsId').startswith('rs'))
        .distinct()
    )

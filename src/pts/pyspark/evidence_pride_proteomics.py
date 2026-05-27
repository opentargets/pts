"""Generate disease/target evidence from PRIDE re-analysed differential proteomics datasets.

Reads per-study ``*_FC.txt`` (limma-style differential expression tables) and
``*sdrf.json`` (sample / experiment metadata) plus a curation TSV mapping the
free-text disease tokens in filenames and SDRF records to EFO ids. Emits records
conforming to the ``pride_proteomics`` block of
``disease_target_evidence.json`` so the existing ``evidence_postprocess`` step
can validate, score and produce the final parquet output.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from loguru import logger
from pyspark.sql import Window
from pyspark.sql import functions as f
from pyspark.sql import types as t

from pts.pyspark.common.session import Session

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


# Schema for the *_FC.txt files (limma topTable export).
FC_SCHEMA = t.StructType(
    [
        t.StructField('logFC', t.DoubleType(), True),
        t.StructField('AveExpr', t.DoubleType(), True),
        t.StructField('t', t.DoubleType(), True),
        t.StructField('P.Value', t.DoubleType(), True),
        t.StructField('adj.P.Val', t.DoubleType(), True),
        t.StructField('B', t.DoubleType(), True),
        t.StructField('ENSG', t.StringType(), True),
        t.StructField('Gene Symbol', t.StringType(), True),
        t.StructField('Protein IDs', t.StringType(), True),
    ]
)

# Pattern matching study + contrast slug from the FC filename.
# Filenames look like: PXD006122_OpenTargets_Alzheimersdisease-Control_FC.txt
_FC_FILENAME_RE = re.compile(r'(PXD\d+)_OpenTargets_(.+?)_FC\.txt$')


def _extract_study_and_slug() -> tuple:
    """Return Spark Columns extracting the studyId and contrast slug from the file path."""
    fname = f.regexp_extract(f.input_file_name(), r'([^/]+_FC\.txt)$', 1)
    study = f.regexp_extract(fname, r'^(PXD\d+)_OpenTargets_.+_FC\.txt$', 1)
    slug = f.regexp_extract(fname, r'^PXD\d+_OpenTargets_(.+)_FC\.txt$', 1)
    return study, slug


def read_fc_files(spark, fc_glob: str) -> DataFrame:
    """Read all ``*_FC.txt`` files under the glob into a single long dataframe."""
    study, slug = _extract_study_and_slug()
    return (
        spark.read.csv(fc_glob, sep='\t', header=True, schema=FC_SCHEMA)
        .withColumn('studyId', study)
        .withColumn('contrastSlug', slug)
    )


def read_sdrf_files(spark, sdrf_glob: str) -> DataFrame:
    """Read all ``*sdrf.json`` files and aggregate experiment-level metadata per study."""
    raw = (
        spark.read.option('multiLine', 'true').json(sdrf_glob)
        # input_file_name lets us recover the PXD code per record.
        .withColumn(
            'studyId',
            f.regexp_extract(f.input_file_name(), r'(PXD\d+)_OpenTargets_sdrf\.json$', 1),
        )
    )

    # pubmedIds may come as a string of comma-separated PMIDs.
    literature = f.when(
        f.col('pubmedIds').isNotNull() & (f.length(f.trim(f.col('pubmedIds'))) > 0),
        f.expr("filter(transform(split(pubmedIds, ','), x -> trim(x)), x -> length(x) > 0)"),
    ).otherwise(f.array().cast(t.ArrayType(t.StringType())))

    # Aggregate the assay-level tissue list per study.
    tissues = (
        raw.select('studyId', f.explode_outer('experimentalDesigns').alias('assay'))
        .select('studyId', f.col('assay.tissue').alias('tissue'))
        .where(f.col('tissue').isNotNull() & (f.length(f.trim(f.col('tissue'))) > 0))
        .groupBy('studyId')
        .agg(f.collect_set('tissue').alias('biosamplesFromSource'))
    )

    study_meta = raw.select(
        'studyId',
        f.col('experimentType').alias('_experimentType'),
        f.col('provider').alias('_provider'),
        literature.alias('literature'),
    ).dropDuplicates(['studyId'])

    return study_meta.join(tissues, on='studyId', how='left').withColumn(
        'studyOverview',
        f.concat_ws(' — ', f.col('_experimentType'), f.col('_provider')),
    ).drop('_experimentType', '_provider')


def read_disease_mapping(spark, path: str) -> DataFrame:
    """Read the curation TSV (one row per studyId/contrast).

    Rows with ``keep != 'true'`` are dropped here so the harmoniser only emits
    evidence for biologically meaningful disease-vs-control contrasts. The
    ``keep`` column is auto-populated by ``auto_map_diseases.py`` and can be
    manually overridden in the TSV.
    """
    raw = spark.read.csv(path, sep='\t', header=True)
    cols = set(raw.columns)
    if 'keep' in cols:
        raw = raw.filter(f.lower(f.trim(f.col('keep'))) == 'true')
    return (
        raw.select(
            'studyId',
            f.col('groupCase').alias('_groupCase'),
            f.col('groupControl').alias('_groupControl'),
            'diseaseFromSourceMappedId',
        )
        .dropDuplicates(['studyId', '_groupCase', '_groupControl'])
    )


def _humanise(token_col):
    """Insert spaces before capital letters in CamelCase / runtogether tokens."""
    # Best-effort: split CamelCase boundaries and replace underscores with spaces.
    spaced = f.regexp_replace(token_col, '_', ' ')
    spaced = f.regexp_replace(spaced, '([a-z])([A-Z])', '$1 $2')
    return f.trim(spaced)


def generate_evidence(
    fc: DataFrame,
    sdrf: DataFrame,
    disease_mapping: DataFrame,
    adj_pval_cutoff: float = 0.05,
    confidence_high_cutoff: float = 0.01,
) -> DataFrame:
    """Build the ``pride_proteomics`` evidence dataframe.

    Args:
        fc: stacked differential expression rows from all studies.
        sdrf: one row per study with aggregated tissue / literature / overview.
        disease_mapping: curation TSV restricted to filename-derived rows.
        adj_pval_cutoff: rows with ``adj.P.Val`` strictly greater than this are dropped.
        confidence_high_cutoff: ``HIGH`` confidence below or equal to this ``adj.P.Val``,
            otherwise ``MEDIUM``.

    Returns:
        Evidence dataframe matching the ``pride_proteomics`` schema.
    """
    # Parse the slug -> (groupCase, groupControl).
    fc_split = (
        fc.withColumn('_parts', f.split(f.col('contrastSlug'), '-'))
        .withColumn('_groupCase', f.col('_parts').getItem(0))
        .withColumn('_groupControl', f.col('_parts').getItem(1))
        .drop('_parts')
    )

    # Attach mapped EFO ids; unmapped contrasts keep a null diseaseFromSourceMappedId.
    mapped = fc_split.join(
        disease_mapping.select(
            'studyId', '_groupCase', '_groupControl', 'diseaseFromSourceMappedId',
        ),
        on=['studyId', '_groupCase', '_groupControl'],
        how='left',
    )

    # Build contrast string from humanised tokens.
    contrast = f.concat_ws(
        ' vs ', _humanise(f.col('_groupCase')), _humanise(f.col('_groupControl'))
    )

    # Filter: drop rows without ENSG and rows failing the adj.P.Val cutoff.
    filtered = (
        mapped
        .where(f.col('ENSG').isNotNull() & (f.col('ENSG') != f.lit('<NA>')))
        .where(f.col('`adj.P.Val`').isNotNull() & (f.col('`adj.P.Val`') <= f.lit(adj_pval_cutoff)))
        .where(f.col('logFC').isNotNull() & f.col('`P.Value`').isNotNull())
        .withColumn('contrast', contrast)
    )

    # Percentile rank of |logFC| within (studyId, contrast).
    window = Window.partitionBy('studyId', 'contrast').orderBy(f.abs(f.col('logFC')))
    ranked = filtered.withColumn(
        'log2FoldChangePercentileRank',
        f.floor(f.percent_rank().over(window) * f.lit(100)).cast(t.IntegerType()),
    )

    confidence = (
        f.when(f.col('`adj.P.Val`') <= f.lit(confidence_high_cutoff), f.lit('HIGH'))
        .otherwise(f.lit('MEDIUM'))
    )

    # Join study-level metadata.
    enriched = ranked.join(sdrf, on='studyId', how='left')

    return enriched.select(
        f.lit('pride_proteomics').alias('datasourceId'),
        f.lit('proteomics_differential').alias('datatypeId'),
        f.col('ENSG').alias('targetFromSourceId'),
        f.col('studyId'),
        f.coalesce(f.col('studyOverview'), f.lit('PRIDE re-analysed differential proteomics')).alias('studyOverview'),
        f.col('contrast'),
        f.col('logFC').alias('log2FoldChangeValue'),
        f.col('log2FoldChangePercentileRank'),
        f.col('`P.Value`').alias('resourceScore'),
        confidence.alias('confidence'),
        _humanise(f.col('_groupCase')).alias('diseaseFromSource'),
        f.col('diseaseFromSourceMappedId'),
        f.col('biosamplesFromSource'),
        f.col('literature'),
    )


def evidence_pride_proteomics(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Pyspark step entry point — see module docstring for behaviour."""
    settings = settings or {}
    adj_pval_cutoff = float(settings.get('adj_pval_cutoff', 0.05))
    confidence_high_cutoff = float(settings.get('confidence_high_cutoff', 0.01))

    session = Session(app_name='evidence_pride_proteomics', properties=properties)

    logger.info(f'reading FC files from {source["fc_files"]}')
    fc = read_fc_files(session.spark, source['fc_files'])

    logger.info(f'reading sdrf files from {source["sdrf_files"]}')
    sdrf = read_sdrf_files(session.spark, source['sdrf_files'])

    logger.info(f'reading disease mapping from {source["disease_mapping"]}')
    disease_mapping = read_disease_mapping(session.spark, source['disease_mapping'])

    evidence = generate_evidence(
        fc, sdrf, disease_mapping,
        adj_pval_cutoff=adj_pval_cutoff,
        confidence_high_cutoff=confidence_high_cutoff,
    )

    logger.info(f'writing pride_proteomics evidence to {destination}')
    evidence.write.mode('overwrite').parquet(destination)

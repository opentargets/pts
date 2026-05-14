"""Collapsed literature_cooccurrence + evidence_epmc step.

Generates target-disease cooccurrences from matches, writes the intermediate
cooccurrence parquet, then re-reads it and computes EPMC evidence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from literature.dataset.match_mapped import MatchMapped
from loguru import logger

from pts.pyspark import evidence_epmc
from pts.pyspark.common.session import Session

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


def _adapt_cooccurrence_for_evidence(cooccurrences: DataFrame) -> DataFrame:
    """Rename ``Cooccurrence``-schema columns to the names ``_compute_evidence`` expects.

    The ``literature`` library's ``Cooccurrence`` dataset names the mapped entity
    ids ``mappedId1``/``mappedId2`` and the score ``evidenceScore``, but
    ``pts.pyspark.evidence_epmc._compute_evidence`` consumes
    ``keywordId1``/``keywordId2``/``evidence_score``. This bridges the two.

    Args:
        cooccurrences: A DataFrame with the ``Cooccurrence`` schema.

    Returns:
        The same DataFrame with the three columns renamed.
    """
    return (
        cooccurrences
        .withColumnRenamed('mappedId1', 'keywordId1')
        .withColumnRenamed('mappedId2', 'keywordId2')
        .withColumnRenamed('evidenceScore', 'evidence_score')
    )


def literature_cooccurrence_evidence(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Collapsed literature_cooccurrence + evidence_epmc step.

    Generates target-disease cooccurrences from the match dataset and writes the
    intermediate cooccurrence parquet. Then re-reads that parquet (a clean
    lineage cut on a memory-tight cluster), bridges its column names, and
    computes EPMC evidence.

    Args:
        source: ``match``.
        destination: ``cooccurrence``, ``evidence``.
        settings: unused.
        properties: Spark properties forwarded to the session.
    """
    spark = Session(app_name='literature', properties=properties)

    logger.info(f'load matches from: {source["match"]}')
    match = spark.load_data(path=source['match'])

    logger.info(f'write cooccurrences to {destination["cooccurrence"]}')
    cooccurrence = MatchMapped(match).generate_target_disease_cooccurrences().df
    cooccurrence.write.mode('overwrite').parquet(destination['cooccurrence'])

    logger.info('re-read cooccurrences and compute EPMC evidence')
    cooccurrence_reread = spark.spark.read.parquet(destination['cooccurrence'])
    evidence = evidence_epmc._compute_evidence(_adapt_cooccurrence_for_evidence(cooccurrence_reread))

    logger.info(f'write EPMC evidence to {destination["evidence"]}')
    evidence.write.mode('overwrite').parquet(destination['evidence'])

"""Collapsed literature_cooccurrence + evidence_epmc step.

Generates target-disease cooccurrences from matches, writes the intermediate
cooccurrence parquet, then re-reads it and computes EPMC evidence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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

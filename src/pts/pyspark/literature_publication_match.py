"""Collapsed literature_publication + literature_match step.

Reads EPMC publications, builds the publication dataset in memory (no
intermediate publication parquet is written), extracts and maps matches, then
writes only the valid and failed match datasets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


def _epmc_read_path(epmc_path: str, kind: str, date_prefix: str | None) -> str:
    """Build the EPMC jsonl glob path for a publication kind.

    Mirrors ``EPMCPublication._read_in_with_schema`` but allows an optional
    day-folder prefix so a single month can be selected. The library hardcodes
    its own glob, hence this reimplementation.

    Args:
        epmc_path: Base EPMC path, e.g. ``gs://otar025-epmc/ml02``.
        kind: Publication kind, ``abstract`` or ``fulltext``.
        date_prefix: Optional day-folder prefix, e.g. ``2026_03``. Falsy values
            (None, empty string) select all dates.

    Returns:
        A glob path string suitable for ``spark.read.json``.
    """
    base = epmc_path.rstrip('/')
    if date_prefix:
        return f'{base}/{kind}/{date_prefix}*/**/*.jsonl'
    return f'{base}/{kind}/**/*.jsonl'


def _maybe_repartition(df: DataFrame, repartition: int | None) -> DataFrame:
    """Repartition a DataFrame to a fixed partition count when configured.

    The raw EPMC input is scattered across many small day-folder files; an
    explicit repartition right after the read keeps Spark task counts sane.

    Args:
        df: DataFrame to repartition.
        repartition: Target partition count. Falsy values leave ``df`` unchanged.

    Returns:
        The repartitioned DataFrame, or ``df`` unchanged when ``repartition`` is falsy.
    """
    if repartition:
        return df.repartition(repartition)
    return df

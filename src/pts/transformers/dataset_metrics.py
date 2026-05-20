"""Config-driven dataset profiler (one row per output dataset).

Sibling of ``release_metrics``: reuses its discovery/count helpers but emits a
dataset-object profile (id, count, file_size, number_of_partitions, and optional
config-driven breakdowns and filter counts).
"""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import fsspec
import polars as pl
from loguru import logger
from otter.config.model import Config

from pts.transformers.release_metrics import (
    _count_parquet_rows,
    _discover_dataset_paths,
    _to_parquet_glob,
)

NULL_KEY = 'null'

OUTPUT_SCHEMA: dict[str, pl.DataType] = {
    'id': pl.String(),
    'count': pl.Int64(),
    'file_size': pl.Int64(),
    'number_of_partitions': pl.Int32(),
    'breakdowns': pl.List(
        pl.Struct({
            'grouping': pl.String(),
            'groups': pl.List(pl.Struct({'value': pl.String(), 'count': pl.Int64()})),
        })
    ),
    'filter_counts': pl.List(pl.Struct({'name': pl.String(), 'count': pl.Int64()})),
}


def _as_lazy(frame: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    """Return a LazyFrame for either a DataFrame or a LazyFrame."""
    return frame if isinstance(frame, pl.LazyFrame) else frame.lazy()


def compute_breakdown(frame: pl.DataFrame | pl.LazyFrame, expression: str) -> dict[str, int]:
    """Group a frame by a SQL expression and count rows per group.

    The expression is evaluated with ``pl.sql_expr``. A plain column, a derived
    expression, or a list column (auto-exploded) are all valid. Null group keys
    are coalesced to ``"null"``. Returns ``{value: count}`` sorted by descending
    count then key. Pass a ``scan_parquet`` LazyFrame for projection pushdown.

    Args:
        frame (pl.DataFrame | pl.LazyFrame): dataset to group.
        expression (str): SQL expression producing the group key.

    Returns:
        dict[str, int]: count per group value.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"studyType": ["gwas", "eqtl", "gwas", None]})
        >>> compute_breakdown(df, "studyType")
        {'gwas': 2, 'eqtl': 1, 'null': 1}
        >>> df2 = pl.DataFrame({"rightStudyType": ["gwas", "eqtl", "eqtl"]})
        >>> compute_breakdown(df2, "concat('gwas-', rightStudyType)")
        {'gwas-eqtl': 2, 'gwas-gwas': 1}
        >>> df3 = pl.DataFrame({"tas": [["a", "b"], ["a"]]})
        >>> compute_breakdown(df3, "tas")
        {'a': 2, 'b': 1}
    """
    selected = _as_lazy(frame).select(pl.sql_expr(expression).alias('k'))
    if isinstance(selected.collect_schema()['k'], pl.List):
        selected = selected.explode('k')
    counts = (
        selected
        .with_columns(pl.col('k').cast(pl.String).fill_null(NULL_KEY))
        .group_by('k')
        .agg(pl.len().alias('count'))
        .collect()
    )
    ordered = sorted(counts.iter_rows(), key=lambda kv: (-kv[1], kv[0]))
    return {key: int(count) for key, count in ordered}


def compute_filter_count(
    frame: pl.DataFrame | pl.LazyFrame, filter_expr: str, distinct: str | None = None
) -> int:
    """Count rows (or distinct values) matching a SQL filter expression.

    Args:
        frame (pl.DataFrame | pl.LazyFrame): dataset to filter.
        filter_expr (str): SQL boolean expression.
        distinct (str | None): when given, count distinct values of this column.

    Returns:
        int: matching row count, or distinct count of ``distinct``.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"score": [0.9, 0.2, 0.7], "geneId": ["g1", "g2", "g1"]})
        >>> compute_filter_count(df, "score > 0.5")
        2
        >>> compute_filter_count(df, "score > 0.5", distinct="geneId")
        1
    """
    filtered = _as_lazy(frame).filter(pl.sql_expr(filter_expr))
    if distinct:
        return int(filtered.select(pl.col(distinct).n_unique()).collect().item())
    return int(filtered.select(pl.len()).collect().item())


def _dataset_file_stats(dataset_path: str) -> tuple[int, int]:
    """Return ``(total bytes, number of part files)`` for a dataset directory.

    Storage-agnostic via fsspec (local ``LocalFileSystem`` or ``gs://`` via
    gcsfs). Only entries whose basename starts with ``part-`` are counted, so
    ``_SUCCESS`` / checksum files are ignored.

    Args:
        dataset_path (str): absolute path to the dataset directory.

    Returns:
        tuple[int, int]: total bytes and number of part files.
    """
    fs, root = fsspec.core.url_to_fs(dataset_path)
    parts = [
        entry
        for entry in fs.ls(root, detail=True)
        if entry['name'].rstrip('/').rsplit('/', 1)[-1].startswith('part-')
    ]
    file_size = sum(int(entry.get('size') or 0) for entry in parts)
    return file_size, len(parts)


def _breakdowns_to_struct(breakdowns: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
    """Shape ``{grouping: {value: count}}`` into the output ``list[struct]``."""
    return [
        {
            'grouping': grouping,
            'groups': [{'value': value, 'count': count} for value, count in groups.items()],
        }
        for grouping, groups in breakdowns.items()
    ]


def _filter_counts_to_struct(filter_counts: dict[str, int]) -> list[dict[str, Any]]:
    """Shape ``{name: count}`` into the output ``list[struct]``."""
    return [{'name': name, 'count': count} for name, count in filter_counts.items()]


def profile_dataset(dataset_path: str, name: str, dataset_config: dict[str, Any]) -> dict[str, Any] | None:
    """Profile one dataset into an output row, or ``None`` if unreadable.

    ``count`` is read from parquet footers; ``file_size`` / partitions from
    storage listing. Breakdowns/filters scan only their columns (lazy). A
    dataset that cannot be read as parquet is skipped (logged), returning None.

    Args:
        dataset_path (str): absolute path to the dataset directory.
        name (str): dataset id (basename).
        dataset_config (dict): optional ``{groupings, filter_counts}`` overlay.

    Returns:
        dict | None: one output row, or None when the dataset cannot be read.
    """
    try:
        count = _count_parquet_rows(dataset_path)
    except Exception:
        logger.warning(f'Skipping unreadable dataset `{name}` at {dataset_path}')
        return None

    file_size, n_partitions = _dataset_file_stats(dataset_path)

    breakdowns: dict[str, dict[str, int]] = {}
    filter_counts: dict[str, int] = {}
    groupings = dataset_config.get('groupings', {})
    filters = dataset_config.get('filter_counts', [])

    if groupings or filters:
        lazy_frame = pl.scan_parquet(_to_parquet_glob(dataset_path))
        for grouping_name, expression in groupings.items():
            breakdowns[grouping_name] = compute_breakdown(lazy_frame, expression)
        for spec in filters:
            filter_counts[spec['name']] = compute_filter_count(
                lazy_frame, spec['filter'], spec.get('distinct')
            )

    return {
        'id': name,
        'count': count,
        'file_size': file_size,
        'number_of_partitions': n_partitions,
        'breakdowns': _breakdowns_to_struct(breakdowns),
        'filter_counts': _filter_counts_to_struct(filter_counts),
    }


def _config_for_dataset(name: str, datasets_config: dict[str, Any]) -> dict[str, Any]:
    """Return the config overlay for a dataset, supporting fnmatch pattern keys.

    Exact-name keys take precedence; otherwise the first matching glob pattern
    key (sorted for determinism) applies. Returns ``{}`` when nothing matches.

    Args:
        name (str): dataset basename.
        datasets_config (dict): the ``settings.datasets`` overlay.

    Returns:
        dict: the matched overlay, or an empty dict.
    """
    if name in datasets_config:
        return datasets_config[name]
    for pattern in sorted(datasets_config):
        if fnmatch(name, pattern):
            return datasets_config[pattern]
    return {}


def dataset_metrics(
    source: dict[str, Path],
    destination: dict[str, Path],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Profile every discovered dataset, writing one parquet per dataset.

    Writes ``{destination['directory']}/{dataset}.parquet`` (a single-row
    profile) for each discovered dataset.

    Args:
        source: unused.
        destination: ``{"directory": <dir>}`` output directory (run-root
            ``metrics/``).
        settings: ``metric_scopes`` (default ``['/output/*']``) and a
            ``datasets`` overlay keyed by dataset basename / fnmatch pattern
            (``{groupings, filter_counts}``).
        config: injected; discovery reads ``release_uri`` or ``work_path``.
    """
    del source

    data_root_uri = config.release_uri or str(config.work_path)
    scope_globs = list(settings.get('metric_scopes', ['/output/*']))
    datasets_config = settings.get('datasets', {})

    destination_dir = str(destination['directory']).rstrip('/')
    destination_name = Path(destination_dir).name

    discovered = _discover_dataset_paths(data_root_uri, scope_globs, config)

    written = 0
    for rel_path in sorted(discovered):
        name = Path(rel_path).name
        if name == destination_name:
            continue
        row = profile_dataset(discovered[rel_path], name, _config_for_dataset(name, datasets_config))
        if row is None:
            continue
        pl.DataFrame([row], schema=OUTPUT_SCHEMA).write_parquet(f'{destination_dir}/{name}.parquet', mkdir=True)
        written += 1

    logger.info(f'Profiled {written} datasets (of {len(discovered)} discovered) under {scope_globs}')

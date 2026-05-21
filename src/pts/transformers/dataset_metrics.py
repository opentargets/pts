"""Config-driven dataset profiler emitting one tidy row per measurement.

Sibling of ``release_metrics``: reuses the shared discovery/count helpers but
emits a long/tidy table — one row per (dataset, metric) measurement — and
combines every discovered dataset into a single parquet for easy querying.

Each row carries the ``run`` identifier (derived from the release/work root),
the ``dataset`` id, the metric ``kind`` (``scalar``/``grouping``/``filter``),
the metric name, the ``expression`` describing how it was computed (NULL for
scalars), the ``group_value`` bucket (NULL for scalar/filter rows), and the
integer ``value``.

Datasets are assumed to be flat parquet directories with ``*.parquet`` files at
the top level. Hive-partitioned datasets (e.g. ``key=value/part-*.parquet``)
are not supported and would be skipped (no top-level parquet files to count).
"""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import fsspec
import polars as pl
from loguru import logger
from otter.config.model import Config

from pts.transformers.parquet_helpers import count_parquet_rows, discover_dataset_paths, to_parquet_glob

NULL_KEY = 'null'

OUTPUT_SCHEMA: dict[str, pl.DataType] = {
    'run': pl.String(),
    'dataset': pl.String(),
    'kind': pl.String(),
    'metric': pl.String(),
    'expression': pl.String(),
    'group_value': pl.String(),
    'value': pl.Int64(),
}


def _as_lazy(frame: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    """Return a LazyFrame for either a DataFrame or a LazyFrame."""
    return frame if isinstance(frame, pl.LazyFrame) else frame.lazy()


def compute_breakdown(
    frame: pl.DataFrame | pl.LazyFrame, expression: str, schema: pl.Schema | None = None
) -> dict[str, int]:
    """Group a frame by a SQL expression and count rows per group.

    The expression is evaluated with ``pl.sql_expr``. A plain column, a derived
    expression, or a list column (auto-exploded) are all valid. Null group keys
    are coalesced to ``"null"``. Returns ``{value: count}`` sorted by descending
    count then key. Pass a ``scan_parquet`` LazyFrame for projection pushdown.

    Args:
        frame (pl.DataFrame | pl.LazyFrame): dataset to group.
        expression (str): SQL expression producing the group key.
        schema (pl.Schema | None): the frame's schema; when given and the
            expression is a bare column, list-detection avoids an extra metadata
            read (useful for remote storage).

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
    if schema is not None and expression in schema:
        is_list = isinstance(schema[expression], pl.List)
    else:
        is_list = isinstance(selected.collect_schema()['k'], pl.List)
    if is_list:
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


def compute_filter_count(frame: pl.DataFrame | pl.LazyFrame, filter_expr: str, distinct: str | None = None) -> int:
    """Count rows (or distinct values) matching a SQL filter expression.

    Args:
        frame (pl.DataFrame | pl.LazyFrame): dataset to filter.
        filter_expr (str): SQL boolean expression.
        distinct (str | None): when given, count distinct non-null values of this column.

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
    if distinct is not None:
        return int(filtered.select(pl.col(distinct).drop_nulls().n_unique()).collect().item())
    return int(filtered.select(pl.len()).collect().item())


def _dataset_file_stats(dataset_path: str) -> tuple[int, int]:
    """Return ``(total bytes, number of parquet files)`` for a dataset directory.

    Storage-agnostic via fsspec (local ``LocalFileSystem`` or ``gs://`` via
    gcsfs). Counts top-level ``*.parquet`` files, covering both Spark
    ``part-*.parquet`` outputs and single-file Polars outputs (e.g. transformer
    datasets such as ``disease``/``clinical_report``); ``_SUCCESS`` and ``.crc``
    markers are ignored. Assumes a flat dataset directory; Hive-partitioned
    layouts (``key=value/part-*.parquet``) have no top-level parquet files and
    would report zero partitions.

    Args:
        dataset_path (str): absolute path to the dataset directory.

    Returns:
        tuple[int, int]: total bytes and number of parquet files.
    """
    # Otter's storage backend and fsspec/gcsfs both authenticate via ambient
    # ADC, so this bare fsspec listing uses the same credentials as discovery;
    # it deliberately fetches counts and sizes in a single listing call.
    fs, root = fsspec.core.url_to_fs(dataset_path)
    parts = [
        entry
        for entry in fs.ls(root, detail=True)
        if entry.get('type') != 'directory' and entry['name'].rstrip('/').endswith('.parquet')
    ]
    file_size = sum(int(entry.get('size') or 0) for entry in parts)
    return file_size, len(parts)


def _metric_row(
    run: str,
    dataset: str,
    kind: str,
    metric: str,
    value: int,
    expression: str | None = None,
    group_value: str | None = None,
) -> dict[str, Any]:
    """Build one tidy metric row matching ``OUTPUT_SCHEMA``."""
    return {
        'run': run,
        'dataset': dataset,
        'kind': kind,
        'metric': metric,
        'expression': expression,
        'group_value': group_value,
        'value': int(value),
    }


def profile_dataset(
    dataset_path: str, name: str, dataset_config: dict[str, Any], run: str
) -> list[dict[str, Any]] | None:
    """Profile one dataset into tidy rows, or ``None`` if it cannot be read.

    Emits scalar rows (``count``, ``file_size``, ``number_of_partitions``), one
    row per grouping bucket, and one row per filter. ``count`` is read from
    parquet footers; ``file_size`` / partitions from a storage listing.

    A dataset that cannot be read or listed is skipped (logged), returning None.
    A malformed grouping/filter expression in config raises ``ValueError`` (fail
    loud — it's an author error to fix).

    Args:
        dataset_path (str): absolute path to the dataset directory.
        name (str): dataset id (basename).
        dataset_config (dict): optional ``{groupings, filter_counts}`` overlay.
        run (str): run identifier stamped on every row.

    Returns:
        list[dict] | None: tidy rows, or None when the dataset cannot be read.
    """
    try:
        count = count_parquet_rows(dataset_path)
        file_size, n_partitions = _dataset_file_stats(dataset_path)
    except Exception:
        logger.opt(exception=True).warning(f'Skipping unreadable dataset `{name}` at {dataset_path}')
        return None

    rows: list[dict[str, Any]] = [
        _metric_row(run, name, 'scalar', 'count', count),
        _metric_row(run, name, 'scalar', 'file_size', file_size),
        _metric_row(run, name, 'scalar', 'number_of_partitions', n_partitions),
    ]

    groupings = dataset_config.get('groupings', {})
    filters = dataset_config.get('filter_counts', [])
    if not groupings and not filters:
        return rows

    lazy_frame = pl.scan_parquet(to_parquet_glob(dataset_path))
    schema = lazy_frame.collect_schema()

    for grouping_name, expression in groupings.items():
        try:
            counts = compute_breakdown(lazy_frame, expression, schema=schema)
        except Exception as e:
            msg = f"Failed to compute grouping '{grouping_name}' (expression '{expression}') for dataset '{name}'"
            raise ValueError(msg) from e
        rows.extend(
            _metric_row(run, name, 'grouping', grouping_name, count, expression=expression, group_value=bucket)
            for bucket, count in counts.items()
        )

    for spec in filters:
        filter_expr = spec['filter']
        distinct = spec.get('distinct')
        try:
            match_count = compute_filter_count(lazy_frame, filter_expr, distinct)
        except Exception as e:
            msg = f"Failed to compute filter '{spec['name']}' (filter '{filter_expr}') for dataset '{name}'"
            raise ValueError(msg) from e
        definition = f'distinct {distinct} where {filter_expr}' if distinct else filter_expr
        rows.append(_metric_row(run, name, 'filter', spec['name'], match_count, expression=definition))

    return rows


def _config_for_dataset(name: str, datasets_config: dict[str, Any]) -> dict[str, Any]:
    """Return the config overlay for a dataset, supporting fnmatch pattern keys.

    Exact-name keys take precedence; otherwise the most specific matching glob
    pattern wins (longest pattern first, then alphabetical for determinism).
    Returns ``{}`` when nothing matches.

    Args:
        name (str): dataset basename.
        datasets_config (dict): the ``settings.datasets`` overlay.

    Returns:
        dict: the matched overlay, or an empty dict.
    """
    if name in datasets_config:
        return datasets_config[name]
    for pattern in sorted(datasets_config, key=lambda p: (-len(p), p)):
        if fnmatch(name, pattern):
            return datasets_config[pattern]
    return {}


def dataset_metrics(
    source: dict[str, Path],
    destination: dict[str, Path],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Profile every discovered dataset into one combined tidy parquet.

    Writes a single ``{destination['directory']}/dataset_metrics.parquet`` table
    containing one row per measurement across all datasets.

    Args:
        source: unused.
        destination: ``{"directory": <dir>}`` output directory (run-root
            ``metrics/``).
        settings: ``metric_scopes`` (default ``['/output/*']``) and a
            ``datasets`` overlay keyed by dataset basename / fnmatch pattern
            (``{groupings, filter_counts}``).
        config: injected; discovery reads ``release_uri`` or ``work_path``, and
            the ``run`` identifier is the last path segment of that root.
    """
    del source

    data_root_uri = config.release_uri or str(config.work_path)
    run = data_root_uri.rstrip('/').rsplit('/', maxsplit=1)[-1]
    scope_globs = list(settings.get('metric_scopes', ['/output/*']))
    datasets_config = settings.get('datasets', {})

    destination_dir = str(destination['directory']).rstrip('/')
    destination_name = Path(destination_dir).name

    discovered = discover_dataset_paths(data_root_uri, scope_globs, config)

    all_rows: list[dict[str, Any]] = []
    profiled = 0
    for rel_path in sorted(discovered):
        name = Path(rel_path).name
        if name == destination_name:
            continue
        rows = profile_dataset(discovered[rel_path], name, _config_for_dataset(name, datasets_config), run)
        if rows is None:
            continue
        all_rows.extend(rows)
        profiled += 1

    out_file = f'{destination_dir}/dataset_metrics.parquet'
    pl.DataFrame(all_rows, schema=OUTPUT_SCHEMA).write_parquet(out_file, mkdir=True)
    logger.info(f'Profiled {profiled} datasets (of {len(discovered)} discovered) into {out_file}')

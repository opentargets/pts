from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from pts.transformers.dataset_metrics import (
    OUTPUT_SCHEMA,
    _breakdowns_to_struct,
    _config_for_dataset,
    _dataset_file_stats,
    _filter_counts_to_struct,
    compute_breakdown,
    compute_filter_count,
    dataset_metrics,
    profile_dataset,
)


def test_compute_breakdown_plain_column_with_null() -> None:
    df = pl.DataFrame({'studyType': ['gwas', 'eqtl', 'gwas', None]})
    assert compute_breakdown(df, 'studyType') == {'gwas': 2, 'eqtl': 1, 'null': 1}


def test_compute_breakdown_derived_expression() -> None:
    df = pl.DataFrame({'rightStudyType': ['gwas', 'eqtl', 'eqtl']})
    assert compute_breakdown(df, "concat('gwas-', rightStudyType)") == {'gwas-eqtl': 2, 'gwas-gwas': 1}


def test_compute_breakdown_explodes_list_column() -> None:
    df = pl.DataFrame({'tas': [['a', 'b'], ['a']]})
    assert compute_breakdown(df, 'tas') == {'a': 2, 'b': 1}


def test_compute_filter_count_rows() -> None:
    df = pl.DataFrame({'score': [0.9, 0.2, 0.7], 'geneId': ['g1', 'g2', 'g1']})
    assert compute_filter_count(df, 'score > 0.5') == 2


def test_compute_filter_count_distinct() -> None:
    df = pl.DataFrame({'score': [0.9, 0.2, 0.7], 'geneId': ['g1', 'g2', 'g1']})
    assert compute_filter_count(df, 'score > 0.5', distinct='geneId') == 1


def test_dataset_file_stats(tmp_path: Path) -> None:
    dataset_dir = tmp_path / 'study'
    dataset_dir.mkdir()
    df = pl.DataFrame({'studyType': ['gwas', 'eqtl']})
    df.write_parquet(dataset_dir / 'part-0.parquet')
    df.write_parquet(dataset_dir / 'part-1.parquet')
    (dataset_dir / '_SUCCESS').write_text('')  # must be ignored

    file_size, n_partitions = _dataset_file_stats(str(dataset_dir))

    assert n_partitions == 2
    assert file_size > 0


def test_breakdowns_to_struct() -> None:
    out = _breakdowns_to_struct({'studyType': {'gwas': 2, 'eqtl': 1}})
    assert out == [
        {'grouping': 'studyType', 'groups': [{'value': 'gwas', 'count': 2}, {'value': 'eqtl', 'count': 1}]}
    ]


def test_filter_counts_to_struct() -> None:
    assert _filter_counts_to_struct({'prioritised_genes': 18422}) == [
        {'name': 'prioritised_genes', 'count': 18422}
    ]


def test_output_schema_builds_and_roundtrips(tmp_path: Path) -> None:
    rows = [
        {
            'id': 'study', 'count': 2, 'file_size': 10, 'number_of_partitions': 1,
            'breakdowns': _breakdowns_to_struct({'studyType': {'gwas': 1, 'eqtl': 1}}),
            'filter_counts': [],
        },
        {
            'id': 'biosample', 'count': 5, 'file_size': 7, 'number_of_partitions': 1,
            'breakdowns': [], 'filter_counts': [],
        },
    ]
    out = tmp_path / 'metrics'
    pl.DataFrame(rows, schema=OUTPUT_SCHEMA).write_parquet(out, mkdir=True)
    back = pl.read_parquet(out)

    assert back.height == 2
    assert back.schema == pl.Schema(OUTPUT_SCHEMA)
    assert back.filter(pl.col('id') == 'study')['breakdowns'].to_list() == [
        [{'grouping': 'studyType', 'groups': [{'value': 'gwas', 'count': 1}, {'value': 'eqtl', 'count': 1}]}]
    ]


def _write_dataset(directory: Path, df: pl.DataFrame, n_files: int = 1) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    rows_per = max(1, df.height // n_files)
    for i in range(n_files):
        chunk = df.slice(i * rows_per, rows_per if i < n_files - 1 else df.height - i * rows_per)
        chunk.write_parquet(directory / f'part-{i}.parquet')
    return str(directory)


def test_profile_dataset_with_grouping_and_filter(tmp_path: Path) -> None:
    df = pl.DataFrame({
        'studyType': ['gwas', 'eqtl', 'gwas'],
        'score': [0.9, 0.2, 0.7],
        'geneId': ['g1', 'g2', 'g1'],
    })
    path = _write_dataset(tmp_path / 'l2g_like', df)
    config = {
        'groupings': {'studyType': 'studyType'},
        'filter_counts': [{'name': 'high', 'filter': 'score > 0.5', 'distinct': 'geneId'}],
    }

    row = profile_dataset(path, 'l2g_like', config)

    assert row is not None
    assert row['id'] == 'l2g_like'
    assert row['count'] == 3
    assert row['number_of_partitions'] == 1
    assert row['file_size'] > 0
    assert row['breakdowns'] == [
        {'grouping': 'studyType', 'groups': [{'value': 'gwas', 'count': 2}, {'value': 'eqtl', 'count': 1}]}
    ]
    assert row['filter_counts'] == [{'name': 'high', 'count': 1}]


def test_profile_dataset_base_stats_only(tmp_path: Path) -> None:
    path = _write_dataset(tmp_path / 'biosample', pl.DataFrame({'id': ['b1', 'b2']}))
    row = profile_dataset(path, 'biosample', {})
    assert row == {
        'id': 'biosample', 'count': 2, 'file_size': row['file_size'],
        'number_of_partitions': 1, 'breakdowns': [], 'filter_counts': [],
    }


def test_profile_dataset_unreadable_returns_none(tmp_path: Path) -> None:
    empty_dir = tmp_path / 'not_parquet'
    empty_dir.mkdir()
    (empty_dir / 'readme.txt').write_text('not parquet')
    assert profile_dataset(str(empty_dir), 'not_parquet', {}) is None


def test_dataset_metrics_writes_one_parquet_per_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    study = _write_dataset(tmp_path / 'study', pl.DataFrame({'studyType': ['gwas', 'eqtl', 'gwas']}))
    biosample = _write_dataset(tmp_path / 'biosample', pl.DataFrame({'id': ['b1', 'b2']}))
    # a discovered dataset whose basename collides with the output dir must be skipped:
    discovered = {
        '/output/study': study,
        '/output/biosample': biosample,
        '/output/metrics': str(tmp_path / 'ignored'),
    }
    monkeypatch.setattr(
        'pts.transformers.dataset_metrics._discover_dataset_paths',
        lambda root, scopes, config: discovered,
    )

    out_dir = tmp_path / 'metrics'
    config = SimpleNamespace(release_uri=str(tmp_path), work_path=tmp_path)
    settings = {'datasets': {'study': {'groupings': {'studyType': 'studyType'}}}}

    dataset_metrics({}, {'directory': str(out_dir)}, settings, config)

    files = sorted(p.name for p in out_dir.glob('*.parquet'))
    assert files == ['biosample.parquet', 'study.parquet']  # 'metrics' skipped

    study_df = pl.read_parquet(out_dir / 'study.parquet')
    assert study_df.height == 1
    assert study_df['id'].item() == 'study'
    assert study_df['count'].item() == 3
    assert study_df['breakdowns'].to_list() == [
        [{'grouping': 'studyType', 'groups': [{'value': 'gwas', 'count': 2}, {'value': 'eqtl', 'count': 1}]}]
    ]
    assert pl.read_parquet(out_dir / 'biosample.parquet')['breakdowns'].to_list() == [[]]


def test_config_for_dataset_pattern_match() -> None:
    cfg = {
        'study': {'groupings': {'a': 'a'}},
        'evidence_*': {'groupings': {'datatype': 'datatypeId'}},
    }
    assert _config_for_dataset('study', cfg) == {'groupings': {'a': 'a'}}
    assert _config_for_dataset('evidence_eva', cfg) == {'groupings': {'datatype': 'datatypeId'}}
    assert _config_for_dataset('biosample', cfg) == {}

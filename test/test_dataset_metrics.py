from __future__ import annotations

from pathlib import Path

import polars as pl

from pts.transformers.dataset_metrics import (
    OUTPUT_SCHEMA,
    _breakdowns_to_struct,
    _dataset_file_stats,
    _filter_counts_to_struct,
    compute_breakdown,
    compute_filter_count,
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

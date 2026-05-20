from __future__ import annotations

import polars as pl

from pts.transformers.dataset_metrics import compute_breakdown


def test_compute_breakdown_plain_column_with_null() -> None:
    df = pl.DataFrame({'studyType': ['gwas', 'eqtl', 'gwas', None]})
    assert compute_breakdown(df, 'studyType') == {'gwas': 2, 'eqtl': 1, 'null': 1}


def test_compute_breakdown_derived_expression() -> None:
    df = pl.DataFrame({'rightStudyType': ['gwas', 'eqtl', 'eqtl']})
    assert compute_breakdown(df, "concat('gwas-', rightStudyType)") == {'gwas-eqtl': 2, 'gwas-gwas': 1}


def test_compute_breakdown_explodes_list_column() -> None:
    df = pl.DataFrame({'tas': [['a', 'b'], ['a']]})
    assert compute_breakdown(df, 'tas') == {'a': 2, 'b': 1}

"""Tests for GroupedCountMetric, GroupedSumMetric, GroupedCountExplodeMetric."""
import polars as pl
import pytest

from pts.metrics.grouped import (
    GroupedCountExplodeMetric,
    GroupedCountExplodeResult,
    GroupedCountMetric,
    GroupedCountResult,
    GroupedSumMetric,
    GroupedSumResult,
)


def test_grouped_count_returns_one_row_per_group():
    df = pl.DataFrame({'studyType': ['gwas', 'eqtl', 'gwas', 'gwas']})
    result = GroupedCountMetric(name='g', group_by=['studyType']).compute(df)
    assert isinstance(result, GroupedCountResult)
    assert len(result.groups) == 2


def test_grouped_count_sorted_descending_by_count():
    df = pl.DataFrame({'studyType': ['gwas', 'eqtl', 'gwas', 'gwas']})
    result = GroupedCountMetric(name='g', group_by=['studyType']).compute(df)
    assert result.groups[0].key == {'studyType': 'gwas'}
    assert result.groups[0].count == 3
    assert result.groups[1].key == {'studyType': 'eqtl'}
    assert result.groups[1].count == 1


def test_grouped_count_excludes_null_keys():
    df = pl.DataFrame({'studyType': ['gwas', None, 'gwas']})
    result = GroupedCountMetric(name='g', group_by=['studyType']).compute(df)
    assert len(result.groups) == 1
    assert result.groups[0].key == {'studyType': 'gwas'}


def test_grouped_count_multi_column_group_by():
    df = pl.DataFrame({
        'studyType': ['gwas', 'gwas', 'eqtl'],
        'method': ['FINEMAP', 'SuSiE', 'FINEMAP'],
    })
    result = GroupedCountMetric(name='g', group_by=['studyType', 'method']).compute(df)
    assert len(result.groups) == 3


def test_grouped_count_empty_dataframe():
    df = pl.DataFrame({'studyType': pl.Series([], dtype=pl.String)})
    result = GroupedCountMetric(name='g', group_by=['studyType']).compute(df)
    assert result.groups == []


def test_grouped_sum_sums_column_per_group():
    df = pl.DataFrame({
        'aggregationValue': ['genetics', 'genetics', 'literature'],
        'evidenceCount': [100, 200, 50],
    })
    result = GroupedSumMetric(name='s', column='evidenceCount', group_by=['aggregationValue']).compute(df)
    assert isinstance(result, GroupedSumResult)
    groups = {g.key['aggregationValue']: g.count for g in result.groups}
    assert groups['genetics'] == 300
    assert groups['literature'] == 50


# ── GroupedCountExplodeMetric ─────────────────────────────────────────────────

def test_grouped_count_explode_counts_per_exploded_value():
    df = pl.DataFrame({'therapeuticAreas': [['TA1', 'TA2'], ['TA1', 'TA3']]})
    result = GroupedCountExplodeMetric(name='g', group_by=['therapeuticAreas']).compute(df)
    assert isinstance(result, GroupedCountExplodeResult)
    counts = {g.key['therapeuticAreas']: g.count for g in result.groups}
    assert counts['TA1'] == 2
    assert counts['TA2'] == 1
    assert counts['TA3'] == 1


def test_grouped_count_explode_sorted_descending():
    df = pl.DataFrame({'therapeuticAreas': [['TA1', 'TA2'], ['TA1', 'TA3'], ['TA1']]})
    result = GroupedCountExplodeMetric(name='g', group_by=['therapeuticAreas']).compute(df)
    assert result.groups[0].key == {'therapeuticAreas': 'TA1'}
    assert result.groups[0].count == 3


def test_grouped_count_explode_excludes_null_rows():
    df = pl.DataFrame({'therapeuticAreas': [['TA1'], None, ['TA1', 'TA2']]})
    result = GroupedCountExplodeMetric(name='g', group_by=['therapeuticAreas']).compute(df)
    counts = {g.key['therapeuticAreas']: g.count for g in result.groups}
    assert counts == {'TA1': 2, 'TA2': 1}


@pytest.mark.parametrize('df', [
    pl.DataFrame({'therapeuticAreas': pl.Series([], dtype=pl.List(pl.String))}),
    pl.DataFrame({'therapeuticAreas': pl.Series([None, None], dtype=pl.List(pl.String))}),
])
def test_grouped_count_explode_returns_empty_groups(df):
    result = GroupedCountExplodeMetric(name='g', group_by=['therapeuticAreas']).compute(df)
    assert result.groups == []

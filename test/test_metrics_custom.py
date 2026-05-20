"""Tests for custom metric types (Issues 5 and 6) — TDD, written before implementation."""
import polars as pl
import pytest

from pts.metrics.custom import DiseaseByTherapeuticAreaMetric, L2GSignificantGeneMetric
from pts.metrics.grouped import GroupedCountResult
from pts.metrics.count import CountResult


# ── Issue 5: DiseaseByTherapeuticAreaMetric ──────────────────────────────────

def test_disease_by_therapeutic_area_counts_per_area():
    df = pl.DataFrame({
        'id': ['D1', 'D2', 'D3'],
        'therapeuticAreas': [['ONCOLOGY', 'NEURO'], ['ONCOLOGY'], ['NEURO']],
    })
    result = DiseaseByTherapeuticAreaMetric(name='ta').compute(df)
    assert isinstance(result, GroupedCountResult)
    groups = {g.key['therapeuticAreas']: g.count for g in result.groups}
    assert groups['ONCOLOGY'] == 2
    assert groups['NEURO'] == 2


def test_disease_by_therapeutic_area_excludes_nulls():
    df = pl.DataFrame({
        'id': ['D1', 'D2'],
        'therapeuticAreas': [['ONCOLOGY'], None],
    })
    result = DiseaseByTherapeuticAreaMetric(name='ta').compute(df)
    assert len(result.groups) == 1


def test_disease_by_therapeutic_area_group_by_field():
    df = pl.DataFrame({'id': ['D1'], 'therapeuticAreas': [['X']]})
    result = DiseaseByTherapeuticAreaMetric(name='ta').compute(df)
    assert result.group_by == ['therapeuticAreas']


# ── Issue 6: L2GSignificantGeneMetric ────────────────────────────────────────

def test_l2g_counts_genes_above_threshold():
    df = pl.DataFrame({
        'geneId': ['G1', 'G1', 'G2', 'G3'],
        'score':  [0.8,  0.6,  0.3,  0.9],
    })
    result = L2GSignificantGeneMetric(name='sig', threshold=0.5).compute(df)
    assert isinstance(result, CountResult)
    assert result.value == 2  # G1 (max 0.8 >= 0.5) and G3 (0.9 >= 0.5)


def test_l2g_threshold_exclusive_below():
    df = pl.DataFrame({'geneId': ['G1'], 'score': [0.49]})
    result = L2GSignificantGeneMetric(name='sig', threshold=0.5).compute(df)
    assert result.value == 0


def test_l2g_default_threshold_is_0_5():
    m = L2GSignificantGeneMetric(name='sig')
    assert m.threshold == 0.5

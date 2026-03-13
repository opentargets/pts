from __future__ import annotations

from typing import cast

import polars as pl
from otter.config.model import Config

from pts.transformers import release_metrics as release_metrics_module
from pts.transformers.release_metrics import _calculate_metrics, release_metrics


def _input_tables() -> dict[str, pl.DataFrame]:
    evidence = pl.DataFrame(
        {
            'datasourceId': ['ds1', 'ds1', 'ds2'],
            'targetFromSourceId': ['T1', 'T2', None],
            'diseaseFromSourceMappedId': ['D1', None, 'D2'],
            'drugId': ['DR1', 'DR1', None],
            'literature': [['PMID:1'], [], None],
        }
    )

    evidence_failed = pl.DataFrame(
        {
            'id': ['e1', 'e2', 'e3'],
            'datasourceId': ['ds1', 'ds2', 'ds2'],
            'qualityControls': [
                ['Duplicated', 'No valid score'],
                ['No valid target'],
                ['No valid disease'],
            ],
        }
    )

    associations_direct = pl.DataFrame(
        {
            'datasourceId': ['ds1', 'ds1', 'ds2'],
            'diseaseId': ['D1', 'D1', 'D2'],
            'targetId': ['T1', 'T1', 'T2'],
        }
    )

    associations_indirect = pl.DataFrame(
        {
            'datasourceId': ['ds1', 'ds2'],
            'diseaseId': ['D3', 'D2'],
            'targetId': ['T3', 'T2'],
        }
    )

    diseases = pl.DataFrame({'id': ['D1', 'D2'], 'name': ['n1', None]})
    targets = pl.DataFrame({'id': ['T1', 'T2', 'T3']})
    drugs = pl.DataFrame({'id': ['DR1', 'DR2'], 'description': ['d1', None]})

    return {
        'evidence': evidence,
        'evidence_failed': evidence_failed,
        'associations_direct': associations_direct,
        'associations_indirect': associations_indirect,
        'diseases': diseases,
        'targets': targets,
        'drugs': drugs,
    }


def _metric_value(metrics: pl.DataFrame, variable: str, datasource: str = 'all') -> int:
    return int(
        metrics
        .filter(pl.col('variable') == variable, pl.col('datasourceId') == datasource)
        .select('value')
        .item()
    )


def test_release_metrics_schema_and_variables() -> None:
    tables = _input_tables()
    metrics = _calculate_metrics(
        evidence=tables['evidence'],
        evidence_failed=tables['evidence_failed'],
        associations_direct=tables['associations_direct'],
        associations_indirect=tables['associations_indirect'],
        diseases=tables['diseases'],
        targets=tables['targets'],
        drugs=tables['drugs'],
        run_id='26.03_2026-03-13',
    )

    assert metrics.columns == ['datasourceId', 'variable', 'field', 'value', 'runId']
    assert set(metrics['variable'].unique()) == {
        'associationsDirectByDatasource',
        'associationsDirectTotalCount',
        'associationsIndirectByDatasource',
        'associationsIndirectTotalCount',
        'diseasesNotNullCount',
        'diseasesTotalCount',
        'drugsNotNullCount',
        'drugsTotalCount',
        'evidenceCountByDatasource',
        'evidenceDistinctFieldsCountByDatasource',
        'evidenceDuplicateCountByDatasource',
        'evidenceDuplicateTotalCount',
        'evidenceFieldNotNullCountByDatasource',
        'evidenceInvalidCountByDatasource',
        'evidenceInvalidTotalCount',
        'evidenceNullifiedScoreCountByDatasource',
        'evidenceNullifiedScoreTotalCount',
        'evidenceTotalCount',
        'evidenceUnresolvedDiseaseCountByDatasource',
        'evidenceUnresolvedDiseaseTotalCount',
        'evidenceUnresolvedTargetCountByDatasource',
        'evidenceUnresolvedTargetTotalCount',
        'targetsTotalCount',
    }


def test_release_metrics_core_counts() -> None:
    tables = _input_tables()
    metrics = _calculate_metrics(
        evidence=tables['evidence'],
        evidence_failed=tables['evidence_failed'],
        associations_direct=tables['associations_direct'],
        associations_indirect=tables['associations_indirect'],
        diseases=tables['diseases'],
        targets=tables['targets'],
        drugs=tables['drugs'],
        run_id='26.03_2026-03-13',
    )

    assert _metric_value(metrics, 'evidenceTotalCount') == 3
    assert _metric_value(metrics, 'evidenceCountByDatasource', datasource='ds1') == 2
    assert _metric_value(metrics, 'evidenceInvalidTotalCount') == 3
    assert _metric_value(metrics, 'evidenceDuplicateTotalCount') == 1
    assert _metric_value(metrics, 'associationsDirectTotalCount') == 2
    assert _metric_value(metrics, 'associationsIndirectTotalCount') == 2
    assert _metric_value(metrics, 'diseasesTotalCount') == 2
    assert _metric_value(metrics, 'targetsTotalCount') == 3
    assert _metric_value(metrics, 'drugsTotalCount') == 2


def test_release_metrics_association_new_schema_counts() -> None:
    tables = _input_tables()

    associations_direct = pl.DataFrame(
        {
            'diseaseId': ['D1', 'D1', 'D2'],
            'targetId': ['T1', 'T1', 'T2'],
            'aggregationType': ['datasourceId', 'datasourceId', 'datasourceId'],
            'aggregationValue': ['ds1', 'ds1', 'ds2'],
        }
    )
    associations_indirect = pl.DataFrame(
        {
            'diseaseId': ['D3', 'D2'],
            'targetId': ['T3', 'T2'],
            'aggregationType': ['datasourceId', 'datasourceId'],
            'aggregationValue': ['ds1', 'ds2'],
        }
    )

    metrics = _calculate_metrics(
        evidence=tables['evidence'],
        evidence_failed=tables['evidence_failed'],
        associations_direct=associations_direct,
        associations_indirect=associations_indirect,
        diseases=tables['diseases'],
        targets=tables['targets'],
        drugs=tables['drugs'],
        run_id='26.03_2026-03-13',
    )

    assert _metric_value(metrics, 'associationsDirectByDatasource', datasource='ds1') == 2
    assert _metric_value(metrics, 'associationsIndirectByDatasource', datasource='ds2') == 1


def test_release_metrics_hf_upload_failure_does_not_fail(tmp_path, monkeypatch) -> None:
    tables = _input_tables()

    source = {
        'evidence': tmp_path / 'evidence.parquet',
        'evidence_failed': tmp_path / 'evidence_failed.parquet',
        'associations_source_direct': tmp_path / 'assoc_direct.parquet',
        'associations_source_indirect': tmp_path / 'assoc_indirect.parquet',
        'diseases': tmp_path / 'diseases.parquet',
        'targets': tmp_path / 'targets.parquet',
        'drugs': tmp_path / 'drugs.parquet',
    }
    tables['evidence'].write_parquet(source['evidence'])
    tables['evidence_failed'].write_parquet(source['evidence_failed'])
    tables['associations_direct'].write_parquet(source['associations_source_direct'])
    tables['associations_indirect'].write_parquet(source['associations_source_indirect'])
    tables['diseases'].write_parquet(source['diseases'])
    tables['targets'].write_parquet(source['targets'])
    tables['drugs'].write_parquet(source['drugs'])

    destination = {
        'parquet': tmp_path / 'out' / 'release_metrics.parquet',
    }
    token_path = tmp_path / 'hf_token'
    token_path.write_text('dummy-token')

    def _boom(*args, **kwargs):
        raise RuntimeError('upload failure')

    monkeypatch.setattr(release_metrics_module, '_upload_metrics_to_hf_hub', _boom)

    release_metrics(
        source=source,
        destination=destination,
        settings={
            'ot_release': '26.03',
            'upload_to_hf_hub': True,
            'hf_token_filename': str(token_path),
            'hf_repo_id': 'opentargets/ot-release-metrics',
            'hf_data_dir': 'metrics',
        },
        config=cast(Config, None),
    )

    assert destination['parquet'].exists()

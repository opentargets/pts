from __future__ import annotations

import polars as pl
from otter.config.model import Config

from pts.transformers import release_metrics as release_metrics_module
from pts.transformers.release_metrics import (
    _emit_association_metrics,
    _emit_evidence_failed_metrics,
    _emit_evidence_metrics,
    _global_rich_metrics,
    release_metrics,
)


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


def _core_metrics_frame(tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    metrics = [
        *_emit_evidence_metrics(tables['evidence']),
        *_emit_evidence_failed_metrics(tables['evidence_failed']),
        *_emit_association_metrics(tables['associations_direct'], 'Direct'),
        *_emit_association_metrics(tables['associations_indirect'], 'Indirect'),
        *_global_rich_metrics(tables['diseases'], 'diseases'),
        *_global_rich_metrics(tables['targets'], 'targets'),
        *_global_rich_metrics(tables['drugs'], 'drugs'),
    ]
    return pl.concat(metrics, how='vertical_relaxed').with_columns(runId=pl.lit('26.03_2026-03-13'))


def test_release_metrics_schema_and_variables() -> None:
    metrics = _core_metrics_frame(_input_tables())

    assert metrics.columns == ['datasourceId', 'variable', 'field', 'value', 'runId']
    assert set(metrics['variable'].unique()) == {
        'associationsDirectDistinctFieldsCountByDatasource',
        'associationsDirectByDatasource',
        'associationsDirectNotNullCountByDatasource',
        'associationsDirectTotalCount',
        'associationsIndirectDistinctFieldsCountByDatasource',
        'associationsIndirectByDatasource',
        'associationsIndirectNotNullCountByDatasource',
        'associationsIndirectTotalCount',
        'diseasesDistinctFieldsCount',
        'diseasesNotNullCount',
        'diseasesTotalCount',
        'drugsDistinctFieldsCount',
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
        'targetsDistinctFieldsCount',
        'targetsNotNullCount',
        'targetsTotalCount',
    }


def test_release_metrics_core_counts() -> None:
    metrics = _core_metrics_frame(_input_tables())

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

    metrics = pl.concat(
        [
            *_emit_association_metrics(associations_direct, 'Direct'),
            *_emit_association_metrics(associations_indirect, 'Indirect'),
        ],
        how='vertical_relaxed',
    ).with_columns(runId=pl.lit('26.03_2026-03-13'))

    assert _metric_value(metrics, 'associationsDirectByDatasource', datasource='ds1') == 2
    assert _metric_value(metrics, 'associationsIndirectByDatasource', datasource='ds2') == 1


def test_generic_rich_metrics_profile() -> None:
    df = pl.DataFrame({'id': ['b1', 'b2'], 'name': ['x', None]})
    metrics = pl.concat(_global_rich_metrics(df, 'biosample'), how='vertical_relaxed').with_columns(
        runId=pl.lit('26.03_2026-03-13')
    )

    assert _metric_value(metrics, 'biosampleTotalCount') == 2
    assert metrics.filter(pl.col('variable') == 'biosampleDistinctFieldsCount').height > 0


def test_generic_rich_metrics_with_heterogeneous_list_struct() -> None:
    df = pl.DataFrame(
        {
            'studyId': ['s1', 's2'],
            'ldPopulationStructure': [
                [{'ldPopulation': 'nfe'}],
                [{'relativeSampleSize': 0.5}],
            ],
        }
    )

    metrics = pl.concat(_global_rich_metrics(df, 'study'), how='vertical_relaxed').with_columns(
        runId=pl.lit('26.03_2026-03-13')
    )

    assert _metric_value(metrics, 'studyTotalCount') == 2
    assert (
        metrics
        .filter(pl.col('variable') == 'studyNotNullCount', pl.col('field') == 'ldPopulationStructure')
        .height
        == 1
    )


def test_release_metrics_hf_upload_failure_does_not_fail(tmp_path, monkeypatch) -> None:
    tables = _input_tables()

    release_root = tmp_path / 'release'
    evidence_dir = release_root / 'output' / 'evidence_test'
    evidence_failed_dir = release_root / 'excluded' / 'evidence' / 'test'
    assoc_direct_dir = release_root / 'output' / 'association_by_datasource_direct'
    assoc_indirect_dir = release_root / 'output' / 'association_by_datasource_indirect'
    diseases_dir = release_root / 'output' / 'disease'
    targets_dir = release_root / 'output' / 'target'
    drugs_dir = release_root / 'output' / 'drug_molecule'

    for directory in [
        evidence_dir,
        evidence_failed_dir,
        assoc_direct_dir,
        assoc_indirect_dir,
        diseases_dir,
        targets_dir,
        drugs_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    source = {
        'evidence': release_root / 'output' / 'evidence_*',
        'evidence_failed': release_root / 'excluded' / 'evidence' / '*',
        'associations_source_direct': assoc_direct_dir,
        'associations_source_indirect': assoc_indirect_dir,
        'diseases': diseases_dir,
        'targets': targets_dir,
        'drugs': drugs_dir,
    }
    tables['evidence'].write_parquet(evidence_dir / 'part-0000.parquet')
    tables['evidence_failed'].write_parquet(evidence_failed_dir / 'part-0000.parquet')
    tables['associations_direct'].write_parquet(assoc_direct_dir / 'part-0000.parquet')
    tables['associations_indirect'].write_parquet(assoc_indirect_dir / 'part-0000.parquet')
    tables['diseases'].write_parquet(diseases_dir / 'part-0000.parquet')
    tables['targets'].write_parquet(targets_dir / 'part-0000.parquet')
    tables['drugs'].write_parquet(drugs_dir / 'part-0000.parquet')

    destination = {
        'parquet': tmp_path / 'out' / 'release_metrics.parquet',
    }
    token_path = tmp_path / 'hf_token'
    token_path.write_text('dummy-token')

    def _boom(*args, **kwargs):
        raise RuntimeError('upload failure')

    monkeypatch.setattr(release_metrics_module, '_upload_metrics_to_hf_hub', _boom)
    monkeypatch.setattr(
        release_metrics_module,
        '_discover_dataset_paths',
        lambda release_uri, scope_globs, config: {
            '/output/evidence_test': str(evidence_dir),
            '/excluded/evidence/test': str(evidence_failed_dir),
            '/output/association_by_datasource_direct': str(assoc_direct_dir),
            '/output/association_by_datasource_indirect': str(assoc_indirect_dir),
            '/output/disease': str(diseases_dir),
            '/output/target': str(targets_dir),
            '/output/drug_molecule': str(drugs_dir),
        },
    )

    release_metrics(
        source=source,
        destination=destination,
        settings={
            'ot_release': '26.03',
            'metric_scopes': ['/output/*', '/excluded/evidence/*'],
            'rich_dataset_whitelist': [
                '/output/evidence_*',
                '/excluded/evidence/*',
                '/output/association_by_datasource_direct',
                '/output/association_by_datasource_indirect',
                '/output/disease',
                '/output/target',
                '/output/drug_molecule',
            ],
            'upload_to_hf_hub': True,
            'hf_token_filename': str(token_path),
            'hf_repo_id': 'opentargets/ot-release-metrics',
            'hf_data_dir': 'metrics',
        },
        config=Config.model_validate(
            {
                'runner_name': 'pts',
                'step': 'release_metrics',
                'steps': ['release_metrics'],
                'work_path': str(tmp_path / 'work'),
                'release_uri': 'gs://open-targets-pipeline-runs/test/release',
            }
        ),
    )

    assert destination['parquet'].exists()

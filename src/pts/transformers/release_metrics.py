"""Release-level metrics generation in Polars."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
from huggingface_hub import HfApi
from loguru import logger
from otter.config.model import Config


def _document_total_count(df: pl.DataFrame, variable: str) -> pl.DataFrame:
    return pl.DataFrame({
        'datasourceId': ['all'],
        'variable': [variable],
        'field': [None],
        'value': [df.height],
    })


def _document_count_by(df: pl.DataFrame, column: str, variable: str) -> pl.DataFrame:
    return (
        df
        .group_by(column)
        .len()
        .rename({column: 'datasourceId', 'len': 'value'})
        .with_columns(
            variable=pl.lit(variable),
            field=pl.lit(None, dtype=pl.String),
        )
        .select('datasourceId', 'variable', 'field', 'value')
    )


def _flatten_columns(schema: pl.Schema) -> list[str]:
    flat_names: list[str] = []

    def _walk(prefix: str | None, dtype: pl.DataType) -> None:
        if isinstance(dtype, pl.List):
            _walk(prefix, dtype.inner)
            return

        if isinstance(dtype, pl.Struct):
            for field in dtype.fields:
                name = f'{prefix}.{field.name}' if prefix else field.name
                _walk(name, field.dtype)
            return

        if prefix is not None:
            flat_names.append(prefix)

    for col_name, col_dtype in schema.items():
        _walk(col_name, col_dtype)

    return flat_names


def _flattened_select_expressions(df: pl.DataFrame) -> tuple[list[pl.Expr], list[str]]:
    expressions: list[pl.Expr] = []
    output_names: list[str] = []

    def _walk(path: str, dtype: pl.DataType, expr: pl.Expr, inside_list: bool) -> None:
        if isinstance(dtype, pl.List):
            _walk(path, dtype.inner, expr, inside_list=True)
            return

        if isinstance(dtype, pl.Struct):
            for field in dtype.fields:
                child_path = f'{path}.{field.name}'
                child_expr = (
                    expr.list.eval(pl.element().struct.field(field.name))
                    if inside_list
                    else expr.struct.field(field.name)
                )
                _walk(child_path, field.dtype, child_expr, inside_list)
            return

        expressions.append(expr.alias(path))
        output_names.append(path)

    for col_name, col_dtype in df.schema.items():
        _walk(col_name, col_dtype, pl.col(col_name), inside_list=False)

    return expressions, output_names


def _not_null_fields_count(df: pl.DataFrame, variable: str, group_by_datasource: bool) -> pl.DataFrame:
    select_exprs, flat_column_names = _flattened_select_expressions(df)
    flat_df = df.select(select_exprs)

    columns_to_count = flat_column_names
    if group_by_datasource:
        columns_to_count = [column for column in columns_to_count if column != 'datasourceId']

    count_exprs = []
    for column in columns_to_count:
        column_expr = pl.col(column)
        if isinstance(flat_df.schema[column], pl.List):
            count_exprs.append(column_expr.list.get(0, null_on_oob=True).is_not_null().sum().alias(column))
        else:
            count_exprs.append(column_expr.is_not_null().sum().alias(column))

    if group_by_datasource:
        aggregated = flat_df.group_by('datasourceId').agg(count_exprs)
        id_vars = ['datasourceId']
    else:
        aggregated = flat_df.select(count_exprs)
        id_vars = []

    cleaned_columns = {name: name.replace('.', '_') for name in aggregated.columns}
    cleaned = aggregated.rename(cleaned_columns)
    value_columns = [column for column in cleaned.columns if column not in id_vars]

    melted = cleaned.unpivot(
        on=value_columns,
        index=id_vars,
        variable_name='field',
        value_name='value',
    ).with_columns(variable=pl.lit(variable))

    if not group_by_datasource:
        melted = melted.with_columns(datasourceId=pl.lit('all'))

    return melted.select('datasourceId', 'variable', 'field', 'value')


def _distinct_fields_count(df: pl.DataFrame, variable: str) -> pl.DataFrame:
    flat_columns = _flatten_columns(df.schema)
    flat_df = df.select([pl.col(column).alias(column) for column in flat_columns])

    value_columns = [column for column in flat_df.columns if column != 'datasourceId']
    unique_exprs = [pl.col(column).n_unique().alias(column.replace('.', '_')) for column in value_columns]

    aggregated = flat_df.group_by('datasourceId').agg(unique_exprs)
    melted = aggregated.unpivot(
        on=[column for column in aggregated.columns if column != 'datasourceId'],
        index=['datasourceId'],
        variable_name='field',
        value_name='value',
    )

    return melted.with_columns(variable=pl.lit(variable)).select('datasourceId', 'variable', 'field', 'value')


def _get_columns_to_report(dataset_columns: list[str]) -> list[str]:
    return [
        'datasourceId',
        'targetFromSourceId',
        'diseaseFromSourceMappedId' if 'diseaseFromSourceMappedId' in dataset_columns else 'diseaseFromSourceId',
        'drugId',
        'literature',
    ]


def _build_run_id(ot_release: str) -> str:
    release_timestamp = datetime.today().strftime('%Y-%m-%d')
    run_release = ot_release
    if run_release.startswith('partners/'):
        run_release = run_release.split('/', maxsplit=1)[1] + '_ppp'
    return f'{run_release}_{release_timestamp}'


def _upload_metrics_to_hf_hub(
    csv_data: bytes,
    csv_filename: str,
    token_filename: Path,
    repo_id: str,
    data_dir: str,
) -> None:
    hf_token = token_filename.read_text().strip()
    if not hf_token:
        msg = f'HF token file is empty: {token_filename}'
        raise ValueError(msg)

    api = HfApi(token=hf_token)
    api.upload_file(
        path_or_fileobj=csv_data,
        path_in_repo=f'{data_dir}/{csv_filename}',
        repo_id=repo_id,
        repo_type='dataset',
    )


def _calculate_metrics(
    evidence: pl.DataFrame,
    evidence_failed: pl.DataFrame,
    associations_direct: pl.DataFrame,
    associations_indirect: pl.DataFrame,
    diseases: pl.DataFrame,
    targets: pl.DataFrame,
    drugs: pl.DataFrame,
    run_id: str,
) -> pl.DataFrame:
    columns_to_report = _get_columns_to_report(evidence.columns)
    datasets: list[pl.DataFrame] = [
        _document_total_count(evidence, 'evidenceTotalCount'),
        _document_count_by(evidence, 'datasourceId', 'evidenceCountByDatasource'),
        _not_null_fields_count(evidence, 'evidenceFieldNotNullCountByDatasource', group_by_datasource=True),
        _distinct_fields_count(evidence.select(columns_to_report), 'evidenceDistinctFieldsCountByDatasource'),
        _document_total_count(evidence_failed, 'evidenceInvalidTotalCount'),
        _document_total_count(
            evidence_failed.filter(pl.col('qualityControls').list.contains('Duplicated')),
            'evidenceDuplicateTotalCount',
        ),
        _document_total_count(
            evidence_failed.filter(pl.col('qualityControls').list.contains('No valid score')),
            'evidenceNullifiedScoreTotalCount',
        ),
        _document_total_count(
            evidence_failed.filter(pl.col('qualityControls').list.contains('No valid target')),
            'evidenceUnresolvedTargetTotalCount',
        ),
        _document_total_count(
            evidence_failed.filter(pl.col('qualityControls').list.contains('No valid disease')),
            'evidenceUnresolvedDiseaseTotalCount',
        ),
        _document_count_by(evidence_failed, 'datasourceId', 'evidenceInvalidCountByDatasource'),
        _document_count_by(
            evidence_failed.filter(pl.col('qualityControls').list.contains('Duplicated')),
            'datasourceId',
            'evidenceDuplicateCountByDatasource',
        ),
        _document_count_by(
            evidence_failed.filter(pl.col('qualityControls').list.contains('No valid score')),
            'datasourceId',
            'evidenceNullifiedScoreCountByDatasource',
        ),
        _document_count_by(
            evidence_failed.filter(pl.col('qualityControls').list.contains('No valid target')),
            'datasourceId',
            'evidenceUnresolvedTargetCountByDatasource',
        ),
        _document_count_by(
            evidence_failed.filter(pl.col('qualityControls').list.contains('No valid disease')),
            'datasourceId',
            'evidenceUnresolvedDiseaseCountByDatasource',
        ),
        _document_total_count(
            associations_direct.select('diseaseId', 'targetId').unique(),
            'associationsDirectTotalCount',
        ),
        _document_count_by(associations_direct, 'datasourceId', 'associationsDirectByDatasource'),
        _document_total_count(
            associations_indirect.select('diseaseId', 'targetId').unique(),
            'associationsIndirectTotalCount',
        ),
        _document_count_by(associations_indirect, 'datasourceId', 'associationsIndirectByDatasource'),
        _document_total_count(diseases, 'diseasesTotalCount'),
        _not_null_fields_count(diseases, 'diseasesNotNullCount', group_by_datasource=False),
        _document_total_count(targets, 'targetsTotalCount'),
        _document_total_count(drugs, 'drugsTotalCount'),
        _not_null_fields_count(drugs, 'drugsNotNullCount', group_by_datasource=False),
    ]

    return (
        pl
        .concat(datasets, how='vertical_relaxed')
        .with_columns(runId=pl.lit(run_id))
        .select(
            'datasourceId',
            'variable',
            'field',
            'value',
            'runId',
        )
    )


def release_metrics(
    source: dict[str, Path],
    destination: dict[str, Path],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Generate release metrics table.

    Args:
        source: Required source paths for post-ETL outputs.
        destination: Destination paths, requires `parquet`.
        settings: Runtime settings, requires `ot_release`.
        config: Otter config object.
    """
    del config

    ot_release = settings['ot_release']
    run_id = _build_run_id(ot_release)

    logger.info(f'Loading metrics inputs for release {ot_release}')
    evidence = pl.read_parquet(source['evidence'])
    evidence_failed = pl.read_parquet(source['evidence_failed'])
    associations_direct = pl.read_parquet(source['associations_source_direct'])
    associations_indirect = pl.read_parquet(source['associations_source_indirect'])
    diseases = pl.read_parquet(source['diseases'])
    targets = pl.read_parquet(source['targets'])
    drugs = pl.read_parquet(source['drugs'])

    logger.info('Calculating release metrics')
    metrics = _calculate_metrics(
        evidence=evidence,
        evidence_failed=evidence_failed,
        associations_direct=associations_direct,
        associations_indirect=associations_indirect,
        diseases=diseases,
        targets=targets,
        drugs=drugs,
        run_id=run_id,
    )

    logger.info(f'Writing metrics parquet to {destination["parquet"]}')
    metrics.write_parquet(destination['parquet'], mkdir=True)

    if settings.get('upload_to_hf_hub'):
        hf_token_filename = settings.get('hf_token_filename')
        if not hf_token_filename:
            logger.warning('HF upload requested but settings["hf_token_filename"] is missing; skipping upload')
            return

        hf_repo_id = settings.get('hf_repo_id', 'opentargets/ot-release-metrics')
        hf_data_dir = settings.get('hf_data_dir', 'metrics')
        csv_filename = f'{run_id}.csv'
        csv_data = metrics.write_csv(include_header=True).encode('utf-8')

        try:
            _upload_metrics_to_hf_hub(
                csv_data=csv_data,
                csv_filename=csv_filename,
                token_filename=Path(hf_token_filename),
                repo_id=hf_repo_id,
                data_dir=hf_data_dir,
            )
            logger.info(f'Uploaded metrics CSV to hf://datasets/{hf_repo_id}/{hf_data_dir}/{csv_filename}')
        except Exception:
            logger.exception('HF upload failed; continuing without failing the step')

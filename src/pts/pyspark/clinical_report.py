"""Clinical report dataset generation."""

import polars as pl
from clinical_mining.data_sources.aact import extract_clinical_report as extract_aact_clinical_report
from clinical_mining.data_sources.chembl.drug_warnings import (
    extract_clinical_report as extract_drug_warning_clinical_report,
)
from clinical_mining.data_sources.chembl.indications import extract_clinical_report as extract_chembl_clinical_report
from clinical_mining.data_sources.ema import extract_clinical_report as extract_ema_clinical_report
from clinical_mining.data_sources.pmda import extract_clinical_report as extract_pmda_clinical_report
from clinical_mining.data_sources.pmda import parse_pmda_approvals
from clinical_mining.data_sources.ttd import extract_clinical_report as extract_ttd_clinical_report
from clinical_mining.dataset import ClinicalReport
from clinical_mining.utils.db import construct_db_uri, load_db_table
from clinical_mining.utils.polars_helpers import union_dfs
from clinical_mining.utils.spark_helpers import spark_session
from loguru import logger


def clinical_report(
    source: dict[str, str], destination: dict[str, str], settings: dict[str, str], properties: dict[str, str]
) -> None:
    """Generate clinical report dataset from ChEMBL molecules and disease data.

    Args:
        source: Dictionary containing paths to input data
        destination: Dictionary containing paths to output data:
            - output: Path to write clinical reports (output/clinical_report)
        settings: Dictionary containing step variables
        properties: Dictionary containing Spark properties
    """
    logger.info(f'source paths: {source}')
    aact_uri = construct_db_uri(
        db_type=settings['aact_db_type'],
        db_uri=settings['aact_db_uri'],
    )
    chembl_uri = construct_db_uri(
        db_type=settings['chembl_db_type'],
        db_uri=settings['chembl_db_uri'],
    )
    spark = spark_session()

    # molecule_index_spark = spark.read.parquet(source['chembl_molecule'])
    molecule_index_spark = spark.read.parquet('/Users/irenelopez/EBI/repos/pts/work/intermediate/chembl_molecule')
    disease_index_spark = spark.read.parquet(source['disease'])
    chembl_curation = pl.read_parquet(source['chembl_curation']) if 'chembl_curation' in source else None

    # TODO: replace db loading
    aact_studies = load_db_table(
        table_name='studies',
        db_url=aact_uri,
        db_schema='ctgov',
        select_cols=[
            'nct_id',
            'overall_status',
            'phase',
            'study_type',
            'start_date',
            'why_stopped',
            'number_of_arms',
            'official_title',
        ],
    )
    aact_interventions = load_db_table(
        table_name='interventions',
        db_url=aact_uri,
        db_schema='ctgov',
        select_cols=['nct_id', 'intervention_type', 'name'],
    )
    aact_conditions = load_db_table(
        table_name='conditions', db_url=aact_uri, db_schema='ctgov', select_cols=['nct_id', 'downcase_name']
    )
    aact_study_references = load_db_table(
        table_name='study_references',
        db_url=aact_uri,
        db_schema='ctgov',
        select_cols=['nct_id', 'pmid', 'reference_type'],
    )
    aact_designs = load_db_table(
        table_name='designs', db_url=aact_uri, db_schema='ctgov', select_cols=['nct_id', 'primary_purpose']
    )
    aact_summaries = load_db_table(
        table_name='brief_summaries', db_url=aact_uri, db_schema='ctgov', select_cols=['nct_id', 'description']
    )
    chembl_indication = load_db_table(
        table_name='drug_indication',
        db_url=chembl_uri,
        db_schema='public',
        select_cols=['drugind_id', 'molregno', 'max_phase_for_ind', 'efo_id', 'efo_term'],
    )
    chembl_indication_references = load_db_table(
        table_name='indication_refs',
        db_url=chembl_uri,
        db_schema='public',
        select_cols=['drugind_id', 'ref_type', 'ref_id', 'ref_url'],
    )
    chembl_molecule = load_db_table(
        table_name='molecule_dictionary', db_url=chembl_uri, db_schema='public', select_cols=['molregno', 'chembl_id']
    )
    chembl_drug_warning = load_db_table(
        table_name='drug_warning',
        db_url=chembl_uri,
        db_schema='public',
        select_cols=['warning_id', 'molregno', 'warning_type', 'warning_year', 'warning_country', 'efo_id', 'efo_term'],
    )
    chembl_drug_warning_references = load_db_table(
        table_name='warning_refs',
        db_url=chembl_uri,
        db_schema='public',
        select_cols=['warning_id', 'ref_type', 'ref_id', 'ref_url'],
    )

    logger.info('extract clinical report')
    pmda = extract_pmda_clinical_report(df=parse_pmda_approvals(pmda_path=source['pmda']), spark=spark)
    aact = extract_aact_clinical_report(
        studies=aact_studies,
        interventions=aact_interventions,
        conditions=aact_conditions,
        additional_metadata=[aact_study_references, aact_designs, aact_summaries],
        aggregation_specs={'pmid': {'group_by': 'nct_id', 'alias': 'literature'}},
    )  # TODO: join with stop reasons
    chembl_indication = extract_chembl_clinical_report(
        drug_indication=chembl_indication,
        molecule_dictionary=chembl_molecule,
        indication_refs=chembl_indication_references,
    )
    chembl_drug_warning = extract_drug_warning_clinical_report(
        drug_warning=chembl_drug_warning,
        molecule_dictionary=chembl_molecule,
        warning_refs=chembl_drug_warning_references,
    )
    ttd = extract_ttd_clinical_report(indications_path=source['ttd'])
    ema = extract_ema_clinical_report(indications_path=source['ema'], spark=spark)

    reports = union_dfs([pmda.df, aact.df, chembl_indication.df, chembl_drug_warning.df, ttd.df, ema.df])
    logger.info('reports generated. map entities...')
    mapped_reports = ClinicalReport.map_entities(
        spark=spark,
        reports=reports,
        disease_index=disease_index_spark,
        drug_index=molecule_index_spark,
        chembl_curation=chembl_curation,
        drug_column_name='drugFromSource',
        disease_column_name='diseaseFromSource',
        ner_extract_drug=True,
        ner_batch_size=settings['ner_batch_size'],
        # ner_cache_path=source['ner_cache_path'],
        ner_cache_path='/Users/irenelopez/EBI/repos/pts/work/input/clinical_report/ner_cache.parquet',
    )

    output = validate_disease(mapped_reports, disease_index=pl.read_parquet(source['disease']))

    logger.info(f'destination paths: {destination}')
    output.df.write_parquet(destination['output'])


def validate_disease(reports: ClinicalReport, disease_index: pl.DataFrame) -> ClinicalReport:
    """Validate disease entities in the reports.

    Args:
        reports: ClinicalReport object with mapped entities
        disease_index: Polars DataFrame with disease index

    Returns:
        ClinicalReport object with validated disease entities
    """
    exploded = reports.df.explode('diseases').unnest('diseases')
    diseases = disease_index.select(pl.col('id').alias('diseaseId'), 'obsoleteTerms').explode('obsoleteTerms')

    # Find valid IDs (those that exist in diseases dataframe or are null in the first place)
    null_ids = exploded.filter(pl.col('diseaseId').is_null())
    non_null = exploded.filter(pl.col('diseaseId').is_not_null())
    valid_ids = non_null.join(diseases.select('diseaseId'), on='diseaseId', how='semi')

    obsolete_ids = non_null.join(
        diseases.select('diseaseId'),
        on='diseaseId',
        how='anti',  # Keep only rows where diseaseId doesn't exist in diseases
    )
    updated_ids = (
        obsolete_ids.join(
            diseases.select('diseaseId', 'obsoleteTerms'), left_on='diseaseId', right_on='obsoleteTerms', how='left'
        )
        .with_columns([
            # Replace obsolete ID with current term, or null if not found
            pl.when(pl.col('diseaseId_right').is_not_null())
            .then(pl.col('diseaseId_right'))
            .otherwise(None)
            .alias('diseaseId'),
        ])
        .drop('diseaseId_right')
    )

    combined = pl.concat([null_ids, valid_ids, updated_ids])
    return ClinicalReport(
        df=(
            # Reconstruct the nested disease structure
            combined.group_by(reports.df.drop('diseases').columns, maintain_order=True).agg([
                pl.struct(['diseaseFromSource', 'diseaseId']).alias('diseases')
            ])
        )
    )

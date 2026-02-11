"""Clinical report dataset generation."""

from pathlib import Path

import polars as pl
from clinical_mining.data_sources.aact import extract_clinical_report as extract_aact_clinical_report
from clinical_mining.data_sources.chembl.drug_warning import (
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


def clinical_report(source: dict[str, Path], destination: dict[str, Path], properties: dict[str, str]) -> None:
    """Generate clinical report dataset from ChEMBL molecules and disease data.

    Args:
        source: Dictionary containing paths to input data
        destination: Dictionary containing paths to output data:
            - output: Path to write clinical reports (output/clinical_report)
        properties: Dictionary containing step variables
    """
    logger.info(f'Source paths: {source}')
    aact_uri = construct_db_uri(
        db_type=properties['aact_db_type'],
        db_uri=properties['aact_db_uri'],
    )
    chembl_uri = construct_db_uri(
        db_type=properties['chembl_db_type'],
        db_uri=properties['chembl_db_uri'],
    )
    spark = spark_session()

    molecule_index_spark = spark.read.parquet(source['chembl_molecule'])
    disease_index_spark = spark.read.parquet(source['disease'])
    chembl_curation = pl.read_parquet(source['chembl_curation']) if 'chembl_curation' in source else None

    aact_studies = load_db_table(
        table_name='studies',
        db_uri=aact_uri,
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
        db_uri=aact_uri,
        db_schema='ctgov',
        select_cols=['nct_id', 'intervention_type', 'name'],
    )
    aact_conditions = load_db_table(
        table_name='conditions', db_uri=aact_uri, db_schema='ctgov', select_cols=['nct_id', 'downcase_name']
    )
    aact_study_references = load_db_table(
        table_name='study_references',
        db_uri=aact_uri,
        db_schema='ctgov',
        select_cols=['nct_id', 'pmid', 'reference_type'],
    )
    aact_designs = load_db_table(
        table_name='designs', db_uri=aact_uri, db_schema='ctgov', select_cols=['nct_id', 'primary_purpose']
    )
    aact_summaries = load_db_table(
        table_name='brief_summaries', db_uri=aact_uri, db_schema='ctgov', select_cols=['nct_id', 'description']
    )
    chembl_indication = load_db_table(
        table_name='drug_indication',
        db_uri=chembl_uri,
        db_schema='public',
        select_cols=['drugind_id', 'molregno', 'max_phase_for_ind', 'efo_id', 'efo_term'],
    )
    chembl_indication_references = load_db_table(
        table_name='indication_refs',
        db_uri=chembl_uri,
        db_schema='public',
        select_cols=['drugind_id', 'ref_type', 'ref_id', 'ref_url'],
    )
    chembl_molecule = load_db_table(
        table_name='molecule_dictionary', db_uri=chembl_uri, db_schema='public', select_cols=['molregno', 'chembl_id']
    )
    chembl_drug_warning = load_db_table(
        table_name='drug_warning',
        db_uri=chembl_uri,
        db_schema='public',
        select_cols=['warning_id', 'molregno', 'warning_type', 'warning_year', 'warning_country', 'efo_id', 'efo_term'],
    )
    chembl_drug_warning_references = load_db_table(
        table_name='warning_refs',
        db_uri=chembl_uri,
        db_schema='public',
        select_cols=['warning_id', 'ref_type', 'ref_id', 'ref_url'],
    )

    pmda = extract_pmda_clinical_report(df=parse_pmda_approvals(pmda_path=source['pmda']), spark=spark)
    aact = extract_aact_clinical_report(
        studies=aact_studies,
        interventions=aact_interventions,
        conditions=aact_conditions,
        additional_metadata=[aact_study_references, aact_designs, aact_summaries],
        aggregation_specs={'pmid': {'group_by': 'nct_id', 'alias': 'literature'}},
    )
    chembl_indication = extract_chembl_clinical_report(
        drug_indication=chembl_indication,
        molectule_dictionary=chembl_molecule,
        indication_refs=chembl_indication_references,
    )
    chembl_drug_warning = extract_drug_warning_clinical_report(
        drug_warning=chembl_drug_warning,
        molectule_dictionary=chembl_molecule,
        warning_refs=chembl_drug_warning_references,
    )
    ttd = extract_ttd_clinical_report(indications_path=source['ttd'])
    ema = extract_ema_clinical_report(indications_path=source['ema'], spark=spark)

    reports = union_dfs([pmda, aact, chembl_indication, chembl_drug_warning, ttd, ema])
    mapped_reports = ClinicalReport.map_entities(
        spark=spark,
        reports=reports,
        disease_index=disease_index_spark,
        drug_index=molecule_index_spark,
        chembl_curation=chembl_curation,
        drug_column_name='drugFromSource',
        disease_column_name='diseaseFromSource',
        ner_extract_drug=True,
        ner_batch_size=properties['ner_batch_size'],
        ner_cache_path=source['ner_cache_path'],
    )

    logger.info(f'Destination paths: {destination}')
    mapped_reports.df.write.mode('overwrite').parquet(destination['output'])

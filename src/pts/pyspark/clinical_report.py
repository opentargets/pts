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
    spark = spark_session()

    # molecule_index_spark = spark.read.parquet(source['chembl_molecule'])
    molecule_index_spark = spark.read.parquet('/Users/irenelopez/EBI/repos/pts/work/intermediate/chembl_molecule')
    disease_index_spark = spark.read.parquet(source['disease'])
    chembl_curation = pl.read_parquet(source['chembl_curation']) if 'chembl_curation' in source else None
    aact_studies = pl.read_parquet(source['aact_studies']).select(
        'nct_id',
        'overall_status',
        'phase',
        'study_type',
        'start_date',
        'why_stopped',
        'number_of_arms',
        'official_title',
    )
    aact_interventions = pl.read_parquet(source['aact_interventions']).select(
        'nct_id',
        'intervention_type',
        'name',
    )
    aact_conditions = pl.read_parquet(source['aact_conditions']).select('nct_id', 'downcase_name')
    aact_study_references = pl.read_parquet(source['aact_study_references']).select('nct_id', 'pmid', 'reference_type')
    aact_designs = pl.read_parquet(source['aact_designs']).select('nct_id', 'primary_purpose')
    aact_summaries = pl.read_parquet(source['aact_summaries']).select('nct_id', 'description')
    chembl_indication = pl.read_parquet(source['chembl_indication']).select(
        'drugind_id', 'molregno', 'max_phase_for_ind', 'efo_id', 'efo_term'
    )
    chembl_indication_references = pl.read_parquet(source['chembl_indication_references']).select(
        'drugind_id', 'ref_type', 'ref_id', 'ref_url'
    )
    chembl_molecule = pl.read_parquet(source['chembl_molecule']).select('molregno', 'chembl_id', 'pref_name')
    chembl_drug_warning = pl.read_parquet(source['chembl_drug_warning']).select(
        'warning_id', 'molregno', 'warning_type', 'warning_year', 'warning_country', 'efo_id', 'efo_term'
    )
    chembl_drug_warning_references = pl.read_parquet(source['chembl_drug_warning_references']).select(
        'warning_id', 'ref_type', 'ref_id', 'ref_url'
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
    output = (
        ClinicalReport.map_entities(
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
        .pipe(validate_disease, disease_index=pl.read_parquet(source['disease']))
        .pipe(validate_phase_iv)
        .pipe(create_title)
    )

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
    logger.info(f'obsolete disease id count: {obsolete_ids.height}')
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


def validate_phase_iv(reports: ClinicalReport) -> ClinicalReport:
    """Remove Phase IV reports annotation if the association is not approved."""
    exploded = reports.df.explode('drugs').explode('diseases').unnest(['drugs', 'diseases'])

    non_phase_iv = exploded.filter(pl.col('clinicalStage') != 'PHASE_4')
    phase_iv = (
        exploded.filter(pl.col('clinicalStage') == 'PHASE_4')
        .filter(pl.col('diseaseId').is_not_null() | pl.col('drugId').is_not_null())
        .unique()
    )
    logger.info(f'validate phase iv reports... Original count: {phase_iv.select("id").unique().height}')

    approved = (
        (exploded.filter(pl.col('clinicalStage').is_in(['APPROVED', 'PREAPPROVAL'])))
        .filter(pl.col('diseaseId').is_not_null() | pl.col('drugId').is_not_null())
        .unique()
    )

    approved_phase_iv = phase_iv.join(
        approved.select(['drugId', 'diseaseId']), on=['drugId', 'diseaseId'], how='semi'
    ).unique()
    logger.info(f'validate phase iv reports... Approved count: {approved_phase_iv.select("id").unique().height}')

    combined = pl.concat([non_phase_iv, approved_phase_iv])
    return ClinicalReport(
        df=(
            # Reconstruct the nested disease structure
            combined.group_by(reports.df.drop('diseases', 'drugs').columns, maintain_order=True).agg([
                pl.struct(['diseaseFromSource', 'diseaseId']).alias('diseases'),
                pl.struct(['drugFromSource', 'drugId']).alias('drugs'),
            ])
        )
    )


def create_title(reports: ClinicalReport) -> ClinicalReport:
    """Create a title for the report based on the information on drugs and diseases."""
    return ClinicalReport(
        df=(
            reports.df.with_columns(
                drugs_count=pl.col('drugs').list.len(),
                diseases_count=pl.col('diseases').list.len(),
            )
            .with_columns(
                # Building blocks for the report description
                _stage=pl.col('clinicalStage').str.to_titlecase().str.replace_all('_', ' '),
                _drug=pl.col('drugs').list.first().struct.field('drugFromSource').str.to_titlecase(),
                _disease=pl.col('diseases').list.first().struct.field('diseaseFromSource').str.to_titlecase(),
                _drug_part=pl.when(pl.col('drugs_count') == 1)
                .then(pl.col('drugs').list.first().struct.field('drugFromSource').str.to_titlecase())
                .otherwise(pl.concat_str(pl.col('drugs_count').cast(pl.String), pl.lit(' molecules'))),
                _disease_part=pl.when(pl.col('diseases_count') == 1)
                .then(pl.col('diseases').list.first().struct.field('diseaseFromSource').str.to_titlecase())
                .otherwise(pl.concat_str(pl.col('diseases_count').cast(pl.String), pl.lit(' diseases'))),
                _source_part=pl.when(pl.col('type') == 'REGULATORY_AGENCY')
                .then(pl.concat_str(pl.lit(' by '), pl.col('source')))
                .otherwise(pl.lit('')),
            )
            .with_columns(
                title=pl.when(pl.col('trialOfficialTitle').is_not_null())
                .then(pl.col('trialOfficialTitle'))
                .otherwise(
                    pl.concat_str(
                        pl.lit('Report in '),
                        pl.col('_stage'),
                        pl.lit(' stage for '),
                        pl.col('_drug_part'),
                        pl.lit(' and '),
                        pl.col('_disease_part'),
                        pl.col('_source_part'),
                        ignore_nulls=True,
                    )
                )
            )
            .drop(
                'drugs_count',
                'diseases_count',
                '_stage',
                '_drug',
                '_disease',
                '_drug_part',
                '_disease_part',
                '_source_part',
            )
        )
    )

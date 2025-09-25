"""Evidence parser for the animal model sources from IMPC."""

import pronto
import pyspark.sql.functions as f
from loguru import logger
from pyspark.sql import DataFrame, Window

from pts.pyspark.common.session import Session
from pts.pyspark.impc_utils import (
    aggregate_mouse_phenotypes,
    build_gene_mapping,
    cleanup_model_identifier,
    find_matched_human_phenotypes,
    format_disease_model_associations,
    map_model_mouse_phenotypes_to_human,
    process_literature_references,
    process_ontology_terms,
)
from pts.utils.ontology import add_efo_mapping


def impc(
    source: dict[str, str],
    destination: str,
    properties: dict[str, str],
) -> DataFrame:
    """Generate IMPC evidence strings."""
    spark = Session(app_name='impc', properties=properties)
    score_cutoff = float(properties['score_cutoff'])
    efo_version = properties['efo_version']
    mapping_cores = int(properties.get('ontology_cores', 1))

    # Load and prepare all required datasets for evidence generation
    datasets = _load_impc_datasets_for_evidence(spark, source)
    datasets = _prepare_impc_datasets_for_evidence(datasets)

    # Process ontology terms and build gene mapping
    mp_terms, hp_terms = process_ontology_terms(datasets['ontology'])
    gene_mapping = build_gene_mapping(
        datasets['mgi_gene_id_to_ensembl_mouse_gene_id'],
        datasets['mouse_to_human_gene'],
        datasets['hgnc_gene_id_to_ensembl_human_gene_id'],
    )

    # Process literature references
    literature = process_literature_references(
        datasets['mgi_pubmed'],
        datasets['disease_model_summary_transformed'],
        datasets['model_mouse_phenotypes_transformed'],
    )

    # Generate evidence strings
    logger.info('generate impc evidence strings')
    evidence = generate_impc_evidence_strings(
        datasets['model_mouse_phenotypes_transformed'],
        datasets['mouse_to_human_phenotype'],
        mp_terms,
        datasets['disease_model_summary_transformed'],
        datasets['disease_human_phenotypes_transformed'],
        hp_terms,
        gene_mapping,
        literature,
        score_cutoff,
    )

    # Apply EFO mapping
    mapped_evidence_df = add_efo_mapping(
        evidence_strings=evidence, spark_instance=spark.spark, efo_version=efo_version, cores=mapping_cores
    )

    # Finalize evidence strings
    final_evidence = _finalise_evidence_strings(mapped_evidence_df)

    # Write evidence data
    logger.info('write impc evidence')
    final_evidence.write.mode('overwrite').parquet(destination)

    return final_evidence


def _load_impc_datasets_for_evidence(spark: Session, source: dict[str, str]) -> dict[str, DataFrame]:
    """Load IMPC datasets required for evidence generation."""
    logger.info(f'load IMPC data for evidence generation from {source}')
    return {
        'mouse_to_human_phenotype': spark.load_data(source['solr_ontology_ontology'], format='csv', header=True),
        'model_mouse_phenotypes': spark.load_data(source['solr_mouse_model'], format='csv', header=True),
        'disease_human_phenotypes': spark.load_data(source['solr_disease'], format='csv', header=True),
        'disease_model_summary': spark.load_data(source['solr_disease_model_summary'], format='csv', header=True),
        'ontology': spark.load_data(source['solr_ontology'], format='csv', header=True),
        'mp_ontology': pronto.Ontology(source['mp_ontology']),
        'mgi_pubmed': spark.load_data(
            source['mouse_pubmed_refs'],
            format='csv',
            schema='_0 string, _1 string, _2 string, mp_id string, literature string, targetInModelMgiId string',
            nullValue='null',
            sep='\\t',
        ),
        'mgi_gene_id_to_ensembl_mouse_gene_id': spark.load_data(
            source['mouse_gene_mappings'], format='csv', header=True, nullValue='null', sep='\\t'
        ),
        'mouse_to_human_gene': spark.load_data(source['solr_gene_gene'], format='csv', header=True),
        'hgnc_gene_id_to_ensembl_human_gene_id': spark.load_data(
            source['hgnc_gene_mappings'], format='csv', header=True, nullValue='null', sep='\\t'
        ),
    }


def _prepare_impc_datasets_for_evidence(datasets: dict[str, DataFrame]) -> dict[str, DataFrame]:
    """Prepare transformed versions of IMPC datasets for evidence generation."""
    # Transform model phenotypes
    datasets['model_mouse_phenotypes_transformed'] = (
        datasets['model_mouse_phenotypes']
        .withColumn('mp_id', f.explode(f.expr(r"regexp_extract_all(model_phenotypes, '(MP:\\d+)', 1)")))
        .select('model_id', 'mp_id')
        .persist()
    )

    # Transform disease phenotypes (only needed for evidence)
    datasets['disease_human_phenotypes_transformed'] = (
        datasets['disease_human_phenotypes']
        .withColumn('hp_id', f.explode(f.expr(r"regexp_extract_all(disease_phenotypes, '(HP:\\d+)', 1)")))
        .select('disease_id', 'hp_id')
        .distinct()
    )

    # Transform disease model summary
    datasets['disease_model_summary_transformed'] = format_disease_model_associations(datasets['disease_model_summary'])

    return datasets


def _finalise_evidence_strings(mapped_evidence: DataFrame) -> DataFrame:
    """Remove duplicates and add final datasource/datatype columns."""
    # Keep only the record with the highest score for each unique combination
    unique_fields = [
        # Specific to IMPC.
        'diseaseFromSource',  # Original disease name.
        'targetInModel',  # Mouse gene name.
        'biologicalModelAllelicComposition',  # Mouse model property.
        'biologicalModelGeneticBackground',  # Mouse model property.
        # General.
        'diseaseFromSourceMappedId',  # EFO mapped disease ID.
        'targetFromSourceId',  # Ensembl mapped human gene ID.
    ]
    w = Window.partitionBy([f.col(c) for c in unique_fields]).orderBy(f.col('resourceScore').desc())
    return (
        mapped_evidence.withColumn('row', f.row_number().over(w))
        .filter(f.col('row') == 1)
        .drop('row')
        .select(
            '*',
            f.lit('impc').alias('datasourceId'),
            f.lit('animal_model').alias('datatypeId'),
        )
    )


def generate_impc_evidence_strings(
    model_mouse_phenotypes: DataFrame,
    mouse_phenotype_to_human_phenotype: DataFrame,
    mp_terms: DataFrame,
    disease_model_summary: DataFrame,
    disease_human_phenotypes: DataFrame,
    hp_terms: DataFrame,
    gene_mapping: DataFrame,
    literature: DataFrame,
    score_cutoff: float,
) -> DataFrame:
    """Generate the evidence by renaming, transforming and joining the columns."""
    # Prepare phenotype mappings
    model_human_phenotypes = map_model_mouse_phenotypes_to_human(
        model_mouse_phenotypes, mouse_phenotype_to_human_phenotype
    )
    all_mouse_phenotypes = aggregate_mouse_phenotypes(model_mouse_phenotypes, mp_terms)
    matched_human_phenotypes = find_matched_human_phenotypes(
        disease_model_summary, disease_human_phenotypes, model_human_phenotypes, hp_terms
    )

    # Build final evidence strings
    return (
        disease_model_summary.filter(~(f.col('resourceScore') < score_cutoff))
        .join(gene_mapping, on='targetInModelMgiId', how='inner')
        .join(all_mouse_phenotypes, on='model_id', how='left')
        .join(matched_human_phenotypes, on=['model_id', 'disease_id'], how='left')
        .join(literature, on=['model_id', 'targetInModelMgiId'], how='left')
        .withColumnRenamed('disease_id', 'diseaseFromSourceId')
        .withColumnRenamed('disease_term', 'diseaseFromSource')
        .withColumn('biologicalModelId', cleanup_model_identifier(f.col('model_id')))
    )

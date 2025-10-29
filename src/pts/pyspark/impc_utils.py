"""Shared utilities for IMPC data processing."""

import pyspark.sql.functions as f
from loguru import logger
from pyspark.sql import Column, DataFrame

from pts.pyspark.common.session import Session


def format_disease_model_associations(disease_model_summary: DataFrame) -> DataFrame:
    """Format disease model associations with proper column names."""
    return disease_model_summary.selectExpr(
        'model_id',
        'model_genetic_background as biologicalModelGeneticBackground',
        'model_description as biologicalModelAllelicComposition',
        'disease_id',
        'disease_term',
        'disease_model_avg_norm',
        'marker_id as targetInModelMgiId',
        'disease_model_avg_norm as resourceScore',
    ).distinct()


def process_ontology_terms(ontology: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Process ontology information to enable MP and HP term lookup based on the ID."""
    logger.info('process ontology information to enable MP and HP term lookup based on the ID.')
    return tuple(
        ontology.filter(f.col('ontology') == ontology_name)
        .withColumnRenamed('phenotype_id', f'{ontology_name.lower()}_id')
        .withColumnRenamed('phenotype_term', f'{ontology_name.lower()}_term')
        .select(f'{ontology_name.lower()}_id', f'{ontology_name.lower()}_term')
        for ontology_name in ('MP', 'HP')
    )


def create_mp_classification(mp_ontology, spark: Session) -> DataFrame:
    """Process MP definitions to extract high level classes for each term."""
    logger.info('process MP definitions to extract high level classes for each term.')
    high_level_classes = set(mp_ontology['MP:0000001'].subclasses(distance=1)) - {mp_ontology['MP:0000001']}
    mp_class_data = [
        [term.id, mp_high_level_class.id, mp_high_level_class.name]
        for mp_high_level_class in high_level_classes
        for term in mp_high_level_class.subclasses()
    ]
    return spark.spark.createDataFrame(
        data=mp_class_data,
        schema=['modelPhenotypeId', 'modelPhenotypeClassId', 'modelPhenotypeClassLabel'],
    )


def build_gene_mapping(
    mgi_gene_id_to_ensembl_mouse_gene_id: DataFrame,
    mouse_to_human_gene: DataFrame,
    hgnc_gene_id_to_ensembl_human_gene_id: DataFrame,
) -> DataFrame:
    """Build complete gene mapping from MGI gene ID to human Ensembl ID."""
    logger.info('construct gene mapping.')
    return (
        mgi_gene_id_to_ensembl_mouse_gene_id.selectExpr(
            '`1. MGI accession id` as targetInModelMgiId',
            '`3. marker symbol` as targetInModel',
            '`11. Ensembl gene id` as targetInModelEnsemblId',
        )
        .filter(f.col('targetInModelEnsemblId').isNotNull())
        .join(
            mouse_to_human_gene.selectExpr('gene_id as targetInModelMgiId', 'hgnc_gene_id'),
            on='targetInModelMgiId',
            how='inner',
        )
        .join(
            hgnc_gene_id_to_ensembl_human_gene_id.selectExpr(
                'hgnc_id as hgnc_gene_id', 'ensembl_gene_id as targetFromSourceId'
            ),
            on='hgnc_gene_id',
            how='inner',
        )
        .filter(f.col('targetFromSourceId').isNotNull())
        .select('targetInModelMgiId', 'targetInModel', 'targetInModelEnsemblId', 'targetFromSourceId')
    )


def process_literature_references(
    mgi_pubmed: DataFrame, disease_model_summary: DataFrame, model_mouse_phenotypes: DataFrame
) -> DataFrame:
    """Process literature references for model-gene combinations."""
    mgi_pubmed_exploded = (
        mgi_pubmed.withColumn('literature', f.explode(f.split(f.col('literature'), r'\\|')))
        .withColumn('targetInModelMgiId', f.explode(f.split(f.col('targetInModelMgiId'), r'\\|')))
        .select('mp_id', 'literature', 'targetInModelMgiId')
        .distinct()
    )

    return (
        disease_model_summary.select('model_id', 'targetInModelMgiId')
        .distinct()
        .join(model_mouse_phenotypes, on='model_id', how='inner')
        .join(mgi_pubmed_exploded, on=['targetInModelMgiId', 'mp_id'], how='inner')
        .groupby('model_id', 'targetInModelMgiId')
        .agg(f.collect_set(f.col('literature')).alias('literature'))
        .select('model_id', 'targetInModelMgiId', 'literature')
    )


def cleanup_model_identifier(model_id_col: Column):
    """Cleanup model identifier by stripping modifiers and filtering MGI namespace."""
    cleaned_id = f.split(model_id_col, '#').getItem(0)
    return f.when(cleaned_id.rlike(r'^MGI:\\d+$'), cleaned_id)


def map_model_mouse_phenotypes_to_human(model_mouse_phenotypes: DataFrame, phenotype_mapping: DataFrame) -> DataFrame:
    """Map mouse model phenotypes into human terms."""
    return model_mouse_phenotypes.join(phenotype_mapping, on='mp_id', how='inner').select('model_id', 'hp_id')


def aggregate_mouse_phenotypes(model_mouse_phenotypes: DataFrame, mp_terms: DataFrame) -> DataFrame:
    """Aggregate all mouse phenotypes for each model."""
    return (
        model_mouse_phenotypes.join(mp_terms, on='mp_id', how='inner')
        .groupby('model_id')
        .agg(
            f.collect_set(f.struct(f.col('mp_id').alias('id'), f.col('mp_term').alias('label'))).alias(
                'diseaseModelAssociatedModelPhenotypes'
            )
        )
        .select('model_id', 'diseaseModelAssociatedModelPhenotypes')
    )


def find_matched_human_phenotypes(
    disease_model_summary: DataFrame,
    disease_human_phenotypes: DataFrame,
    model_human_phenotypes: DataFrame,
    hp_terms: DataFrame,
) -> DataFrame:
    """Find human phenotypes that are present in both disease and model (after MP->HP mapping)."""
    return (
        disease_model_summary.select('model_id', 'disease_id')
        .join(disease_human_phenotypes, on='disease_id', how='inner')
        .join(model_human_phenotypes, on=['model_id', 'hp_id'], how='inner')
        .join(hp_terms, on='hp_id', how='inner')
        .groupby('model_id', 'disease_id')
        .agg(
            f.collect_set(f.struct(f.col('hp_id').alias('id'), f.col('hp_term').alias('label'))).alias(
                'diseaseModelAssociatedHumanPhenotypes'
            )
        )
        .select('model_id', 'disease_id', 'diseaseModelAssociatedHumanPhenotypes')
    )

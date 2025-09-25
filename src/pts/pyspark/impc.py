"""Evidence parser for the animal model sources from IMPC."""

import pronto
import pyspark.sql.functions as f
from loguru import logger
from pyspark.sql import Column, DataFrame, Window

from pts.pyspark.common.session import Session
from pts.utils.ontology import add_efo_mapping


def impc(
    source: dict[str, str],
    destination: str,
    properties: dict[str, str],
) -> tuple[DataFrame, DataFrame]:
    spark = Session(app_name='impc', properties=properties)
    score_cutoff = float(properties['score_cutoff'])
    efo_version = properties['efo_version']
    mapping_cores = int(properties.get('ontology_cores', 1))

    # Load all required datasets
    datasets = _load_impc_datasets(spark, source)
    datasets['disease_model_summary_transformed'] = _format_disease_model_associations(
        datasets['disease_model_summary']
    )
    datasets['model_mouse_phenotypes_transformed'] = (
        datasets['model_mouse_phenotypes']
        .withColumn('mp_id', f.explode(f.expr(r"regexp_extract_all(model_phenotypes, '(MP:\\d+)', 1)")))
        .select('model_id', 'mp_id')
        # E. g. 'MGI:3800884', 'MP:0001304'.
        .persist()
    )
    datasets['disease_human_phenotypes_transformed'] = (
        datasets['disease_human_phenotypes']
        .withColumn('hp_id', f.explode(f.expr(r"regexp_extract_all(disease_phenotypes, '(HP:\\d+)', 1)")))
        .select('disease_id', 'hp_id')
        # E.g. 'OMIM:609258', 'HP:0000545 Myopia'.
        .distinct()
    )

    # Process ontology terms and classifications
    mp_terms, hp_terms = _process_ontology_terms(datasets['ontology'])
    mp_class = _create_mp_classification(datasets['mp_ontology'], spark)

    # Build gene mapping from mouse to human
    gene_mapping = _build_gene_mapping(
        datasets['mgi_gene_id_to_ensembl_mouse_gene_id'],
        datasets['mouse_to_human_gene'],
        datasets['hgnc_gene_id_to_ensembl_human_gene_id'],
    )

    # Process literature references
    literature = _process_literature_references(
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
    mapped_evidence_df = add_efo_mapping(
        evidence_strings=evidence, spark_instance=spark.spark, efo_version=efo_version, cores=mapping_cores
    )
    final_evidence = _finalise_evidence_strings(mapped_evidence_df)
    logger.info('generate mouse phenotypes dataset')
    mouse_phenotypes = generate_mouse_phenotypes_dataset(
        datasets['disease_model_summary_transformed'],
        gene_mapping,
        datasets['model_mouse_phenotypes_transformed'],
        mp_terms,
        literature,
        mp_class,
    )

    logger.info('write impc datasets')
    final_evidence.write.mode('overwrite').parquet(destination['evidence'])
    mouse_phenotypes.write.mode('overwrite').parquet(destination['mouse_phenotypes'])
    return final_evidence, mouse_phenotypes


def _format_disease_model_associations(disease_model_summary: DataFrame) -> DataFrame:
    return disease_model_summary.selectExpr(
        'model_id',
        'model_genetic_background as biologicalModelGeneticBackground',
        'model_description as biologicalModelAllelicComposition',
        'disease_id',
        'disease_term',
        'disease_model_avg_norm',
        'marker_id as targetInModelMgiId',
        # In Phenodigm, the scores report the association between diseases and animal models, not genes. The
        # phenotype similarity is computed using an algorithm called OWLSim which expresses the similarity in terms
        # of the Jaccard Index (simJ) or Information Content (IC). Therefore, to compute the score you can take the
        # maximum score of both analyses (disease_model_max_norm) or a combination of them both
        # (disease_model_avg_norm). In the Results and discussion section of the Phenodigm paper, the methods are
        # compared to a number of gold standards. It is concluded that the geometric mean of both analyses is the
        # superior metric and should therefore be used as the score.
        'disease_model_avg_norm as resourceScore',
    ).distinct()


def _load_impc_datasets(spark: Session, source: dict[str, str]) -> dict[str, DataFrame]:
    """Load all required IMPC datasets."""
    logger.info(f'load data from {source}')
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
            sep='\t',
        ),
        'mgi_gene_id_to_ensembl_mouse_gene_id': spark.load_data(
            source['mouse_gene_mappings'], format='csv', header=True, nullValue='null', sep='\t'
        ),
        'mouse_to_human_gene': spark.load_data(source['solr_gene_gene'], format='csv', header=True),
        'hgnc_gene_id_to_ensembl_human_gene_id': spark.load_data(
            source['hgnc_gene_mappings'], format='csv', header=True, nullValue='null', sep='\t'
        ),
    }


def _process_ontology_terms(ontology: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Process ontology information to enable MP and HP term lookup based on the ID."""
    logger.info('process ontology information to enable MP and HP term lookup based on the ID.')
    return tuple(
        ontology.filter(f.col('ontology') == ontology_name)
        .withColumnRenamed('phenotype_id', f'{ontology_name.lower()}_id')
        .withColumnRenamed('phenotype_term', f'{ontology_name.lower()}_term')
        .select(f'{ontology_name.lower()}_id', f'{ontology_name.lower()}_term')
        for ontology_name in ('MP', 'HP')
    )


def _create_mp_classification(mp_ontology, spark: Session) -> DataFrame:
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
        # E.g. 'MP:0000275', 'MP:0005385', 'cardiovascular system phenotype'
    )


def _build_gene_mapping(
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


def _process_literature_references(
    mgi_pubmed: DataFrame, disease_model_summary: DataFrame, model_mouse_phenotypes: DataFrame
) -> DataFrame:
    """Process literature references for model-gene combinations."""
    mgi_pubmed_exploded = (
        mgi_pubmed.withColumn('literature', f.explode(f.split(f.col('literature'), r'\|')))
        .withColumn('targetInModelMgiId', f.explode(f.split(f.col('targetInModelMgiId'), r'\|')))
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


def _cleanup_model_identifier(model_id_col: Column):
    """Cleanup model identifier.

    1. Strip the trailing modifiers, where present. The original ID, used for table
    joins, may look like 'MGI:6274930#hom#early', where the first part is the allele ID and the second
    specifies the zygotic state. There can be several models for the same allele ID with different phenotypes.
    However, this information is also duplicated in `biologicalModelGeneticBackground` (for example:
    'C57BL/6NCrl,Ubl7<em1(IMPC)Tcp> hom early'), so in this field we strip those modifiers.

    2. We only want to output the model names from the MGI namespace. An example of something we *don't*
    want is 'NOT-RELEASED-025eb4a791'. This will be converted to null.
    """
    cleaned_id = f.split(model_id_col, '#').getItem(0)
    return f.when(cleaned_id.rlike(r'^MGI:\d+$'), cleaned_id)


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
    model_human_phenotypes = _map_model_mouse_phenotypes_to_human(
        model_mouse_phenotypes, mouse_phenotype_to_human_phenotype
    )
    all_mouse_phenotypes = _aggregate_mouse_phenotypes(model_mouse_phenotypes, mp_terms)
    matched_human_phenotypes = _find_matched_human_phenotypes(
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
        .withColumn('biologicalModelId', _cleanup_model_identifier(f.col('model_id')))
    )


def generate_mouse_phenotypes_dataset(
    disease_model_summary: DataFrame,
    gene_mapping: DataFrame,
    model_mouse_phenotypes: DataFrame,
    mp_terms: DataFrame,
    literature: DataFrame,
    mp_class: DataFrame,
):
    """Generate the related mousePhenotypes dataset for the corresponding widget in the target object."""
    mouse_phenotypes = (
        # Extract base model-target associations.
        disease_model_summary.select(
            'model_id', 'biologicalModelAllelicComposition', 'biologicalModelGeneticBackground', 'targetInModelMgiId'
        )
        .distinct()
        # Add gene mapping information.
        .join(gene_mapping, on='targetInModelMgiId', how='inner')
        # Add mouse phenotypes.
        .join(model_mouse_phenotypes, on='model_id', how='inner')
        .join(mp_terms, on='mp_id', how='inner')
        # Add literature references.
        .join(literature, on=['model_id', 'targetInModelMgiId'], how='left')
        # Rename fields.
        .withColumnRenamed('mp_id', 'modelPhenotypeId')
        .withColumnRenamed('mp_term', 'modelPhenotypeLabel')
        # Join phenotype class information.
        .join(mp_class, on='modelPhenotypeId', how='inner')
        # Post-process model ID field.
        .withColumn('biologicalModelId', _cleanup_model_identifier(f.col('model_id')))
    )

    return (
        # Convert the schema from flat to partially nested, grouping related models and phenotype classes.
        mouse_phenotypes.groupby(
            'targetInModel',
            'targetInModelMgiId',
            'targetInModelEnsemblId',
            'targetFromSourceId',
            'modelPhenotypeId',
            'modelPhenotypeLabel',
        ).agg(
            f.collect_set(
                f.struct(
                    f.col('biologicalModelAllelicComposition').alias('allelicComposition'),
                    f.col('biologicalModelGeneticBackground').alias('geneticBackground'),
                    f.col('biologicalModelId').alias('id'),
                    f.col('literature'),
                )
            ).alias('biologicalModels'),
            f.collect_set(
                f.struct(f.col('modelPhenotypeClassId').alias('id'), f.col('modelPhenotypeClassLabel').alias('label'))
            ).alias('modelPhenotypeClasses'),
        )
    )


def _map_model_mouse_phenotypes_to_human(model_mouse_phenotypes: DataFrame, phenotype_mapping: DataFrame) -> DataFrame:
    """Map mouse model phenotypes into human terms."""
    return model_mouse_phenotypes.join(phenotype_mapping, on='mp_id', how='inner').select('model_id', 'hp_id')


def _aggregate_mouse_phenotypes(model_mouse_phenotypes: DataFrame, mp_terms: DataFrame) -> DataFrame:
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


def _find_matched_human_phenotypes(
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

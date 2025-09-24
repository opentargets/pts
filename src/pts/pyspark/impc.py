"""Evidence parser for the animal model sources from IMPC."""

import pronto
import pyspark.sql.functions as f
from loguru import logger
from pyspark.sql import Column, DataFrame, Window

from pts.pyspark.common.session import Session
from pts.utils.ontology import add_efo_mapping

# List of fields on which to enforce uniqueness by only keeping the record with the highest score.
UNIQUE_FIELDS = [
    # Specific to IMPC.
    'diseaseFromSource',  # Original disease name.
    'targetInModel',  # Mouse gene name.
    'biologicalModelAllelicComposition',  # Mouse model property.
    'biologicalModelGeneticBackground',  # Mouse model property.
    # General.
    'diseaseFromSourceMappedId',  # EFO mapped disease ID.
    'targetFromSourceId',  # Ensembl mapped human gene ID.
]


def impc(
    source: dict[str, str],
    destination: str,
    properties: dict[str, str],
) -> DataFrame:
    spark = Session(app_name='impc', properties=properties)
    score_cutoff = float(properties['score_cutoff'])
    efo_version = properties['efo_version']
    mapping_cores = int(properties.get('ontology_cores', 1))

    logger.info(f'load data from {source}')
    mouse_phenotype_to_human_phenotype = spark.load_data(source['solr_ontology_ontology'], format='csv', header=True)
    model_mouse_phenotypes = spark.load_data(source['solr_mouse_model'], format='csv', header=True)
    disease_human_phenotypes = spark.load_data(source['solr_disease'], format='csv', header=True)
    disease_model_summary = spark.load_data(source['solr_disease_model_summary'], format='csv', header=True)
    ontology = spark.load_data(source['solr_ontology'], format='csv', header=True)
    mp = pronto.Ontology(source['mp_ontology'])
    mgi_pubmed = spark.load_data(
        source['mouse_pubmed_refs'],
        format='csv',
        schema='_0 string, _1 string, _2 string, mp_id, literature, targetInModelMgiId',
    )
    mgi_gene_id_to_ensembl_mouse_gene_id = spark.load_data(source['mp_ontology'])
    mouse_gene_to_human_gene = spark.load_data(source['solr_gene_gene'], format='csv', header=True)
    hgnc_gene_id_to_ensembl_human_gene_id = spark.load_data(source['hgnc_gene_mappings'])

    logger.info('process ontology information to enable MP and HP term lookup based on the ID.')
    mp_terms, hp_terms = (
        ontology.filter(f.col('ontology') == ontology_name)
        .withColumnRenamed('phenotype_id', f'{ontology_name.lower()}_id')
        .withColumnRenamed('phenotype_term', f'{ontology_name.lower()}_term')
        .select(f'{ontology_name.lower()}_id', f'{ontology_name.lower()}_term')
        for ontology_name in ('MP', 'HP')
    )
    logger.info('process MP definitions to extract high level classes for each term.')
    high_level_classes = set(mp['MP:0000001'].subclasses(distance=1)) - {mp['MP:0000001']}
    mp_class = [
        [term.id, mp_high_level_class.id, mp_high_level_class.name]
        for mp_high_level_class in high_level_classes
        for term in mp_high_level_class.subclasses()
    ]
    mp_class = spark.createDataFrame(
        data=mp_class,
        schema=['modelPhenotypeId', 'modelPhenotypeClassId', 'modelPhenotypeClassLabel'],
        # E.g. 'MP:0000275', 'MP:0005385', 'cardiovascular system phenotype'
    )

    # Using the three datasets above, we construct the complete gene mapping from MGI gene ID (the only type of
    # identifier used in the source data) to gene name, mouse Ensembl ID and human Ensembl ID. In cases where
    # mappings are not one to one, joins will handle the necessary explosions.
    logger.info('construct gene mapping.')
    gene_mapping = (
        mgi_gene_id_to_ensembl_mouse_gene_id.selectExpr(
            '1. MGI accession id as targetInModelMgiId',
            '3. marker symbol as targetInModel',
            '11. Ensembl gene id as targetInModelEnsemblId',
        )
        .filter(f.col('targetInModelEnsemblId').isNotNull())
        .join(
            mouse_gene_to_human_gene.selectExpr('gene_id as targetInModelMgiId', 'hgnc_gene_id'),
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
        # For both the evidence and mousePhenotypes datasets, entries without human gene mapping are unusable.
        .filter(f.col('targetFromSourceId').isNotNull())
        .select('targetInModelMgiId', 'targetInModel', 'targetInModelEnsemblId', 'targetFromSourceId')
        # E.g. 'MGI:87859', 'Abl1', 'ENSMUSG00000026842', 'ENSG00000121410'.
    )

    # Literature references for a given (model, gene) combination.
    mgi_pubmed_t = (
        mgi_pubmed.withColumn(
            'literature',
            f.explode(f.split(f.col('literature'), r'\|')),
        )
        .withColumn(
            'targetInModelMgiId',
            f.explode(f.split(f.col('targetInModelMgiId'), r'\|')),
        )
        .select('mp_id', 'literature', 'targetInModelMgiId')
        # E.g. 'MP:0000600', '12529408', 'MGI:97874'.
        .distinct()
    )
    literature = (
        disease_model_summary.select('model_id', 'targetInModelMgiId')
        .distinct()
        .join(model_mouse_phenotypes, on='model_id', how='inner')
        .join(
            mgi_pubmed_t,
            on=['targetInModelMgiId', 'mp_id'],
            how='inner',
        )
        .groupby('model_id', 'targetInModelMgiId')
        .agg(f.collect_set(f.col('literature')).alias('literature'))
        .select('model_id', 'targetInModelMgiId', 'literature')
    )

    logger.info('generate impc evidence strings')
    evidence = generate_impc_evidence_strings(
        model_mouse_phenotypes,
        mouse_phenotype_to_human_phenotype,
        mp_terms,
        disease_model_summary,
        disease_human_phenotypes,
        hp_terms,
        gene_mapping,
        literature,
        score_cutoff,
    )
    mapped_evidence_df = add_efo_mapping(
        evidence_strings=evidence, spark_instance=spark.spark, efo_version=efo_version, cores=mapping_cores
    )

    # In case of multiple records with the same unique fields, keep only the one record with the highest score. This
    # is done to avoid duplicates where multiple source ontology records map to the same EFO record with slightly
    # different scores.
    w = Window.partitionBy([f.col(c) for c in UNIQUE_FIELDS]).orderBy(f.col('resourceScore').desc())
    final_evidence = (
        mapped_evidence_df.withColumn('row', f.row_number().over(w))
        .filter(f.col('row') == 1)
        .drop('row')
        .select(
            '*',
            f.lit('impc').alias('datasourceId'),
            f.lit('animal_model').alias('datatypeId'),
        )
    )
    logger.info('write impc evidence strings')
    final_evidence.write.mode('overwrite').parquet(destination)
    return final_evidence


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
    return (
        dataset.withColumn('biologicalModelId', f.split(f.col('model_id'), '#').getItem(0))
        .drop('model_id')
        .withColumn(
            'biologicalModelId', f.when(f.col('biologicalModelId').rlike(r'^MGI:\d+$'), f.col('biologicalModelId'))
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
):
    """Generate the evidence by renaming, transforming and joining the columns."""
    # Map mouse model phenotypes into human terms.
    model_human_phenotypes = model_mouse_phenotypes.join(
        mouse_phenotype_to_human_phenotype, on='mp_id', how='inner'
    ).select('model_id', 'hp_id')

    # We are reporting all mouse phenotypes for a model, regardless of whether they can be mapped into any human
    # disease.
    all_mouse_phenotypes = (
        model_mouse_phenotypes.join(mp_terms, on='mp_id', how='inner')
        .groupby('model_id')
        .agg(
            f.collect_set(f.struct(f.col('mp_id').alias('id'), f.col('mp_term').alias('label'))).alias(
                'diseaseModelAssociatedModelPhenotypes'
            )
        )
        .select('model_id', 'diseaseModelAssociatedModelPhenotypes')
    )
    # For human phenotypes, we only want to include the ones which are present in the disease *and* also can be
    # traced back to the model phenotypes through the MP → HP mapping relationship.
    matched_human_phenotypes = (
        # We start with all possible pairs of model-disease associations.
        disease_model_summary.select('model_id', 'disease_id')
        # Add all disease phenotypes. Now we have: model_id, disease_id, hp_id (from disease).
        .join(disease_human_phenotypes, on='disease_id', how='inner')
        # Only keep the phenotypes which also appear in the mouse model (after mapping).
        .join(model_human_phenotypes, on=['model_id', 'hp_id'], how='inner')
        # Add ontology terms in addition to IDs. Now we have: model_id, disease_id, hp_id, hp_term.
        .join(hp_terms, on='hp_id', how='inner')
        .groupby('model_id', 'disease_id')
        .agg(
            f.collect_set(f.struct(f.col('hp_id').alias('id'), f.col('hp_term').alias('label'))).alias(
                'diseaseModelAssociatedHumanPhenotypes'
            )
        )
        .select('model_id', 'disease_id', 'diseaseModelAssociatedHumanPhenotypes')
    )

    evidence = (
        # This table contains all unique (model_id, disease_id) associations which form the base of the evidence
        # strings.
        disease_model_summary
        .filter(~(f.col('resourceScore') < score_cutoff))
        # Add mouse gene mapping information. The mappings are not necessarily one to one. When this happens, join
        # will handle the necessary explosions, and a single row from the original table will generate multiple
        # evidence strings. This adds the fields 'targetFromSourceId', 'targetInModelEnsemblId', and
        # 'targetFromSourceId'.
        .join(gene_mapping, on='targetInModelMgiId', how='inner')
        # Add all mouse phenotypes of the model → `diseaseModelAssociatedModelPhenotypes`.
        .join(all_mouse_phenotypes, on='model_id', how='left')
        # Add the matched model/disease human phenotypes → `diseaseModelAssociatedHumanPhenotypes`.
        .join(matched_human_phenotypes, on=['model_id', 'disease_id'], how='left')
        # Add literature references → 'literature'.
        .join(literature, on=['model_id', 'targetInModelMgiId'], how='left')
        .withColumnRenamed('disease_id', 'diseaseFromSourceId')
        .withColumnRenamed('disease_term', 'diseaseFromSource')
    )
    return _cleanup_model_identifier(evidence)

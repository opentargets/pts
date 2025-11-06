"""Target facets computation for Open Targets search functionality.

This module computes various facets (filters) for targets based on different attributes
such as tractability, gene ontology, pathways, subcellular locations, and more.

Each facet function takes target DataFrames and produces a standardized facet output
with the schema: (label, category, entityIds, datasourceId).

The facets are used in the Open Targets Platform search interface to allow users
to filter and explore targets based on different characteristics.
"""

from __future__ import annotations

from loguru import logger
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from pts.pyspark.common.session import Session
from pts.pyspark.facets.helpers import compute_simple_facet, get_relevant_dataset


class FacetSearchCategories:
    """Configuration class for facet category names.

    This class holds the standardised category names used across different facets.
    These categories correspond to the filter options displayed in the UI.
    """

    def __init__(self, config: dict[str, str] | None = None):
        """Initialise facet categories with optional custom values.

        Args:
            config: Optional dictionary of category names. If not provided, uses defaults.
        """
        if config is None:
            config = {}

        # Tractability modalities
        self.SM = config.get('SM', 'Small Molecule')
        self.AB = config.get('AB', 'Antibody')
        self.PR = config.get('PR', 'Protac')
        self.OC = config.get('OC', 'Other Clinical')

        # Gene Ontology aspects
        self.goF = config.get('goF', 'GO Molecular Function')
        self.goP = config.get('goP', 'GO Biological Process')
        self.goC = config.get('goC', 'GO Cellular Component')

        # Simple facets
        self.target_id = config.get('targetId', 'Target ID')
        self.approved_symbol = config.get('approvedSymbol', 'Target Symbol')
        self.approved_name = config.get('approvedName', 'Target Name')
        self.subcellular_location = config.get('subcellularLocation', 'Subcellular Location')
        self.target_class = config.get('targetClass', 'Target Class')
        self.pathways = config.get('pathways', 'Pathways')


def compute_tractability_facets(
    targets_df: DataFrame,
    category_values: FacetSearchCategories,
    spark: SparkSession,
) -> DataFrame:
    """Compute tractability facets for the given targets DataFrame.

    Process:
    1. Extract target ID and tractability array from targets
    2. Explode tractability array to individual rows
    3. Filter for tractability values that are True
    4. Group by tractability modality and label, collecting target IDs
    5. Map modality codes to readable category names (SM -> Small Molecule, etc.)

    Args:
        targets_df: DataFrame containing target data with 'id' and 'tractability' columns
        category_values: FacetSearchCategories instance with category name mappings
        spark: SparkSession instance

    Returns:
        DataFrame with facet schema containing tractability facets
    """
    logger.info('Computing tractability facets')

    # Create mapping for tractability modality codes to full names
    tractability_modality_mappings = F.create_map([
        F.lit('SM'), F.lit(category_values.SM),
        F.lit('AB'), F.lit(category_values.AB),
        F.lit('PR'), F.lit(category_values.PR),
        F.lit('OC'), F.lit(category_values.OC),
    ])

    # Extract relevant data with tractability array
    tractability_with_id = get_relevant_dataset(
        targets_df, 'id', 'ensemblGeneId', 'tractability'
    )

    # Process tractability data
    # 1. Explode the tractability array
    # 2. Extract modality, id (label), and value from each tractability entry
    # 3. Filter for true values only
    # 4. Group and collect target IDs
    return (
        tractability_with_id
        .select(
            F.col('ensemblGeneId'),
            F.explode('tractability').alias('t')
        )
        .select(
            F.col('ensemblGeneId'),
            F.col('t.modality').alias('category'),
            F.col('t.id').alias('label'),
            F.col('t.value').alias('value')
        )
        .where(F.col('value'))
        .groupBy('category', 'label')
        .agg(F.collect_set('ensemblGeneId').alias('entityIds'))
        .drop('value')
        .withColumn(
            'category',
            F.when(
                tractability_modality_mappings[F.col('category')].isNotNull(),
                tractability_modality_mappings[F.col('category')]
            ).otherwise(F.col('category'))
        )
        .withColumn('datasourceId', F.lit(None).cast('string'))
        .distinct()
    )


def compute_target_id_facets(
    targets_df: DataFrame,
    category_values: FacetSearchCategories,
    spark: SparkSession,
) -> DataFrame:
    """Compute target ID facets for the given targets DataFrame.

    Args:
        targets_df: DataFrame containing target data with 'id' column
        category_values: FacetSearchCategories instance
        spark: SparkSession instance

    Returns:
        DataFrame with facet schema containing target ID facets
    """
    logger.info('Computing target ID facets')
    return compute_simple_facet(targets_df, 'id', category_values.target_id, 'id', spark)


def compute_approved_symbol_facets(
    targets_df: DataFrame,
    category_values: FacetSearchCategories,
    spark: SparkSession,
) -> DataFrame:
    """Compute approved gene symbol facets for the given targets DataFrame.

    Args:
        targets_df: DataFrame containing target data with 'approvedSymbol' column
        category_values: FacetSearchCategories instance
        spark: SparkSession instance

    Returns:
        DataFrame with facet schema containing approved symbol facets
    """
    logger.info('Computing approved symbol facets')
    return compute_simple_facet(
        targets_df, 'approvedSymbol', category_values.approved_symbol, 'id', spark
    )


def compute_approved_name_facets(
    targets_df: DataFrame,
    category_values: FacetSearchCategories,
    spark: SparkSession,
) -> DataFrame:
    """Compute approved gene name facets for the given targets DataFrame.

    Args:
        targets_df: DataFrame containing target data with 'approvedName' column
        category_values: FacetSearchCategories instance
        spark: SparkSession instance

    Returns:
        DataFrame with facet schema containing approved name facets
    """
    logger.info('Computing approved name facets')
    return compute_simple_facet(
        targets_df, 'approvedName', category_values.approved_name, 'id', spark
    )


def compute_subcellular_locations_facets(
    targets_df: DataFrame,
    category_values: FacetSearchCategories,
    spark: SparkSession,
) -> DataFrame:
    """Compute subcellular location facets for the given targets DataFrame.

    Process:
    1. Extract target ID and subcellularLocations array
    2. Explode array and extract location and source (termSl)
    3. Group by location, collecting target IDs
    4. Keep datasourceId to track the source of the annotation

    Args:
        targets_df: DataFrame with 'id' and 'subcellularLocations' columns
        category_values: FacetSearchCategories instance
        spark: SparkSession instance

    Returns:
        DataFrame with facet schema containing subcellular location facets
    """
    logger.info('Computing subcellular location facets')

    subcellular_location_with_id = get_relevant_dataset(
        targets_df, 'id', 'ensemblGeneId', 'subcellularLocations'
    )

    return (
        subcellular_location_with_id
        .select(
            F.col('ensemblGeneId'),
            F.explode('subcellularLocations').alias('s')
        )
        .select(
            F.col('ensemblGeneId').alias('id'),
            F.col('s.location').alias('label'),
            F.lit(category_values.subcellular_location).alias('category'),
            F.col('s.termSl').alias('datasourceId')
        )
        .groupBy('label', 'category', 'datasourceId')
        .agg(F.collect_set('id').alias('entityIds'))
        .distinct()
    )


def compute_target_class_facets(
    targets_df: DataFrame,
    category_values: FacetSearchCategories,
    spark: SparkSession,
) -> DataFrame:
    """Compute target class facets for the given targets DataFrame.

    Process:
    1. Extract target ID and targetClass array
    2. Explode array and extract class labels
    3. Group by label, collecting target IDs

    Args:
        targets_df: DataFrame with 'id' and 'targetClass' columns
        category_values: FacetSearchCategories instance
        spark: SparkSession instance

    Returns:
        DataFrame with facet schema containing target class facets
    """
    logger.info('Computing target class facets')

    target_class_with_id = get_relevant_dataset(
        targets_df, 'id', 'ensemblGeneId', 'targetClass'
    )

    return (
        target_class_with_id
        .select(
            F.col('ensemblGeneId'),
            F.explode('targetClass').alias('t')
        )
        .select(
            F.col('ensemblGeneId'),
            F.col('t.label').alias('label'),
            F.lit(category_values.target_class).alias('category')
        )
        .groupBy('label', 'category')
        .agg(F.collect_set('ensemblGeneId').alias('entityIds'))
        .withColumn('datasourceId', F.lit(None).cast('string'))
        .distinct()
    )


def compute_pathways_facets(
    targets_df: DataFrame,
    category_values: FacetSearchCategories,
    spark: SparkSession,
) -> DataFrame:
    """Compute pathway facets for the given targets DataFrame.

    Process:
    1. Extract target ID and pathways array (from Reactome)
    2. Explode array and extract pathway name and pathway ID
    3. Group by pathway, collecting target IDs
    4. Keep pathwayId as datasourceId for reference

    Args:
        targets_df: DataFrame with 'id' and 'pathways' columns
        category_values: FacetSearchCategories instance
        spark: SparkSession instance

    Returns:
        DataFrame with facet schema containing pathway facets
    """
    logger.info('Computing pathway facets')

    pathways_with_id = get_relevant_dataset(
        targets_df, 'id', 'id', 'pathways'
    )

    return (
        pathways_with_id
        .select(
            F.col('id'),
            F.explode('pathways').alias('p')
        )
        .select(
            F.col('id').alias('ensemblGeneId'),
            F.col('p.pathway').alias('label'),
            F.lit(category_values.pathways).alias('category'),
            F.col('p.pathwayId').alias('datasourceId')
        )
        .groupBy('label', 'category', 'datasourceId')
        .agg(F.collect_set('ensemblGeneId').alias('entityIds'))
        .distinct()
    )


def compute_go_facets(
    targets_df: DataFrame,
    go_df: DataFrame,
    category_values: FacetSearchCategories,
    spark: SparkSession,
) -> DataFrame:
    """Compute Gene Ontology (GO) facets for the given targets DataFrame.

    Gene Ontology provides standardized annotations about gene function across
    three aspects:
    - Molecular Function (F): What the gene product does at molecular level
    - Biological Process (P): Larger biological programs the gene participates in
    - Cellular Component (C): Where the gene product is located

    Process:
    1. Extract target ID and GO array from targets
    2. Explode GO array to get individual GO term IDs and aspects
    3. Join with GO reference data to get term names
    4. Map GO aspects (F, P, C) to readable category names
    5. Group by GO term, collecting target IDs

    Args:
        targets_df: DataFrame with 'id' and 'go' columns
        go_df: Reference DataFrame with GO term information (id, name, aspect)
        category_values: FacetSearchCategories instance
        spark: SparkSession instance

    Returns:
        DataFrame with facet schema containing GO facets
    """
    logger.info('Computing GO facets')

    # Create mapping for GO aspect codes to full names
    go_aspect_mappings = F.create_map([
        F.lit('F'), F.lit(category_values.goF),
        F.lit('P'), F.lit(category_values.goP),
        F.lit('C'), F.lit(category_values.goC),
    ])

    # Extract relevant GO data from targets
    go_with_id = get_relevant_dataset(
        targets_df, 'id', 'ensemblId', 'go'
    )

    # Process GO data
    # 1. Explode GO array to individual terms
    # 2. Join with GO reference to get term names
    # 3. Map aspect codes to category names
    # 4. Group and collect target IDs
    return (
        go_with_id
        .select(
            F.col('ensemblId'),
            F.explode('go').alias('g')
        )
        .select(
            F.col('ensemblId').alias('ensemblGeneId'),
            F.col('g.id').alias('id'),
            F.col('g.aspect').alias('category')
        )
        .join(go_df, on='id', how='left')
        # GO dataframe already has 'label' column (not 'name')
        .withColumn('datasourceId', F.col('id'))
        .groupBy('label', 'category', 'datasourceId')
        .agg(F.collect_set('ensemblGeneId').alias('entityIds'))
        .withColumn(
            'category',
            F.when(
                go_aspect_mappings[F.col('category')].isNotNull(),
                go_aspect_mappings[F.col('category')]
            ).otherwise(F.col('category'))
        )
        .distinct()
    )


def compute_all_target_facets(
    targets_df: DataFrame,
    go_df: DataFrame,
    category_values: FacetSearchCategories | None = None,
    spark: SparkSession | None = None,
) -> DataFrame:
    """Compute all target facets and union them into a single DataFrame.

    This convenience function computes all available facet types and combines
    them into a single output DataFrame.

    Args:
        targets_df: DataFrame containing target data
        go_df: DataFrame containing GO reference data
        category_values: Optional FacetSearchCategories instance. Uses defaults if None.
        spark: Optional SparkSession. Extracted from targets_df if None.

    Returns:
        DataFrame containing all computed facets
    """
    if spark is None:
        spark = targets_df.sparkSession

    if category_values is None:
        category_values = FacetSearchCategories()

    logger.info('Computing all target facets')

    # Compute all facet types
    facets_list = [
        compute_tractability_facets(targets_df, category_values, spark),
        compute_target_id_facets(targets_df, category_values, spark),
        compute_approved_symbol_facets(targets_df, category_values, spark),
        compute_approved_name_facets(targets_df, category_values, spark),
        compute_subcellular_locations_facets(targets_df, category_values, spark),
        compute_target_class_facets(targets_df, category_values, spark),
        compute_pathways_facets(targets_df, category_values, spark),
        compute_go_facets(targets_df, go_df, category_values, spark),
    ]

    # Union all facets into single DataFrame
    all_facets = facets_list[0]
    for facets_df in facets_list[1:]:
        all_facets = all_facets.union(facets_df)

    return all_facets


def target_facets(
    source: dict[str, str],
    destination: dict[str, str],
    properties: dict[str, str] | None = None,
    category_config: dict[str, str] | None = None,
) -> None:
    """Main entry point for computing target facets.

    This function reads target and GO data, computes all facet types,
    and writes the output to the specified destination. It supports both
    local filesystem and Google Cloud Storage (gs://) paths.

    Configuration via source dict:
        - 'targets': Path to targets parquet data
        - 'go': Path to GO reference parquet data

    Configuration via destination dict:
        - 'targets': Path to write target facets output

    Args:
        source: Dictionary with 'targets' and 'go' keys pointing to input paths
        destination: Dictionary with 'targets' key pointing to output path
        properties: Optional Spark configuration properties
        category_config: Optional custom category name mappings

    Example:
        >>> target_facets(
        ...     source={
        ...         'targets': 'gs://bucket/targets.parquet',
        ...         'go': 'gs://bucket/go.parquet'
        ...     },
        ...     destination={'targets': 'gs://bucket/output/facets.parquet'},
        ...     properties={'spark.executor.memory': '8g'}
        ... )
    """
    # Initialize Spark session with GCS support
    session = Session(app_name='target_facets', properties=properties)
    spark = session.spark

    try:
        logger.info('Executing Target Facets step (PySpark)')
        logger.info(f'Source paths: {source}')
        logger.info(f'Destination path: {destination}')

        # Load input data
        logger.info(f"Loading targets from: {source['targets']}")
        targets_df = spark.read.parquet(source['targets'])

        logger.info(f"Loading GO data from: {source['go']}")
        go_df = spark.read.parquet(source['go'])

        # Initialize category configuration
        category_values = FacetSearchCategories(category_config)

        # Compute all facets
        all_facets = compute_all_target_facets(targets_df, go_df, category_values, spark)

        # Write output
        output_path = destination['targets']
        logger.info(f'Writing target facets to: {output_path}')
        all_facets.write.mode('overwrite').parquet(output_path)

        logger.info('Target Facets step complete')

    finally:
        session.stop()

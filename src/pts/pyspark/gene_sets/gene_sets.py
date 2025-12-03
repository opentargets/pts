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
from pyspark.sql import DataFrame
from pyspark.sql import functions as f

from pts.pyspark.common.session import Session
from pts.pyspark.gene_sets.helpers import compute_simple_facet, get_relevant_dataset
from pts.pyspark.gene_sets.propagation import propagate_entity_ids_pyspark_efficiently


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
        self.SM = config.get('SM', 'Tractability Small Molecule')
        self.AB = config.get('AB', 'Tractability Antibody')
        self.PR = config.get('PR', 'Tractability PROTAC')
        self.OC = config.get('OC', 'Tractability Other Modalities')

        # Gene Ontology aspects
        self.goF = config.get('goF', 'GO:MF')
        self.goP = config.get('goP', 'GO:BP')
        self.goC = config.get('goC', 'GO:CC')

        # Simple facets
        self.target_id = config.get('targetId', 'Target ID')
        self.approved_symbol = config.get('approvedSymbol', 'Approved Symbol')
        self.approved_name = config.get('approvedName', 'Approved Name')
        self.subcellular_location = config.get('subcellularLocation', 'Subcellular Location')
        self.target_class = config.get('targetClass', 'ChEMBL Target Class')
        self.pathways = config.get('pathways', 'Reactome')


def compute_tractability_facets(
    targets_df: DataFrame,
    category_values: FacetSearchCategories,
) -> DataFrame:
    """Compute tractability facets for the given targets DataFrame.

    Process:
    1. Extract target ID and tractability array from targets
    2. Explode tractability array to individual rows
    3. Filter for tractability values that are True
    4. Group by tractability modality and label, collecting target IDs
    5. Map modality codes to category names (SM/AB/PR/OC -> Tractability variants)

    Args:
        targets_df: DataFrame containing target data with 'id' and 'tractability' columns
        category_values: FacetSearchCategories instance with category name mappings

    Returns:
        DataFrame with facet schema containing tractability facets
    """
    logger.info('Computing tractability facets')

    # Create mapping for tractability modality codes to full names
    tractability_modality_mappings = f.create_map([
        f.lit('SM'),
        f.lit(category_values.SM),
        f.lit('AB'),
        f.lit(category_values.AB),
        f.lit('PR'),
        f.lit(category_values.PR),
        f.lit('OC'),
        f.lit(category_values.OC),
    ])

    # Extract relevant data with tractability array
    tractability_with_id = get_relevant_dataset(targets_df, 'id', 'ensemblGeneId', 'tractability')

    # Process tractability data
    # 1. Explode the tractability array
    # 2. Extract modality, id (label), and value from each tractability entry
    # 3. Filter for true values only
    # 4. Group and collect target IDs
    return (
        tractability_with_id.select(f.col('ensemblGeneId'), f.explode('tractability').alias('t'))
        .select(
            f.col('ensemblGeneId'),
            f.col('t.modality').alias('category'),
            f.col('t.id').alias('label'),
            f.col('t.value').alias('value'),
        )
        .where(f.col('value'))
        .groupBy('category', 'label')
        .agg(f.collect_set('ensemblGeneId').alias('entityIds'))
        .drop('value')
        .withColumn(
            'category',
            f.when(
                tractability_modality_mappings[f.col('category')].isNotNull(),
                tractability_modality_mappings[f.col('category')],
            ).otherwise(f.col('category')),
        )
        .withColumn('datasourceId', f.lit(None).cast('string'))
        .withColumn('parentId', f.array().cast('array<string>'))
        .select('label', 'category', 'entityIds', 'datasourceId', 'parentId')
        .distinct()
    )


def compute_target_id_facets(
    targets_df: DataFrame,
    category_values: FacetSearchCategories,
) -> DataFrame:
    """Compute target ID facets for the given targets DataFrame.

    Args:
        targets_df: DataFrame containing target data with 'id' column
        category_values: FacetSearchCategories instance

    Returns:
        DataFrame with facet schema containing target ID facets
    """
    logger.info('Computing target ID facets')
    return compute_simple_facet(targets_df, 'id', category_values.target_id, 'id')


def compute_approved_symbol_facets(
    targets_df: DataFrame,
    category_values: FacetSearchCategories,
) -> DataFrame:
    """Compute approved gene symbol facets for the given targets DataFrame.

    Args:
        targets_df: DataFrame containing target data with 'approvedSymbol' column
        category_values: FacetSearchCategories instance

    Returns:
        DataFrame with facet schema containing approved symbol facets
    """
    logger.info('Computing approved symbol facets')
    return compute_simple_facet(targets_df, 'approvedSymbol', category_values.approved_symbol, 'id')


def compute_approved_name_facets(
    targets_df: DataFrame,
    category_values: FacetSearchCategories,
) -> DataFrame:
    """Compute approved gene name facets for the given targets DataFrame.

    Args:
        targets_df: DataFrame containing target data with 'approvedName' column
        category_values: FacetSearchCategories instance

    Returns:
        DataFrame with facet schema containing approved name facets
    """
    logger.info('Computing approved name facets')
    return compute_simple_facet(targets_df, 'approvedName', category_values.approved_name, 'id')


def compute_subcellular_locations_facets(
    targets_df: DataFrame,
    category_values: FacetSearchCategories,
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

    Returns:
        DataFrame with facet schema containing subcellular location facets
    """
    logger.info('Computing subcellular location facets')

    subcellular_location_with_id = get_relevant_dataset(targets_df, 'id', 'ensemblGeneId', 'subcellularLocations')

    return (
        subcellular_location_with_id.select(f.col('ensemblGeneId'), f.explode('subcellularLocations').alias('s'))
        .select(
            f.col('ensemblGeneId').alias('id'),
            f.col('s.location').alias('label'),
            f.lit(category_values.subcellular_location).alias('category'),
            f.col('s.termSl').alias('datasourceId'),
        )
        .groupBy('label', 'category', 'datasourceId')
        .agg(f.collect_set('id').alias('entityIds'))
        .withColumn('parentId', f.array().cast('array<string>'))
        .select('label', 'category', 'entityIds', 'datasourceId', 'parentId')
        .distinct()
    )


def compute_target_class_facets(
    targets_df: DataFrame,
    category_values: FacetSearchCategories,
) -> DataFrame:
    """Compute target class facets for the given targets DataFrame.

    Process:
    1. Extract target ID and targetClass array
    2. Explode array and extract class labels
    3. Infer parent-child relationships from hierarchy levels (l1, l2, l3, etc.)
    4. Map children to their immediate parents
    5. Group by label, collecting target IDs and adding parentId

    Args:
        targets_df: DataFrame with 'id' and 'targetClass' columns
        category_values: FacetSearchCategories instance

    Returns:
        DataFrame with facet schema containing target class facets
    """
    logger.info('Computing target class facets')

    target_class_with_id = get_relevant_dataset(targets_df, 'id', 'ensemblGeneId', 'targetClass')

    # Step 1: Explode targetClass arrays into flat structure
    # Extract id, label, and level from each targetClass entry
    targets_flat = target_class_with_id.select(f.col('ensemblGeneId'), f.explode('targetClass').alias('t')).select(
        f.col('ensemblGeneId'),
        f.col('t.id').alias('id'),
        f.col('t.label').alias('label'),
        f.col('t.level').alias('level'),
    )

    # Step 2: Extract numeric level (l1 → 1, l2 → 2, etc.)
    # Use regex to extract the number from level string
    targets_flat_extr = targets_flat.withColumn('level_num', f.regexp_extract('level', r'l(\d+)', 1).cast('int'))

    # Step 3: Self-join to find parent→child relationships
    # Join where child level = parent level + 1 (immediate parent only)
    # This creates a mapping: child label → parent label
    targets_rel = (
        targets_flat_extr.alias('p')
        .join(
            targets_flat_extr.alias('c'),
            (f.col('p.id') == f.col('c.id')) & (f.col('c.level_num') == f.col('p.level_num') + 1),
            how='inner',
        )
        .select(f.col('c.label').alias('child_label'), f.col('p.label').alias('parent_label'))
        .distinct()
    )

    # Step 4: Group by child label and collect all parent labels into an array
    # This creates the parentId mapping: child_label → [parent1, parent2, ...]
    parent_mapping = targets_rel.groupBy('child_label').agg(f.collect_set('parent_label').alias('parentId'))

    # Step 5: Compute facets and join with parent mapping
    facets = (
        target_class_with_id.select(f.col('ensemblGeneId'), f.explode('targetClass').alias('t'))
        .select(
            f.col('ensemblGeneId'),
            f.col('t.label').alias('label'),
            f.lit(category_values.target_class).alias('category'),
        )
        .groupBy('label', 'category')
        .agg(f.collect_set('ensemblGeneId').alias('entityIds'))
        .withColumn('datasourceId', f.lit(None).cast('string'))
    )

    # Step 6: Join facets with parent mapping to add parentId
    # Left join: if no parent found, use empty array
    return (
        facets.join(parent_mapping, on=f.col('label') == f.col('child_label'), how='left')
        .withColumn(
            'parentId',
            f.when(f.col('parentId').isNotNull(), f.col('parentId')).otherwise(f.array().cast('array<string>')),
        )
        .select('label', 'category', 'entityIds', 'datasourceId', 'parentId')
        .distinct()
    )


def compute_pathways_facets(
    targets_df: DataFrame,
    reactome_df: DataFrame,
    category_values: FacetSearchCategories,
) -> DataFrame:
    """Compute pathway facets for the given targets DataFrame.

    Process:
    1. Extract target ID and pathways array (from Reactome)
    2. Explode array and extract pathway name and pathway ID
    3. Join with Reactome reference data to get parent pathways
    4. Group by pathway, collecting target IDs
    5. Keep pathwayId as datasourceId for reference

    Args:
        targets_df: DataFrame with 'id' and 'pathways' columns
        reactome_df: Reference DataFrame with Reactome pathway information (id, label, parents)
        category_values: FacetSearchCategories instance

    Returns:
        DataFrame with facet schema containing pathway facets
    """
    logger.info('Computing pathway facets')

    pathways_with_id = get_relevant_dataset(targets_df, 'id', 'id', 'pathways')

    # Process pathways and join with Reactome reference to get parents
    # Select only needed columns from reactome_df to avoid column name conflicts
    reactome_parents = reactome_df.select(f.col('id').alias('reactome_id'), f.col('parents').alias('reactome_parents'))

    return (
        pathways_with_id.select(f.col('id'), f.explode('pathways').alias('p'))
        .select(
            f.col('id').alias('ensemblGeneId'),
            f.col('p.pathway').alias('label'),
            f.lit(category_values.pathways).alias('category'),
            f.col('p.pathwayId').alias('datasourceId'),
        )
        .join(reactome_parents, on=f.col('datasourceId') == f.col('reactome_id'), how='left')
        # Extract parents from Reactome reference, use empty array if null
        .withColumn(
            'parentId',
            f.when(f.col('reactome_parents').isNotNull(), f.col('reactome_parents')).otherwise(
                f.array().cast('array<string>')
            ),
        )
        .groupBy('label', 'category', 'datasourceId', 'parentId')
        .agg(f.collect_set('ensemblGeneId').alias('entityIds'))
        .select('label', 'category', 'entityIds', 'datasourceId', 'parentId')
        .distinct()
    )


def compute_go_facets(
    targets_df: DataFrame,
    go_df: DataFrame,
    category_values: FacetSearchCategories,
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

    Returns:
        DataFrame with facet schema containing GO facets
    """
    logger.info('Computing GO facets')

    # Create mapping for GO aspect codes to full names
    go_aspect_mappings = f.create_map([
        f.lit('F'),
        f.lit(category_values.goF),
        f.lit('P'),
        f.lit(category_values.goP),
        f.lit('C'),
        f.lit(category_values.goC),
    ])

    # Extract relevant GO data from targets
    go_with_id = get_relevant_dataset(targets_df, 'id', 'ensemblId', 'go')

    # Process GO data
    # 1. Explode GO array to individual terms
    # 2. Join with GO reference to get term names
    # 3. Map aspect codes to category names
    # 4. Group and collect target IDs
    return (
        go_with_id.select(f.col('ensemblId'), f.explode('go').alias('g'))
        .select(
            f.col('ensemblId').alias('ensemblGeneId'), f.col('g.id').alias('id'), f.col('g.aspect').alias('category')
        )
        .join(go_df, on='id', how='left')
        # GO dataframe already has 'label' column (not 'name')
        .withColumn('datasourceId', f.col('id'))
        # Combine is_a and part_of arrays into parentId
        .withColumn(
            'parentId',
            f.when(
                f.col('is_a').isNotNull() | f.col('part_of').isNotNull(),
                f.array_distinct(
                    f.concat(f.coalesce(f.col('is_a'), f.array()), f.coalesce(f.col('part_of'), f.array()))
                ),
            ).otherwise(f.array().cast('array<string>')),
        )
        .groupBy('label', 'category', 'datasourceId', 'parentId')
        .agg(f.collect_set('ensemblGeneId').alias('entityIds'))
        .withColumn(
            'category',
            f.when(go_aspect_mappings[f.col('category')].isNotNull(), go_aspect_mappings[f.col('category')]).otherwise(
                f.col('category')
            ),
        )
        .select('label', 'category', 'entityIds', 'datasourceId', 'parentId')
        .distinct()
    )


def prepare_dataset_for_propagation(facets_df: DataFrame) -> DataFrame:
    """Prepare dataset for propagation by creating id column and exploding parentId.

    This function prepares a facets DataFrame for propagation by:
    1. Creating an 'id' column based on category:
       - For GO:BP, GO:MF, GO:CC, Reactome: use datasourceId
       - For other categories: use label
    2. Selecting id, parentId, entityIds
    3. Exploding parentId to get parent_id (string)

    Args:
        facets_df: DataFrame with columns (label, category, datasourceId, parentId, entityIds).
            parentId should be an array of strings.

    Returns:
        DataFrame with columns (id, parent_id, entityIds) where:
        - id: str (from datasourceId or label based on category)
        - parent_id: str (exploded from parentId array)
        - entityIds: array<string>
    """
    # Create id column based on category
    prepared = facets_df.withColumn(
        'id',
        f.when(
            f.col('category').isin(['GO:BP', 'GO:MF', 'GO:CC', 'Reactome']),
            f.col('datasourceId'),
        ).otherwise(f.col('label')),
    )

    # Select id, parentId, entityIds
    selected = prepared.select('id', 'parentId', 'entityIds')

    # Explode parentId to get parent_id (string)
    # Filter out rows where parentId is null or empty to avoid issues
    return (
        selected.filter(f.col('parentId').isNotNull() & (f.size(f.col('parentId')) > 0))
        .withColumn('parent_id', f.explode('parentId'))
        .select('id', 'parent_id', 'entityIds')
    )


def propagate_entity_ids_with_dataset_prep(
    facets_df: DataFrame, return_iterations: bool = False
) -> DataFrame | tuple[DataFrame, int]:
    """Propagate entityIds from children to parents with dataset preparation.

    This function combines prepare_dataset_for_propagation and propagate_entity_ids_pyspark_efficiently:
    1. Prepares the facets DataFrame by creating id column and exploding parentId
    2. Propagates entityIds from children to parents iteratively

    Args:
        facets_df: DataFrame with columns (label, category, datasourceId, parentId, entityIds).
            parentId should be an array of strings.
        return_iterations: If True, returns a tuple (result_df, iterations). Defaults to False.

    Returns:
        DataFrame with columns (id, parent_id, entityIds) where entityIds have been
        propagated from children to parents transitively.
        If return_iterations is True, returns (DataFrame, int) where int is the number of iterations.
    """
    # Step 1: Prepare dataset for propagation
    prepared_df = prepare_dataset_for_propagation(facets_df)

    # Step 2: Propagate entityIds
    return propagate_entity_ids_pyspark_efficiently(prepared_df, return_iterations=return_iterations)


def merge_propagated_entity_ids(facets_df: DataFrame, propagated_df: DataFrame) -> DataFrame:
    """Merge propagated entityIds back into the original facets DataFrame.

    This function takes the original facets DataFrame and the propagated result,
    and adds a new column 'entityIdsPropagated' containing the propagated entityIds
    while keeping the original 'entityIds' column unchanged.

    Args:
        facets_df: Original DataFrame with columns (label, category, datasourceId, parentId, entityIds).
        propagated_df: Propagated DataFrame with columns (id, parent_id, entityIds)
            where entityIds have been propagated from children to parents.

    Returns:
        DataFrame with all original columns plus 'entityIdsPropagated' column.
        The original 'entityIds' column is preserved unchanged.
    """
    # Step 1: Get unique entityIds per id
    # The propagated result has one row per (id, parent_id) pair, but entityIds are already
    # aggregated at the node level, so all rows for the same id have the same entityIds.
    # We just need to collapse multiple rows into one per id.
    propagated_aggregated = propagated_df.groupBy('id').agg(f.first('entityIds').alias('entityIdsPropagated'))

    # Step 2: Create id column in original facets using same logic as prepare_dataset_for_propagation
    facets_with_id = facets_df.withColumn(
        'id',
        f.when(
            f.col('category').isin(['GO:BP', 'GO:MF', 'GO:CC', 'Reactome']),
            f.col('datasourceId'),
        ).otherwise(f.col('label')),
    )

    # Step 3: Left join to add entityIdsPropagated, keeping all original columns
    # Use coalesce to handle cases where id doesn't exist in propagated result
    return (
        facets_with_id.join(propagated_aggregated, on='id', how='left')
        .withColumn(
            'entityIdsPropagated',
            f.coalesce(f.col('entityIdsPropagated'), f.array().cast('array<string>')),
        )
        .drop('id')  # Remove temporary id column
    )


def compute_all_target_facets(
    targets_df: DataFrame,
    go_df: DataFrame,
    reactome_df: DataFrame,
    category_values: FacetSearchCategories | None = None,
) -> DataFrame:
    """Compute all target facets and union them into a single DataFrame.

    This convenience function computes all available facet types and combines
    them into a single output DataFrame.

    Args:
        targets_df: DataFrame containing target data
        go_df: DataFrame containing GO reference data
        reactome_df: DataFrame containing Reactome reference data
        category_values: Optional FacetSearchCategories instance. Uses defaults if None.

    Returns:
        DataFrame containing all computed facets
    """
    if category_values is None:
        category_values = FacetSearchCategories()

    logger.info('Computing all target facets')

    # Compute all facet types
    facets_list = [
        compute_tractability_facets(targets_df, category_values),
        compute_target_id_facets(targets_df, category_values),
        compute_approved_symbol_facets(targets_df, category_values),
        compute_approved_name_facets(targets_df, category_values),
        compute_subcellular_locations_facets(targets_df, category_values),
        compute_target_class_facets(targets_df, category_values),
        compute_pathways_facets(targets_df, reactome_df, category_values),
        compute_go_facets(targets_df, go_df, category_values),
    ]

    # Union all facets into single DataFrame
    all_facets = facets_list[0]
    for facets_df in facets_list[1:]:
        all_facets = all_facets.union(facets_df)

    # Transitive propagation: ensure parents contain entityIds from all descendants
    propagated_result = propagate_entity_ids_with_dataset_prep(all_facets)

    # Merge propagated entityIds back into original facets, keeping original entityIds
    return merge_propagated_entity_ids(all_facets, propagated_result)


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
        - 'reactome': Path to Reactome reference parquet data

    Configuration via destination dict:
        - 'targets': Path to write target facets output

    Args:
        source: Dictionary with 'targets', 'go', and 'reactome' keys pointing to input paths
        destination: Dictionary with 'targets' key pointing to output path
        properties: Optional Spark configuration properties
        category_config: Optional custom category name mappings
    """
    # Initialize Spark session with GCS support
    session = Session(app_name='target_facets', properties=properties)
    spark = session.spark

    try:
        logger.info('Executing Target Facets step (PySpark)')
        logger.info(f'Source paths: {source}')
        logger.info(f'Destination path: {destination}')

        # Load input data
        logger.info(f'Loading targets from: {source["targets"]}')
        targets_df = spark.read.parquet(source['targets'])

        logger.info(f'Loading GO data from: {source["go_processed"]}')
        go_df = spark.read.parquet(source['go_processed'])
        go_df = go_df.filter(f.col('isObsolete').isNull() | (~f.col('isObsolete')))

        logger.info(f'Loading Reactome data from: {source["reactome"]}')
        reactome_df = spark.read.parquet(source['reactome'])

        # Initialize category configuration
        category_values = FacetSearchCategories(category_config)

        # Compute all facets
        all_facets = compute_all_target_facets(targets_df, go_df, reactome_df, category_values)

        # Write output
        output_path = destination['gene_sets']
        logger.info(f'Writing gene_sets (target facets) to: {output_path}')
        all_facets.write.mode('overwrite').parquet(output_path)

        logger.info('Target Facets step complete')

    finally:
        session.stop()

from __future__ import annotations

from functools import reduce
from typing import Any

from loguru import logger
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as f

from pts.pyspark.common.session import Session


class AggregateExpression:
    """This class is used to take the average of the pseudobulked expression data.

    The output of this class is a dataframe with the average expression of each
     gene (rows) per annotation (columns).
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.df: DataFrame | None = None

    def load_data(self, directory: str, local: bool = False):
        """This function loads the data from the directory and returns a list of DataFrames."""
        # Use spark to read the parquet files in the directory
        if local and not directory.startswith('file://') and not directory.startswith('gs://'):
            directory = f'file://{directory}'

        logger.info(f'Loading data from {directory}')
        df = self.spark.read.parquet(directory)

        self.df = df

    def override_biosample_id(self, biosample_mapping_reference_path: str,
                                key_col: str,
                                source_col: str,
                                id_col: str):
        """Assign a new biosample Id to each row in self.df.

        Use  exact matching to match `source_col` to entries in a wide TSV mapping file.
        """
        spark = self.spark
        df = self.df  # your expression dataframe

        # ---- checks ----
        if source_col not in df.columns or id_col not in df.columns:
            raise ValueError(f"`self.df` must contain '{source_col}' and '{id_col}'")

        # ---- load mapping (TSV) ----
        mapping_df = (
            spark.read
            .option('header', True)
            .option('inferSchema', True)
            .option('sep', '\t')
            .csv(biosample_mapping_reference_path)
        ).drop('Count')  # drop any extraneous columns
        if key_col not in mapping_df.columns:
            raise ValueError(f"Mapping file must contain '{key_col}'")

        other_cols = [c for c in mapping_df.columns if c != key_col]
        if not other_cols:
            raise ValueError('Mapping file must have at least one non-key column with tokens to match')

        # ---- helpers (UDF-free) ----
        def norm(col):
            # Exact-match key: lowercased, non-alphanumerics removed.
            return f.regexp_replace(f.lower(f.col(col)), r'[^a-z0-9]', '')

        def pretty(col):
            """Human-friendly display of strings.

            - '_'/'-' → spaces
            - split camelCase and 'TReg' → 'T Reg'
            - keep acronyms/digit mixes (CD4, HLA, NK) and single-letter caps as-is
            - lowercase other words
            - remove , and .
            """
            s = f.coalesce(f.col(col).cast('string'), f.lit(''))

            # unify separators
            s = f.regexp_replace(s, r'[_-]+', ' ')
            # 'TReg' -> 'T Reg' (without breaking 'CD4')
            s = f.regexp_replace(s, r'([A-Z])([A-Z][a-z])', r'\1 \2')
            # split camelCase: adrenalGland -> adrenal Gland
            s = f.regexp_replace(s, r'(?<=[a-z])(?=[A-Z])', ' ')
            # number/letter boundaries (won't split inside 'CD4')
            s = f.regexp_replace(s, r'(?<=[a-z])(?=\d)', ' ')
            s = f.regexp_replace(s, r'(?<=\d)(?=[a-z])', ' ')
            # remove , and .
            s = f.regexp_replace(s, r'[.,]', '')
            # normalize spaces
            s = f.regexp_replace(s, r'\s+', ' ')
            s = f.trim(s)

            words = f.split(s, ' ')
            words2 = f.transform(
                words,
                lambda w: f.when(
                    w.rlike(r'^[A-Z0-9]{2,}$') | w.rlike(r'^[A-Z]$'),  # keep CD4/HLA/NK/T/B
                    w
                ).otherwise(f.lower(w))
            )
            return f.array_join(words2, ' ')

        def clean_tokens(colname, delim=';'):
            s = f.coalesce(f.col(colname).cast('string'), f.lit(''))
            arr = f.split(s, delim)
            arr = f.transform(arr, lambda x: f.trim(x))
            return f.filter(arr, lambda x: x.isNotNull() & (x != ''))  # noqa: PLC1901

        # ---- build long mapping directly (skip any list middleman) ----
        long_parts = [
            mapping_df.select(f.col(key_col), f.explode_outer(clean_tokens(c)).alias('value'))
            for c in other_cols
        ]
        mapping_long = reduce(lambda a, b: a.unionByName(b), long_parts) \
            .filter(f.col('value').isNotNull() & (f.col('value') != ''))  # noqa: PLC1901

        # Enrich with norm & pretty; choose ONE row per norm deterministically (min key).
        enriched = (
            mapping_long
            .withColumn('norm', norm('value'))
            .withColumn('pretty', pretty('value'))
            .filter(f.col('norm') != '')  # noqa: PLC1901
        )
        min_key_by_norm = enriched.groupBy('norm').agg(f.min(f.col(key_col)).alias(key_col))
        mapping_norm = (
            enriched.join(min_key_by_norm, on=['norm', key_col], how='inner')
            .select('norm', key_col, 'pretty')
            .dropDuplicates(['norm'])  # one mapping per normalized term
        )

        # ---- normalize df[source_col], join, and replace ----
        df_norm = df.withColumn('norm', norm(source_col)).drop(id_col)
        joined = df_norm.join(mapping_norm, on='norm', how='left')

        df_updated = (
            joined
            .withColumn(id_col, f.col(key_col))
            .withColumn(source_col, f.coalesce(f.col('pretty'), f.col(source_col)))
            .drop('norm', key_col, 'pretty')
        )

        # store and return
        self.df = df_updated

    def add_parental_biosample_id(self,
                                   biosample_index_path: str,
                                   potential_parents_path: str,
                                   id_col: str):
        """Add a parental biosample ID.

        Add a parental biosample ID by finding the first ancestor that matches a
        list of potential parents.
        """
        spark = self.spark
        df = self.df

        # Check that id_col exists
        if id_col not in df.columns:
            raise ValueError(f"`self.df` must contain '{id_col}'")

        # Determine parent column name and source column name
        if id_col == 'tissueBiosampleId':
            parent_col_name = 'tissueBiosampleParentId'
            source_col = 'tissueBiosampleFromSource'
        elif id_col == 'celltypeBiosampleId':
            parent_col_name = 'celltypeBiosampleParentId'
            source_col = 'celltypeBiosampleFromSource'
        else:
            raise ValueError(f"id_col must be 'tissueBiosampleId' or 'celltypeBiosampleId', got '{id_col}'")

        # Load biosample index (parquet) - expecting columns: biosampleId, ancestors (array)
        biosample_index = spark.read.parquet(biosample_index_path)

        # Load potential parents (TSV) - expecting at least one column with biosample IDs
        potential_parents_df = (
            spark.read
            .option('header', True)
            .option('inferSchema', True)
            .option('sep', '\t')
            .csv(potential_parents_path)
        )

        # Get the list of potential parents as an ordered array
        # Assuming first column contains the parent IDs
        parent_col = potential_parents_df.columns[0]
        potential_parents_list = [row[parent_col] for row in potential_parents_df.select(parent_col).collect()]

        # Calculate ancestor counts for potential parents to determine priority
        # We want to prioritize parents with MORE ancestors (more specific)
        parent_ancestor_counts = (
            biosample_index
            .filter(f.col('biosampleId').isin(potential_parents_list))
            .select('biosampleId', f.size(f.coalesce(f.col('ancestors'), f.array())).alias('ancestor_count'))
            .collect()
        )

        # Create a map of id -> count
        count_map = {row.biosampleId: row.ancestor_count for row in parent_ancestor_counts}

        # Sort the list by ancestor count descending
        # Parents with no info in biosample_index get count -1 (lowest priority)
        potential_parents_list.sort(key=lambda x: count_map.get(x, -1), reverse=True)

        # Join with biosample index to get ancestors
        df_with_ancestors = df.join(
            biosample_index.select(
                f.col('biosampleId').alias(id_col),
                f.col('ancestors')
            ),
            on=id_col,
            how='left'
        )

        # Build a CASE WHEN expression that checks parents in order
        # Start with None/null as the default
        parent_expr = f.lit(None).cast('string')

        # Iterate through parents in reverse order (so first parent has highest priority)
        for parent_id in reversed(potential_parents_list):
            parent_expr = f.when(
                (f.col(id_col) == f.lit(parent_id)) |
                f.array_contains(
                    f.coalesce(f.col('ancestors'), f.array()),
                    f.lit(parent_id)
                ),
                f.lit(parent_id)
            ).otherwise(parent_expr)

        df_with_parent = df_with_ancestors.withColumn(
            parent_col_name,
            parent_expr
        ).drop('ancestors')

        # Store temporarily
        self.df = df_with_parent

        # Save original dataframe
        original_df = self.df

        # Filter to rows with null parent
        null_parent_df = self.df.filter(f.col(parent_col_name).isNull())

        if null_parent_df.count() > 0:
            # Use potential_parents TSV as fallback for null parents via override mapping
            logger.info(f'  Using override mapping as fallback for null {parent_col_name}...')
            # Apply override_biosample_id to get parent from mapping
            # Use the same potential_parents TSV which has BiosampleId + labels/synonyms
            self.df = null_parent_df
            self.override_biosample_id(
                biosample_mapping_reference_path=potential_parents_path,
                key_col='BiosampleId',
                source_col=source_col,
                id_col=parent_col_name  # Write directly to parent column
            )
            override_df = self.df

            # Filter to rows with non-null parent
            non_null_parent_df = original_df.filter(f.col(parent_col_name).isNotNull())

            # Union the two dataframes back together
            self.df = non_null_parent_df.unionByName(override_df)
        else:
            logger.info(f'  No null {parent_col_name} found, skipping override fallback.')
            self.df = original_df

    def drop_null_biosample_ids(self):
        """Drop rows where any of the specified celltypeBiosample AND tissueBiosample are null."""
        # First check if the columns exist and add them if they don't, populating with nulls
        if 'tissueBiosampleId' not in self.df.columns:
            self.df = self.df.withColumn('tissueBiosampleId', f.lit(None).cast('string'))
            self.df = self.df.withColumn('tissueBiosampleFromSource', f.lit(None).cast('string'))
            self.df = self.df.withColumn('tissueBiosampleParentId', f.lit(None).cast('string'))
        if 'celltypeBiosampleId' not in self.df.columns:
            self.df = self.df.withColumn('celltypeBiosampleId', f.lit(None).cast('string'))
            self.df = self.df.withColumn('celltypeBiosampleFromSource', f.lit(None).cast('string'))
            self.df = self.df.withColumn('celltypeBiosampleParentId', f.lit(None).cast('string'))

        # Find the rows where both tissue and celltype biosample ID are null or
        # tissue and cell biosample parent ID are null
        nulls = self.df.filter(
            ((self.df['tissueBiosampleId'].isNull()) & (self.df['celltypeBiosampleId'].isNull())) |
            ((self.df['tissueBiosampleParentId'].isNull()) & (self.df['celltypeBiosampleParentId'].isNull()))
        )
        logger.info(f'The following biosample from source have null biosample IDs: {nulls}')
        # Then drop those rows from the dataframe
        self.df = self.df.filter(
            ~((self.df['tissueBiosampleId'].isNull()) & (self.df['celltypeBiosampleId'].isNull()))
        )
        self.df = self.df.filter(
            ~(
                (self.df['tissueBiosampleParentId'].isNull())
                & (self.df['celltypeBiosampleParentId'].isNull())
            )
        )

    def within_donor_mean(self,
                        tissue_col: str = 'tissueBiosampleId',
                        celltype_col: str = 'celltypeBiosampleId',
                        expr_col: str = 'expression',
                        sep: str = ', '):
        """Group by ALL columns except `expr_col`, `tissue_col`, and `celltype_col`."""
        df = self.df

        # Columns to group by
        group_cols = [c for c in df.columns if c not in {expr_col, tissue_col, celltype_col}]

        if group_cols:
            agg_df = (
                df.groupBy(group_cols)
                .agg(
                    f.avg(f.col(expr_col)).alias(expr_col),
                    f.collect_set(f.col(tissue_col)).alias('_tissues'),
                    f.collect_set(f.col(celltype_col)).alias('_celltypes'),
                )
            )
        else:
            # If nothing to group by, aggregate globally
            agg_df = df.agg(
                f.avg(f.col(expr_col)).alias(expr_col),
                f.collect_set(f.col(tissue_col)).alias('_tissues'),
                f.collect_set(f.col(celltype_col)).alias('_celltypes'),
            )

        # Helper: trim -> drop null/empty -> sort -> join
        def join_sorted(arr_col, out_name):
            arr_trim = f.transform(f.col(arr_col), lambda x: f.when(x.isNull(), None).otherwise(f.trim(x)))
            arr_keep = f.filter(arr_trim, lambda x: x.isNotNull() & (x != ''))  # noqa: PLC1901
            arr_sorted = f.array_sort(arr_keep)  # simple lexicographic sort; duplicates preserved
            return f.when(f.size(arr_sorted) > 0, f.array_join(arr_sorted, sep)).otherwise(f.lit(None)).alias(out_name)

        agg_df = (
            agg_df
            .withColumn(tissue_col, join_sorted('_tissues', tissue_col))
            .withColumn(celltype_col, join_sorted('_celltypes', celltype_col))
            .drop('_tissues', '_celltypes')
        )

        self.df = agg_df

    def calculate_quartiles(self, local=False):
        """This function calculates the expression quartiles of each gene across all donors."""
        # Define the grouping cols
        groupby_cols = [
            'targetId',
            'datasourceId',
            'datatypeId',
            'unit'
        ]
        if 'tissueBiosampleId' in self.df.columns:
            groupby_cols.append('tissueBiosampleId')
            groupby_cols.append('tissueBiosampleFromSource')
        if 'celltypeBiosampleId' in self.df.columns:
            groupby_cols.append('celltypeBiosampleId')
            groupby_cols.append('celltypeBiosampleFromSource')

        # Partition by grouping keys
        quartile_df = self.df.repartition(
            *groupby_cols
        )

        # Define the quantile probabilities
        quartile_probs = [0, 0.25, 0.50, 0.75, 1]

        # Group and compute approximate quantiles, default params
        quartile_df = quartile_df.groupBy(
            *groupby_cols
        ).agg(
            f.percentile_approx('expression', quartile_probs).alias('q_vals'))

        quartile_df = quartile_df.select(
            *groupby_cols,
            f.col('q_vals')[0].alias('min'),
            f.col('q_vals')[1].alias('q1'),
            f.col('q_vals')[2].alias('median'),
            f.col('q_vals')[3].alias('q3'),
            f.col('q_vals')[4].alias('max')
        )
        self.df = quartile_df

    def apply_qc_threshold(self, expr_col: str = 'expression', threshold: float = 0.5):
        """Quality control step: set expression values below `threshold` to 0."""
        if expr_col not in self.df.columns:
            # nothing to do
            return

        # Use when/otherwise to set values < threshold to 0
        self.df = self.df.withColumn(
            expr_col,
            f.when(f.col(expr_col) < f.lit(threshold), f.lit(0)).otherwise(
                f.col(expr_col)
            )
        )

    def calculate_expression_distribution(self, local=False, threshold: float = 0.5):
        """This function groups the dataframe by datasourceId, targetId and datatypeId.

        Then calculates the distribution of expression values for each gene.
        """
        # Group by the relevant columns and count the number of non-zero expressions
        return self.df.groupBy(
            'targetId',
            'datasourceId',
            'datatypeId',
            'unit'
        ).agg(
            (f.sum(f.when(f.col('median') > threshold, 1).otherwise(0)) / f.count('*')).alias('distribution_score')
        )

    def load_cellex_data(self, cellex_path, biosample_type='tissue'):
        """Load cellex biosample scores and convert from wide to long format."""
        # Read the cellex data (CSV.gz format)
        cellex_df = (
            self.spark.read
            .option('header', True)
            .option('inferSchema', True)
            .option('sep', ',')
            .csv(cellex_path)
        )

        # Get the first column (gene IDs) and all other columns (biosample IDs)
        gene_col = cellex_df.columns[0]  # First column is gene ID
        biosample_cols = cellex_df.columns[1:]  # All other columns are biosample IDs

        # Create array of structs for each biosample
        biosample_structs = f.array(*[
            f.struct(f.lit(col_name).alias('biosampleId'),
                    f.col(col_name).alias('specificity_score'))
            for col_name in biosample_cols
        ]).alias('biosample_scores')

        # Convert to long format
        return (
            cellex_df
            .select(f.col(gene_col).alias('targetId'),
                   f.explode(biosample_structs).alias('x'))
            .select(
                'targetId',
                f.col('x.biosampleId').alias('biosampleId'),
                f.col('x.specificity_score').alias('specificity_score')
            )
            .filter(f.col('specificity_score').isNotNull())  # Remove null scores
        )

    def add_expression_specificity(self, cellex_path, biosample_type='tissue'):
        """Add expression specificity scores to the aggregated expression data."""
        # Load cellex data
        cellex_df = self.load_cellex_data(cellex_path, biosample_type)

        # Determine which biosample column to join on
        if biosample_type == 'tissue':
            biosample_col = 'tissueBiosampleId'
        elif biosample_type == 'celltype':
            biosample_col = 'celltypeBiosampleId'
        elif biosample_type == 'both':
            biosample_col = 'celltypeBiosampleId__tissueBiosampleId'
            # Create the combined column in the main dataframe
            self.df = self.df.withColumn(
                biosample_col,
                f.concat(f.col('celltypeBiosampleId'), f.lit('__'), f.col('tissueBiosampleId'))
            )
        else:
            raise ValueError("biosample_type must be  'tissue', 'celltype' or 'both'")

        # rename the biosample column to the biosample column in the main dataframe
        cellex_df = cellex_df.withColumnRenamed('biosampleId', biosample_col)

        # Join with the main dataframe
        self.df = (
            self.df
            .join(cellex_df,
                  on=['targetId', biosample_col],
                  how='left')
        )
        # Drop the celltypeBiosampleId__tissueBiosampleId column
        if biosample_type == 'both':
            self.df = self.df.drop('celltypeBiosampleId__tissueBiosampleId')

    def write_data(self, output_directory, json=False):
        """This function writes the DataFrame to parquet format."""
        # if not os.path.exists(output_directory):
        #     os.makedirs(output_directory)
        if json:
            self.df.write.mode('overwrite') \
                .json(output_directory)
        else:
            # If not JSON, write as parquet
            self.df.write.mode('overwrite') \
                .parquet(output_directory)


def baseline_expression_aggregate(
    source: str | dict[str, str],
    destination: str | dict[str, str],
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    logger.info('Starting baseline expression aggregation')

    # Initialize Spark Session
    if properties is None:
        properties = {}

    session = Session(app_name='baseline_expression_aggregate', properties=properties)
    spark = session.spark

    # Extract arguments
    if not isinstance(source, dict):
        raise TypeError(f'Source must be a dictionary containing file paths. Got: {type(source)}')

    directory = source.get('expression_data')
    biosample_index = source.get('biosample_index')
    tissue_parents = source.get('tissue_parents')
    celltype_parents = source.get('celltype_parents')
    tissue_cellex = source.get('tissue_cellex')
    celltype_cellex = source.get('celltype_cellex')
    both_cellex = source.get('both_cellex')

    if isinstance(destination, str):
        output_dir = destination
    else:
        output_dir = list(destination.values())[0]

    # Settings
    json_output = settings.get('json', False)
    local = settings.get('local', False)

    eq = AggregateExpression(spark)
    eq.load_data(directory, local=local)

    logger.info('Calculating quartiles...')
    eq.calculate_quartiles(local=local)

    logger.info('Calculating expression distribution...')
    distribution_df = eq.calculate_expression_distribution()
    # Add the distribution score to the main dataframe
    eq.df = eq.df.join(distribution_df, on=['targetId', 'datasourceId', 'datatypeId', 'unit'], how='left')

    # Add expression specificity scores if cellex file is provided
    if tissue_cellex:
        logger.info('Adding tissue-specific expression specificity scores...')
        eq.add_expression_specificity(tissue_cellex, biosample_type='tissue')
    elif celltype_cellex:
        logger.info('Adding celltype-specific expression specificity scores...')
        eq.add_expression_specificity(celltype_cellex, biosample_type='celltype')
    elif both_cellex:
        logger.info('Adding combined tissue+celltype expression specificity scores...')
        eq.add_expression_specificity(both_cellex, biosample_type='both')

    # Add parental biosample IDs if requested
    if tissue_parents is not None:
        logger.info('Adding tissue parental biosample IDs...')
        eq.add_parental_biosample_id(
            biosample_index_path=biosample_index,
            potential_parents_path=tissue_parents,
            id_col='tissueBiosampleId'
        )
    if celltype_parents is not None:
        logger.info('Adding celltype parental biosample IDs...')
        eq.add_parental_biosample_id(
            biosample_index_path=biosample_index,
            potential_parents_path=celltype_parents,
            id_col='celltypeBiosampleId'
        )

    logger.info('Packing data for output...')
    # Logic from original script to append datasource/format/dataset_name
    # But here we might want to trust the destination path provided in config
    # The original script did:
    # parts = [seg for seg in os.path.normpath(directory).split(os.sep) if seg]
    # datasource = parts[-3]
    # dataset_name   = parts[-1]
    # output_dir = f"{output_dir}/{datasource}/{file_format}/{dataset_name}"

    # If we use the destination from config directly, it is `intermediate/baseline_expression/dice_aggregated`
    # The user might expect the same structure.
    # However, `pts` usually defines explicit output paths.
    # I will use the destination path as is.

    eq.write_data(output_dir, json=json_output)
    logger.info(f'Data written to {output_dir}')

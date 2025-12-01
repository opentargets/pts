"""PySpark-based entityId propagation from children to parents."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as f


def separate_tree_to_nodes_and_edges(df: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Separate tree structure into nodes and edges DataFrames.

    This function processes the output of prepare_dataset_for_propagation and
    extracts two separate DataFrames: one for nodes (with aggregated entityIds)
    and one for edges (parent-child relationships).

    Args:
        df: PySpark DataFrame with columns:
            - id: string - the entity identifier
            - parent_id: string - the parent entity identifier
            - entityIds: array<string> - list of entity IDs associated with this (id, parent_id) pair

    Returns:
        Tuple of two PySpark DataFrames:
        1. Nodes DataFrame with columns:
           - id: string - unique node identifiers
           - entityIds: array<string> - aggregated/union of all entityIds for each id
        2. Edges DataFrame with columns:
           - child_id: string - the child node identifier (from input id)
           - parent_id: string - the parent node identifier (from input parent_id)
    """
    # Extract edges: select id as child_id and parent_id
    edges = df.select(
        f.col('id').alias('child_id'),
        f.col('parent_id'),
    ).distinct()

    # Extract nodes: group by id and aggregate entityIds
    # For each id, we need to union all entityIds from all rows where that id appears
    nodes = (
        df.groupBy('id')
        .agg(f.collect_list('entityIds').alias('entityIds_list'))
        .withColumn(
            'entityIds',
            f.array_distinct(
                f.flatten(
                    f.filter(
                        f.col('entityIds_list'),
                        lambda x: x.isNotNull(),
                    )
                )
            ),
        )
        .select('id', 'entityIds')
    )

    return nodes, edges


def efficient_entity_id_propagation(
    nodes_df: DataFrame, edges_df: DataFrame, max_iterations: int = 100, return_iterations: bool = False
) -> DataFrame | tuple[DataFrame, int]:
    """Efficiently propagate entityIds from children to parents using leaf-to-root algorithm.

    This function implements an efficient bottom-up propagation algorithm that processes
    nodes in layers, starting from leaves (nodes with no children) and propagating
    entityIds upward to their parents. This approach is more efficient than top-down
    propagation as it processes nodes in batches and removes processed edges.

    Args:
        nodes_df: PySpark DataFrame with columns:
            - id: string - unique node identifiers
            - entityIds: array<string> - initial entity IDs (may be empty or null)
        edges_df: PySpark DataFrame with columns:
            - child_id: string - the child node identifier
            - parent_id: string - the parent node identifier
        max_iterations: Maximum number of iterations to prevent infinite loops. Defaults to 100.
        return_iterations: If True, returns a tuple (result_df, iterations). Defaults to False.

    Returns:
        PySpark DataFrame with columns:
            - id: string - unique node identifiers
            - entityIds: array<string> - updated entity IDs after propagation
              (includes propagated values from all descendants)
        If return_iterations is True, returns (DataFrame, int) where int is the number of iterations.

    Algorithm:
        1. Cache edges and nodes DataFrames for efficiency
        2. Repeat until no edges remain:
           a. Find leaves: nodes that never appear as parent (using left_anti join)
           b. Send messages from leaves to their parents: collect entityIds from leaves grouped by parent_id
           c. Update parents: merge (union) parent's stored entityIds with propagated ones
           d. Remove processed leaves from the edge list (using left_anti join)
           e. Cache updated DataFrames
        3. Return updated nodes DataFrame

    Raises:
        RuntimeError: If propagation does not converge within max_iterations.
    """
    edges = edges_df.cache()
    nodes = nodes_df.cache()

    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        # 1. Find leaves = nodes that never appear as parent
        parents = edges.select('parent_id').distinct()
        leaves = edges.select('child_id').distinct().join(parents, f.col('child_id') == f.col('parent_id'), 'left_anti')

        leaf_count = leaves.count()
        if leaf_count == 0:
            break

        # 2. Send messages from leaves to their parents
        # Join leaves with nodes to get their entityIds, then join with edges to get parents
        leaf_messages = (
            leaves.alias('leaves')
            .join(nodes.alias('nodes'), f.col('leaves.child_id') == f.col('nodes.id'))
            .join(edges.alias('edges'), f.col('leaves.child_id') == f.col('edges.child_id'))
            .groupBy('edges.parent_id')
            .agg(
                f.array_distinct(
                    f.flatten(
                        f.filter(
                            f.collect_list('nodes.entityIds'),
                            lambda x: x.isNotNull(),
                        )
                    )
                ).alias('msgs')
            )
            .select(f.col('parent_id'), f.col('msgs'))
        )

        # 3. Update parents: merge (union) parent's stored entityIds with propagated ones
        new_nodes = (
            nodes.join(leaf_messages, f.col('id') == f.col('parent_id'), 'left')
            .withColumn(
                'entityIds',
                f.when(
                    f.col('msgs').isNotNull(),
                    f.array_distinct(
                        f.array_union(
                            f.coalesce(f.col('entityIds'), f.array().cast('array<string>')),
                            f.col('msgs'),
                        )
                    ),
                ).otherwise(f.coalesce(f.col('entityIds'), f.array().cast('array<string>'))),
            )
            .drop('parent_id', 'msgs')
            .cache()
        )

        # 4. Remove processed leaves from the edge list
        new_edges = (
            edges.alias('edges')
            .join(leaves.alias('leaves'), f.col('edges.child_id') == f.col('leaves.child_id'), 'left_anti')
            .select(f.col('edges.child_id'), f.col('edges.parent_id'))
            .cache()
        )

        edges.unpersist()
        nodes.unpersist()
        new_edges.cache()
        new_nodes.cache()
        edges = new_edges
        nodes = new_nodes

        # break lineage
        if iteration % 2 == 0:
            edges = edges.checkpoint()
            nodes = nodes.checkpoint()

            edges.count()
            nodes.count()

    if iteration >= max_iterations:
        raise RuntimeError(
            f'Propagation did not converge after {max_iterations} iterations. '
            'This indicates a potential infinite loop or very deep hierarchy.'
        )

    if return_iterations:
        return nodes, iteration
    return nodes


def merge_nodes_and_edges(nodes_df: DataFrame, edges_df: DataFrame) -> DataFrame:
    """Merge nodes and edges DataFrames back into a single DataFrame.

    This function is the inverse of separate_tree_to_nodes_and_edges. It takes
    separate nodes and edges DataFrames and merges them back into a single
    DataFrame with (id, parent_id, entityIds) columns.

    Args:
        nodes_df: PySpark DataFrame with columns:
            - id: string - unique node identifiers
            - entityIds: array<string> - entity IDs for each node
        edges_df: PySpark DataFrame with columns:
            - child_id: string - the child node identifier
            - parent_id: string - the parent node identifier

    Returns:
        PySpark DataFrame with columns:
            - id: string - the entity identifier (from edges.child_id)
            - parent_id: string - the parent entity identifier (from edges.parent_id)
            - entityIds: array<string> - entity IDs from the corresponding node
              (empty array if node not found)

    Example:
        >>> nodes = spark.createDataFrame([('A', ['ENSG1']), ('B', ['ENSG2'])], ['id', 'entityIds'])
        >>> edges = spark.createDataFrame([('A', 'B')], ['child_id', 'parent_id'])
        >>> merged = merge_nodes_and_edges(nodes, edges)
        >>> # Result: (id='A', parent_id='B', entityIds=['ENSG1'])
    """
    # Join edges with nodes where edges.child_id == nodes.id
    # Use left join to keep all edges even if node is missing
    from pyspark.sql.types import ArrayType, StringType, StructField, StructType

    result_df = (
        edges_df.alias('edges')
        .join(nodes_df.alias('nodes'), f.col('edges.child_id') == f.col('nodes.id'), 'left')
        .select(
            f.col('edges.child_id').alias('id'),
            f.col('edges.parent_id'),
            f.coalesce(f.col('nodes.entityIds'), f.array().cast('array<string>')).alias('entityIds'),
        )
    )

    # Reconstruct DataFrame with explicit non-nullable schema for entityIds
    # We need to create a new DataFrame with the correct schema
    spark = result_df.sparkSession
    expected_schema = StructType([
        StructField('id', result_df.schema['id'].dataType, nullable=False),
        StructField('parent_id', result_df.schema['parent_id'].dataType, nullable=True),
        StructField('entityIds', ArrayType(StringType(), containsNull=False), nullable=False),
    ])

    # Create DataFrame with explicit schema
    return spark.createDataFrame(result_df.rdd, schema=expected_schema)


def propagate_entity_ids_pyspark_efficiently(
    df: DataFrame, return_iterations: bool = False
) -> DataFrame | tuple[DataFrame, int]:
    """Propagate entityIds from children to parents using efficient leaf-to-root algorithm.

    This function uses a more efficient bottom-up propagation algorithm. It separates
    the input DataFrame into nodes and edges, runs efficient propagation, and merges
    the results back.

    Args:
        df: PySpark DataFrame with columns:
            - id: string - the entity identifier
            - parent_id: string - the parent entity identifier (can be null for root nodes)
            - entityIds: array<string> - list of entity IDs associated with this (id, parent_id) pair
        return_iterations: If True, returns a tuple (result_df, iterations). Defaults to False.

    Returns:
        PySpark DataFrame with same structure but updated entityIds where parents
        now include entityIds from all their descendants.
        If return_iterations is True, returns (DataFrame, int) where int is the number of iterations.

    Algorithm:
        1. Separate input DataFrame into nodes and edges using separate_tree_to_nodes_and_edges
        2. Run efficient_entity_id_propagation on nodes and edges
        3. Merge updated nodes with original edges using merge_nodes_and_edges
        4. Verify output schema matches input schema
        5. Return result (with iterations if requested)
    """
    # Store original schema for validation
    input_schema = df.schema

    # Step 1: Separate input DataFrame to nodes and edges
    nodes, edges = separate_tree_to_nodes_and_edges(df)

    # Step 2: Run efficient propagation
    if return_iterations:
        updated_nodes, iterations = efficient_entity_id_propagation(nodes, edges, return_iterations=True)
    else:
        updated_nodes = efficient_entity_id_propagation(nodes, edges, return_iterations=False)
        iterations = None

    # Step 3: Merge updated nodes with original edges
    result = merge_nodes_and_edges(updated_nodes, edges)

    # Step 4: Verify output schema matches input schema
    output_schema = result.schema
    if len(output_schema) != len(input_schema):
        raise ValueError(f'Schema mismatch: output has {len(output_schema)} columns, expected {len(input_schema)}')

    # Check column names and types match
    for i, (input_field, output_field) in enumerate(zip(input_schema, output_schema, strict=False)):
        if input_field.name != output_field.name:
            raise ValueError(
                f'Column name mismatch at position {i}: expected {input_field.name}, got {output_field.name}'
            )
        if input_field.dataType != output_field.dataType:
            raise ValueError(
                f'Column type mismatch for {input_field.name}: '
                f'expected {input_field.dataType}, got {output_field.dataType}'
            )

    # Step 5: Return result
    if return_iterations:
        return result, iterations
    return result

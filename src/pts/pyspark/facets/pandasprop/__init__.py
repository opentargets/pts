"""Pandas-based and PySpark-based entityId propagation from children to parents."""

from __future__ import annotations

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def propagate_entity_ids(df: pd.DataFrame, return_iterations: bool = False) -> pd.DataFrame | tuple[pd.DataFrame, int]:
    """Propagate entityIds from children to parents iteratively.

    This function propagates entityIds through a hierarchy where each row
    represents a (id, parent_id) pair with associated entityIds. The function
    iteratively propagates entityIds from children to their parents using
    tree traversal from the top until all nodes are processed.

    Args:
        df: DataFrame with columns:
            - id: str - the entity identifier
            - parent_id: str - the parent entity identifier
            - entityIds: list[str] - list of entity IDs associated with this (id, parent_id) pair
        return_iterations: If True, returns a tuple (result_df, iterations). Defaults to False.

    Returns:
        DataFrame with same structure but updated entityIds where parents
        now include entityIds from all their descendants.
        If return_iterations is True, returns (DataFrame, int) where int is the number of iterations.

    Algorithm:
        1. Step a0: Find nodes that have children (nodes whose id appears as parent_id)
        2. Step a: Left join to get parent-child relationships and split nodes:
           - Nodes without children → add to result
           - Nodes with children → process (copy children's entityIds to parent)
        3. Step b: Group by (id, parent_id) and union entityIds from children
        4. Repeat until no nodes with children remain
    """
    current_df = df.copy()
    max_iterations = 40
    iteration = 0
    result_df = pd.DataFrame(columns=['id', 'parent_id', 'entityIds'])

    while iteration < max_iterations:
        iteration += 1

        # Step a0: Find nodes that have children (nodes whose id appears as parent_id)
        # These are the nodes that will be processed in this iteration
        nodes_with_children = (
            current_df[current_df['parent_id'].notna()]
            .merge(
                current_df[['id']].drop_duplicates(),
                left_on='parent_id',
                right_on='id',
                how='inner',
            )
            .drop_duplicates(subset=['parent_id'])[['parent_id']]
            .rename(columns={'parent_id': 'id'})
        )

        # If no nodes have children, we're done
        if len(nodes_with_children) == 0:
            # Add all remaining nodes to result
            result_df = pd.concat([result_df, current_df], ignore_index=True)
            break

        # Get children for nodes that have children
        # Find all rows where parent_id is in nodes_with_children
        children_df = current_df.merge(
            nodes_with_children,
            left_on='parent_id',
            right_on='id',
            how='inner',
            suffixes=('', '_parent'),
        ).rename(columns={'id': 'child_id', 'parent_id': 'child_parent_id', 'entityIds': 'child_entityIds'})[
            ['child_id', 'child_parent_id', 'child_entityIds']
        ]

        # Step a: Left join to get parent-child relationships
        # Left join: current_df with children_df where current_df.id = children_df.child_parent_id
        joined = current_df.merge(children_df, left_on='id', right_on='child_parent_id', how='left')
        # Result columns: id, parent_id, entityIds, child_id, child_parent_id, child_entityIds

        # Split nodes: nodes without children (child_id is NaN) vs nodes with children
        nodes_without_children = joined[joined['child_id'].isna()][['id', 'parent_id', 'entityIds']].copy()
        nodes_to_process = joined[joined['child_id'].notna()].copy()

        # Add nodes without children to result
        if len(nodes_without_children) > 0:
            result_df = pd.concat([result_df, nodes_without_children], ignore_index=True)

        # Convergence check: if leftover nodes (nodes without children) are non-empty, we continue
        # If no nodes to process, we're done
        if len(nodes_to_process) == 0:
            break

        # Clean up nodes_to_process: drop redundant child_parent_id, rename for clarity
        nodes_to_process = nodes_to_process.drop(columns=['child_parent_id'])
        nodes_to_process = nodes_to_process.rename(columns={'child_entityIds': 'child_entity_ids'})
        # Final columns: id, parent_id, entityIds, child_id, child_entity_ids

        # Step b: Group by (id, parent_id) and union entityIds
        def union_entity_ids(group):
            # Get original entityIds for this (id, parent_id) pair
            original_ids = set(group['entityIds'].iloc[0]) if len(group) > 0 and group['entityIds'].iloc[0] else set()

            # Get all child entityIds
            child_ids = set()
            for child_entity_ids in group['child_entity_ids'].dropna():
                if child_entity_ids:
                    child_ids.update(child_entity_ids)

            # Union and return as sorted list
            return sorted(original_ids.union(child_ids))

        processed = (
            nodes_to_process.groupby(['id', 'parent_id'], dropna=False)
            .apply(lambda group: pd.Series({'entityIds': union_entity_ids(group)}), include_groups=False)
            .reset_index()
        )
        # Result columns: id, parent_id, entityIds (updated)

        # Check convergence: leftover nodes (nodes that didn't match in left join) should be non-empty
        # This means we still have nodes to process
        current_df = processed

    if iteration >= max_iterations:
        # Add any remaining nodes to result
        if len(current_df) > 0:
            result_df = pd.concat([result_df, current_df], ignore_index=True)
        # Raise error if max iterations reached
        raise RuntimeError(
            f'Propagation did not converge after {max_iterations} iterations. '
            'This indicates a potential infinite loop or very deep hierarchy.'
        )

    if return_iterations:
        return result_df, iteration
    return result_df


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
        F.col('id').alias('child_id'),
        F.col('parent_id'),
    ).distinct()

    # Extract nodes: group by id and aggregate entityIds
    # For each id, we need to union all entityIds from all rows where that id appears
    nodes = (
        df.groupBy('id')
        .agg(F.collect_list('entityIds').alias('entityIds_list'))
        .withColumn(
            'entityIds',
            F.array_distinct(
                F.flatten(
                    F.filter(
                        F.col('entityIds_list'),
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
        leaves = edges.select('child_id').distinct().join(parents, F.col('child_id') == F.col('parent_id'), 'left_anti')

        leaf_count = leaves.count()
        if leaf_count == 0:
            break

        # 2. Send messages from leaves to their parents
        # Join leaves with nodes to get their entityIds, then join with edges to get parents
        leaf_messages = (
            leaves.alias('leaves')
            .join(nodes.alias('nodes'), F.col('leaves.child_id') == F.col('nodes.id'))
            .join(edges.alias('edges'), F.col('leaves.child_id') == F.col('edges.child_id'))
            .groupBy('edges.parent_id')
            .agg(
                F.array_distinct(
                    F.flatten(
                        F.filter(
                            F.collect_list('nodes.entityIds'),
                            lambda x: x.isNotNull(),
                        )
                    )
                ).alias('msgs')
            )
            .select(F.col('parent_id'), F.col('msgs'))
        )

        # 3. Update parents: merge (union) parent's stored entityIds with propagated ones
        new_nodes = (
            nodes.join(leaf_messages, F.col('id') == F.col('parent_id'), 'left')
            .withColumn(
                'entityIds',
                F.when(
                    F.col('msgs').isNotNull(),
                    F.array_distinct(
                        F.array_union(
                            F.coalesce(F.col('entityIds'), F.array().cast('array<string>')),
                            F.col('msgs'),
                        )
                    ),
                ).otherwise(F.coalesce(F.col('entityIds'), F.array().cast('array<string>'))),
            )
            .drop('parent_id', 'msgs')
            .cache()
        )

        # 4. Remove processed leaves from the edge list
        new_edges = (
            edges.alias('edges')
            .join(leaves.alias('leaves'), F.col('edges.child_id') == F.col('leaves.child_id'), 'left_anti')
            .select(F.col('edges.child_id'), F.col('edges.parent_id'))
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
        .join(nodes_df.alias('nodes'), F.col('edges.child_id') == F.col('nodes.id'), 'left')
        .select(
            F.col('edges.child_id').alias('id'),
            F.col('edges.parent_id'),
            F.coalesce(F.col('nodes.entityIds'), F.array().cast('array<string>')).alias('entityIds'),
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

    This function has the same interface as propagate_entity_ids_pyspark but uses
    a more efficient bottom-up propagation algorithm. It separates the input DataFrame
    into nodes and edges, runs efficient propagation, and merges the results back.

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
                f'Column type mismatch for {input_field.name}: expected {input_field.dataType}, got {output_field.dataType}'
            )

    # Step 5: Return result
    if return_iterations:
        return result, iterations
    return result

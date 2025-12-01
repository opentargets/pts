"""Backend-compatible entry point for gene sets computation.

This module provides a backend-compatible wrapper for the gene sets (target facets) functionality.
It follows the backend's expected signature: (source, destination, settings, properties).
"""

from typing import Any

from loguru import logger

from pts.pyspark.facets.gene_sets import target_facets as compute_target_facets


def gene_sets(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Backend-compatible entry point for computing gene sets.

    This function wraps the main target_facets implementation to match the backend's
    expected signature. It extracts category_config from settings and passes it along.

    Args:
        source: Dictionary with 'targets', 'go', and 'reactome' keys pointing to input paths.
            Paths are relative to work_path or release_uri (backend handles prefixing).
        destination: Dictionary with 'targets' key pointing to output path.
            Path is relative to work_path or release_uri (backend handles prefixing).
        settings: Optional dictionary with settings. Can contain 'category_config' for
            custom category name mappings.
        properties: Optional Spark configuration properties.

    Example config.yaml entry:
        steps:
          gene_sets:
            - name: compute target facets
              source:
                targets: output/target
                go: output/go/go.parquet
                reactome: input/reactome
              destination:
                targets: output/gene_sets
              pyspark: gene_sets
    """
    logger.info('Starting gene sets computation (backend entry point)')

    # Extract category_config from settings if provided
    category_config = settings.get('category_config') if settings else None

    # Call the main implementation
    compute_target_facets(
        source=source,
        destination=destination,
        properties=properties,
        category_config=category_config,
    )

    logger.info('Gene sets computation complete')

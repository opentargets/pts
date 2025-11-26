"""Facets module for computing search facets from target and disease data.

This module provides functionality to compute various facets (filters) that are used
in the Open Targets Platform search interface. Facets allow users to filter and
explore targets and diseases based on different attributes.
"""

from pts.pyspark.facets.target_facets import (
    FacetSearchCategories,
    compute_all_target_facets,
    compute_approved_name_facets,
    compute_approved_symbol_facets,
    compute_go_facets,
    compute_pathways_facets,
    compute_subcellular_locations_facets,
    compute_target_class_facets,
    compute_target_id_facets,
    compute_tractability_facets,
    target_facets,
)

__all__ = [
    'FacetSearchCategories',
    'compute_all_target_facets',
    'compute_approved_name_facets',
    'compute_approved_symbol_facets',
    'compute_go_facets',
    'compute_pathways_facets',
    'compute_subcellular_locations_facets',
    'compute_target_class_facets',
    'compute_target_id_facets',
    'compute_tractability_facets',
    'target_facets',
]

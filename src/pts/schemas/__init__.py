"""Schema definitions for PySpark DataFrames.

This module contains PySpark schema definitions for various data structures
used in the Open Targets pipelines.
"""

from pts.schemas.facet import facet_schema
from pts.schemas.go import go_schema
from pts.schemas.reactome import reactome_schema

__all__ = ['facet_schema', 'go_schema', 'reactome_schema']


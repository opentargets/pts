"""Schema definitions for PySpark DataFrames.

This module contains PySpark schema definitions for various data structures
used in the Open Targets pipelines.
"""

from pts.schemas.go import go_schema
from pts.schemas.reactome import reactome_schema

__all__ = ['go_schema', 'reactome_schema']

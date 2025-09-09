"""JSON Schema validation module for PTS.

This module provides functionality to validate JSON data against schemas
using the opentargets-validator package.
"""

from pts.validation.schema_validator import validate_json_schema

__all__ = ['validate_json_schema']

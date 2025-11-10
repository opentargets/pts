# Target Facets Module

Python implementation of target facets computation for Open Targets search functionality.

## Overview

This module has been migrated from Scala to Python, providing the same functionality for computing various search facets (filters) based on target attributes. The facets are used in the Open Targets Platform to enable users to filter and explore targets based on different characteristics.

## Migration from Scala

### Original Scala Files
- `TargetFacets.scala` - Main facet computation logic
- `Helpers.scala` - Helper functions

### New Python Files
- `target_facets.py` - Main facet computation functions
- `helpers.py` - Helper utility functions
- `__init__.py` - Module exports

### Key Changes
1. **Type System**: Scala case classes → Python type hints with PySpark DataFrames
2. **Logging**: Scala LazyLogging → Python loguru
3. **Spark Session**: Explicit SparkSession management with GCS support
4. **API**: Functional style maintained, but with Pythonic conventions

## Features

### Facet Types Computed

1. **Tractability Facets**: Druggability across different therapeutic modalities
   - Small Molecule (SM) → category: "Tractability Small Molecule"
   - Antibody (AB) → category: "Tractability Antibody"
   - PROTAC (PR) → category: "Tractability PROTAC"
   - Other Modalities (OC) → category: "Tractability Other Modalities"

2. **Target ID Facets**: Ensembl gene IDs (category: "Target ID")

3. **Approved Symbol Facets**: HGNC gene symbols (e.g., TP53, BRCA1) (category: "Approved Symbol")

4. **Approved Name Facets**: Full gene names (category: "Approved Name")

5. **Subcellular Location Facets**: Protein cellular localization (category: "Subcellular Location")

6. **Target Class Facets**: Molecular function categories (e.g., Enzyme, Transporter) (category: "ChEMBL Target Class")

7. **Pathway Facets**: Reactome pathway associations (category: "Reactome")

8. **Gene Ontology Facets**: GO terms across three aspects
   - Molecular Function (F) → category: "GO:MF"
   - Biological Process (P) → category: "GO:BP"
   - Cellular Component (C) → category: "GO:CC"

### Storage Support

The module works with:
- **Local filesystem**: Standard file paths
- **Google Cloud Storage (GCS)**: `gs://` URLs with automatic authentication

## Usage

### Basic Usage

```python
from pts.pyspark.facets import target_facets

# Define source data paths
source = {
    'targets': 'gs://bucket/targets.parquet',
    'go': 'gs://bucket/go.parquet'
}

# Define output path
destination = {
    'targets': 'gs://bucket/output/target_facets.parquet'
}

# Compute all facets
target_facets(source=source, destination=destination)
```

### Local Filesystem Example

```python
from pts.pyspark.facets import target_facets

source = {
    'targets': 'work/output/targets/targets.parquet',
    'go': 'work/output/go/go.parquet'
}

destination = {
    'targets': 'work/output/facets/target_facets.parquet'
}

target_facets(source=source, destination=destination)
```

### Custom Configuration

```python
# Custom Spark properties
properties = {
    'spark.sql.shuffle.partitions': '200',
    'spark.executor.memory': '8g',
}

# Custom category names
category_config = {
    'SM': 'Tractability Small Molecule',
    'AB': 'Tractability Antibody',
    'PR': 'Tractability PROTAC',
    'OC': 'Tractability Other Modalities',
    'goF': 'GO:MF',
    'goP': 'GO:BP',
    'goC': 'GO:CC',
    'approvedSymbol': 'Approved Symbol',
    'approvedName': 'Approved Name',
    'subcellularLocation': 'Subcellular Location',
    'targetClass': 'ChEMBL Target Class',
    'pathways': 'Reactome',
}

target_facets(
    source=source,
    destination=destination,
    properties=properties,
    category_config=category_config
)
```

### Using Individual Facet Functions

For fine-grained control, you can compute individual facet types:

```python
from pts.pyspark.common.session import Session
from pts.pyspark.facets import (
    FacetSearchCategories,
    compute_tractability_facets,
    compute_go_facets,
    compute_pathways_facets,
)

# Initialize Spark
session = Session(app_name='facets')
spark = session.spark

# Load data
targets_df = spark.read.parquet('work/output/targets/targets.parquet')
go_df = spark.read.parquet('work/output/go/go.parquet')

# Initialize categories
categories = FacetSearchCategories()

# Compute specific facets
tractability = compute_tractability_facets(targets_df, categories, spark)
go_facets = compute_go_facets(targets_df, go_df, categories, spark)
pathways = compute_pathways_facets(targets_df, categories, spark)

# Process or write results
tractability.write.mode('overwrite').parquet('work/output/tractability.parquet')
```

### Computing All Facets

```python
from pts.pyspark.facets import compute_all_target_facets, FacetSearchCategories
from pts.pyspark.common.session import Session

session = Session(app_name='all_facets')
spark = session.spark

targets_df = spark.read.parquet('work/output/targets/targets.parquet')
go_df = spark.read.parquet('work/output/go/go.parquet')

categories = FacetSearchCategories()
all_facets = compute_all_target_facets(targets_df, go_df, categories, spark)

all_facets.show(10)
```

## Output Schema

All facet functions produce DataFrames with the following schema:

```
root
 |-- label: string (nullable = false)
 |-- category: string (nullable = false)
 |-- entityIds: array (nullable = false)
 |    |-- element: string (containsNull = false)
 |-- datasourceId: string (nullable = true)
```

**Fields:**
- `label`: The facet label (e.g., gene symbol, GO term name, pathway name)
- `category`: The facet category (e.g., "Approved Symbol", "GO:BP", "Tractability Small Molecule")
- `entityIds`: Array of Ensembl gene IDs associated with this facet
- `datasourceId`: Optional ID referencing the data source (used for GO, pathways, etc.)

## Architecture

### Class Structure

```
FacetSearchCategories
├─ Holds category name configuration
└─ Can be customized via dictionary

Helper Functions (helpers.py)
├─ compute_simple_facet()
│  └─ Creates basic facets from a single field
└─ get_relevant_dataset()
   └─ Filters and selects relevant columns

Facet Computation Functions
├─ compute_tractability_facets()
├─ compute_target_id_facets()
├─ compute_approved_symbol_facets()
├─ compute_approved_name_facets()
├─ compute_subcellular_locations_facets()
├─ compute_target_class_facets()
├─ compute_pathways_facets()
└─ compute_go_facets()

Main Entry Point
└─ target_facets()
   └─ Orchestrates all facet computation
```

### Processing Flow

1. **Input**: Targets DataFrame + GO Reference DataFrame
2. **Extraction**: Extract relevant fields for each facet type
3. **Transformation**: 
   - Explode nested arrays
   - Join with reference data (for GO)
   - Map codes to human-readable names
4. **Aggregation**: Group by facet values, collect entity IDs
5. **Output**: Union all facets into single DataFrame

## Dependencies

All required dependencies are already in `pyproject.toml`:
- `pyspark>=4.0.0` - PySpark for distributed processing
- `loguru==0.7.3` - Logging
- `smart-open[gcs]>=7.0.0` - GCS support

## Configuration Mapping

The module expects input data following this structure (matching the original Scala config):

```yaml
source:
  targets: ${common.path}/output/targets
  go: ${steps.go.output.go}

destination:
  targets: ${common.output_path}/view/search_facet_target
```

Python equivalent:

```python
source = {
    'targets': f'{common_path}/output/targets',
    'go': f'{go_output_path}'
}

destination = {
    'targets': f'{output_path}/view/search_facet_target'
}
```


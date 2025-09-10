# PTS Utils

This directory contains utility modules for the PTS (Pipeline Transformation Stage) project.

## Ontology Module

The `ontology.py` module provides functionality for mapping disease information to EFO (Experimental Factor Ontology) using the OnToma library.

### Usage

```python
from pts.utils.ontology import add_efo_mapping
from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("OntologyMapping").getOrCreate()

# Your evidence strings DataFrame with diseaseFromSource and diseaseFromSourceId columns
evidence_df = spark.createDataFrame([
    ("Alzheimer's disease", "MONDO:0004975"),
    ("Type 2 diabetes", "MONDO:0005148"),
    # ... more rows
], ["diseaseFromSource", "diseaseFromSourceId"])

# Add EFO mappings
mapped_df = add_efo_mapping(
    evidence_strings=evidence_df,
    spark_instance=spark,
    ontoma_cache_dir="/path/to/cache",  # Optional
    efo_version="latest"  # Optional, defaults to latest
)

# The resulting DataFrame will have an additional 'diseaseFromSourceMappedId' column
# with EFO IDs where mappings were found
```

### Dependencies

The ontology module requires the following dependencies:
- `numpy>=1.24.0`
- `ontoma>=1.0.0`
- `pandarallel>=1.6.0`
- `pyspark>=4.0.0`

These are automatically included when installing the PTS package.

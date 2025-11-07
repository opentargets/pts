# Column Order Fix Summary

## Issue
When using UNION on multiple DataFrames, PySpark requires all DataFrames to have columns in the exact same order and types.

The facet schema requires: `label, category, entityIds, datasourceId`

However, after `groupBy()` and `agg()`, Spark orders columns as:
- Columns from groupBy (in groupBy order)  
- Aggregated columns
- Columns added via withColumn

## Fixed Functions

### ✅ 1. `compute_tractability_facets` (Line 125)
**Issue**: groupBy('category', 'label') → order was `category, label, entityIds, datasourceId`
**Fix**: Added `.select('label', 'category', 'entityIds', 'datasourceId')` before `.distinct()`

### ✅ 2. `compute_target_class_facets` (Line 277)
**Issue**: groupBy('label', 'category') but withColumn added datasourceId at end
**Fix**: Added `.select('label', 'category', 'entityIds', 'datasourceId')` before `.distinct()`

### ✅ 3. `compute_subcellular_locations_facets` (Line 231)
**Already Fixed**: Had `.select('label', 'category', 'entityIds', 'datasourceId')`

### ✅ 4. `compute_pathways_facets` (Line 322)
**Already Fixed**: Had `.select('label', 'category', 'entityIds', 'datasourceId')`

### ✅ 5. `compute_go_facets` (Line 398)
**Issue**: groupBy('label', 'category', 'datasourceId') → order was `label, category, datasourceId, entityIds`
**Fix**: Added `.select('label', 'category', 'entityIds', 'datasourceId')` before `.distinct()`

### ✅ 6-8. Simple Facets (target_id, approved_symbol, approved_name)
**OK**: All use `compute_simple_facet()` helper which already has correct column order

## Solution Pattern

All facet functions now explicitly select columns in the correct order before union:

```python
.select('label', 'category', 'entityIds', 'datasourceId')
.distinct()
```

This ensures consistent schema across all 8 facet types!

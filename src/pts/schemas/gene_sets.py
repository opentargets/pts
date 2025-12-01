from pyspark.sql.types import ArrayType, StringType, StructField, StructType

gene_sets_schema = StructType([
    StructField('label', StringType(), nullable=False),
    StructField('category', StringType(), nullable=False),
    StructField('entityIds', ArrayType(StringType(), containsNull=False), nullable=False),
    StructField('datasourceId', StringType(), nullable=True),
    StructField('parentId', ArrayType(StringType(), containsNull=False), nullable=True),
])

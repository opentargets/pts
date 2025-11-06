from pyspark.sql.types import ArrayType, StringType, StructField, StructType

facet_schema = StructType([
    StructField('label', StringType(), nullable=False),
    StructField('category', StringType(), nullable=False),
    StructField('entityIds', ArrayType(StringType(), containsNull=False), nullable=False),
    StructField('datasourceId', StringType(), nullable=True),
])

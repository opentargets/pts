from pyspark.sql.types import ArrayType, StringType, StructField, StructType

reactome_schema = StructType([
    StructField('id', StringType(), False),
    StructField('label', StringType(), True),
    StructField('ancestors', ArrayType(StringType(), containsNull=False), True),
    StructField('descendants', ArrayType(StringType(), containsNull=False), True),
    StructField('children', ArrayType(StringType(), containsNull=False), True),
    StructField('parents', ArrayType(StringType(), containsNull=False), True),
    StructField('path', ArrayType(StringType(), containsNull=False), True),
])

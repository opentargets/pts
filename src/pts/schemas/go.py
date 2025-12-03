from pyspark.sql.types import ArrayType, BooleanType, StringType, StructField, StructType

go_schema = StructType([
    StructField('id', StringType(), False),
    StructField('label', StringType(), True),
    StructField('namespace', StringType(), True),
    StructField('alt_ids', ArrayType(StringType(), containsNull=False), True),
    StructField('is_a', ArrayType(StringType(), containsNull=False), True),
    StructField('part_of', ArrayType(StringType(), containsNull=False), True),
    StructField('regulates', ArrayType(StringType(), containsNull=False), True),
    StructField('negatively_regulates', ArrayType(StringType(), containsNull=False), True),
    StructField('positively_regulates', ArrayType(StringType(), containsNull=False), True),
    StructField('isObsolete', BooleanType(), True),
])

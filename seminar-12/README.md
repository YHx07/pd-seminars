# Seminar 12

# PySpark. DataFrame

..in progress

### Прочитать данные

Источники данных: CSV, JSON, Hive, HBase, Cassandra,  MySQL, PostgreSQL, Parquet, ORC, Kafka, ElasticSearch, Amazon S3, ...

```python
df = spark.read.format('csv').option('sep', '|').load('/lectures/lecture02/data/ml-100k/u.user')
df
# DataFrame[_c0: string, _c1: string, _c2: string, _c3: string, _c4: string]
df.show(4)
```

```plain
+---+---+---+----------+-----+
|_c0|_c1|_c2|       _c3|  _c4|
+---+---+---+----------+-----+
|  1| 24|  M|technician|85711|
|  2| 53|  F|     other|94043|
|  3| 23|  M|    writer|32067|
|  4| 24|  M|technician|43537|
+---+---+---+----------+-----+
only showing top 4 rows
```

### Schema

Чтобы задать тип колонок и названия воспользуемся schema

```python
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

schema = StructType(fields=[
    StructField('user_id', IntegerType()),
    StructField('age', IntegerType()),
    StructField('gender', StringType()),
    StructField('occupation', StringType()),
    StructField('zip', IntegerType()),
])

df = spark.read.schema(schema).format('csv').option('sep', '|').load('/lectures/lecture02/data/ml-100k/u.user')
df
# DataFrame[user_id: int, age: int, gender: string, occupation: string, zip: int]
df.printSchema()
```

```plain
root
 |-- user_id: integer (nullable = true)
 |-- age: integer (nullable = true)
 |-- gender: string (nullable = true)
 |-- occupation: string (nullable = true)
 |-- zip: integer (nullable = true)
```

```python
df.show(4)
```

```plain
+-------+---+------+----------+-----+
|user_id|age|gender|occupation|  zip|
+-------+---+------+----------+-----+
|      1| 24|     M|technician|85711|
|      2| 53|     F|     other|94043|
|      3| 23|     M|    writer|32067|
|      4| 24|     M|technician|43537|
+-------+---+------+----------+-----+
only showing top 4 rows
```

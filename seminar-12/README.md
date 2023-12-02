# Seminar 12

# Как сдавать дз?

Точка входа в программу -- файл `run.sh`. В нём команда: `spark-submit main.py` (или другое развание файла).

В `main.py` начать со строк:

```python
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

config = SparkConf().setAppName("spark_graph").setMaster("yarn")
sc = SparkContext(conf=config)

spark = SparkSession.builder.appName('Spark DF practice').master('yarn').getOrCreate()
```

Эти команды создают Spark сессию. В [прошлом семинаре](https://github.com/YHx07/pd-seminars/blob/main/seminar-11/README.md) Spark сессия создавалась при выполнении команды `PYSPARK_DRIVER_PYTHON=jupyter PYSPARK_PYTHON=/usr/bin/python3 PYSPARK_DRIVER_PYTHON_OPTS='notebook --ip="*" --port=<PORT> --no-browser' pyspark2 --master=yarn --num-executors=<N>` -- вместе с запуском Jupyter Notebook поднималась Spark сессия.

В config `SparkConf().setAppName("spark_graph").setMaster("yarn")` есть установка `setMaster`, в данном случае написан `yarn` -- эта установка для запуска программы на Spark на клестере. Есть еще `local[1]` и прочие -- это для локального запуска.

# PySpark. DataFrame

..in progress

### Прочитать данные

Источники данных: CSV, JSON, Hive, HBase, Cassandra,  MySQL, PostgreSQL, Parquet, ORC, Kafka, ElasticSearch, Amazon S3, ...

```python
import pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("yarn") \
                    .appName('SparkByExamples.com') \
                    .getOrCreate()

spark
```

```plain
SparkSession - in-memory

SparkContext

Spark UI

Version
v2.4.0.cloudera2
Master
yarn
AppName
SparkByExamples.com
```

```
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

```
from pyspark.sql import functions as F

df.select(
    F.col("_c0").alias("destination"),
    F.col("_c1").alias("source")
).show(1)
```

```plain
+-----------+------+
|destination|source|
+-----------+------+
|         12|  2241|
+-----------+------+
only showing top 1 row
```

```
df.toPandas()
```

<img width="292" alt="Screenshot 2023-12-03 at 01 30 56" src="https://github.com/YHx07/pd-seminars/assets/36137274/c06a9bd2-d957-4b6a-badb-7602f670e920">

```
df.groupBy('source').count().show()
```

```plain
+--------+-----+
|  source|count|
+--------+-----+
|19593065|    1|
|20651832|    1|
|21215059|    1|
|21240863|    1|
|21705463|    1|
|22197844|    1|
|22377590|    1|
|22935113|    1|
|23993819|    1|
|24268091|    1|
|24299946|    1|
|24326074|    1|
|25053014|    1|
|26715269|    1|
|27121291|    1|
|27385940|    1|
|27628353|    1|
|28817312|    1|
|29847995|    1|
|30290044|    1|
+--------+-----+
only showing top 20 rows
```

```
df.groupBy('source').count().orderBy(F.col('count').desc()).show()
```

```plain
+--------+-----+
|  source|count|
+--------+-----+
|      53|    7|
| 9598762|    4|
|15458708|    4|
|13342022|    4|
|      20|    4|
|19489341|    4|
|      12|    4|
|14206015|    3|
|26468557|    3|
|14287820|    3|
|     107|    3|
|52041136|    3|
|17184081|    3|
|19788155|    3|
|21494147|    3|
|18662758|    3|
|18234522|    3|
|16227030|    3|
|47516482|    3|
|      23|    3|
+--------+-----+
only showing top 20 rows
```

```
from pyspark.sql.functions import sum,avg,max,count

df1 = df1.groupBy("source").agg(
    count("*").alias("count1"), 
    avg("source").alias("mean")
)
df1.show()
```

```plain
+--------+------+-----------+
|  source|count1|       mean|
+--------+------+-----------+
|19593065|     1|1.9593065E7|
|20651832|     1|2.0651832E7|
|21215059|     1|2.1215059E7|
|21240863|     1|2.1240863E7|
|21705463|     1|2.1705463E7|
|22197844|     1|2.2197844E7|
|22377590|     1| 2.237759E7|
|22935113|     1|2.2935113E7|
|23993819|     1|2.3993819E7|
|24268091|     1|2.4268091E7|
|24299946|     1|2.4299946E7|
|24326074|     1|2.4326074E7|
|25053014|     1|2.5053014E7|
|26715269|     1|2.6715269E7|
|27121291|     1|2.7121291E7|
|27385940|     1| 2.738594E7|
|27628353|     1|2.7628353E7|
|28817312|     1|2.8817312E7|
|29847995|     1|2.9847995E7|
|30290044|     1|3.0290044E7|
+--------+------+-----------+
only showing top 20 rows
```

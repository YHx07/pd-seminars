Библиотека работает с двумя типами векторов: 
- DenseVecotor — плотный вектор набор всех значений
- SparseVector — сжатый вектор с ненулевыми значениями.

`from pyspark.ml.linalg import DenseVector, SparseVector`

- DenseVector
    
    ```python
    v = DenseVector([1, 2, 3, 4])
    
    v.values
    
    type(v.values)
    
    v.toArray()
    # array([1., 2., 3., 4.])
    ```
    
    Вектора индексированные:
    
    ```python
    v[0]
    v[-1]
    v[2:4]
    ```
    
    Можно совершать операции:
    
    ```python
    v - 2
    v / 3
    ```
    
    Менять значения
    
    ```python
    v.values[0] = 0
    ```
    
    L1 норма
    
    ```python
    v.norm(1)
    ```
    
    L2 норма
    
    ```python
    v.norm(2)
    ```
    
    Можно конструировать через обертку `Vectors`:
    
    ```python
    from pyspark.ml.linalg import Vectors
    
    u = Vectors.dense([1, 2, 3, 5])
    
    u - v
    
    v.squared_distance(u)
    ```
    
    Косинуская мера близости:
    
    ```python
    v.dot(u) / (v.norm(2) * u.norm(2))
    ```

- Sparse vectors

- arse vectors
    
    ```python
    from pyspark.ml.linalg import Vectors
    
    ndx_value = tuple(zip(range(4), range(1,5)))
    
    ndx_value
    # ((0, 1), (1, 2), (2, 3), (3, 4))
    
    v = SparseVector(len(ndx_value), ndx_value)
    v
    # SparseVector(4, {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0})
    ```
    
    Можно преобразовать к `DenseVector`:
    
    ```python
    DenseVector(v)
    # DenseVector([1.0, 2.0, 3.0, 4.0])
    ```
    
    Одинаковые конструкции: враппер или конструкто
    
    ```python
    u = Vectors.sparse(4, range(4), [1, 2, 3, 5])
    u = SparseVector(4, range(4), [1, 2, 3, 5]) 
    
    u
    # SparseVector(4, {0: 1.0, 1: 2.0, 2: 3.0, 3: 5.0})
    
    v * 2
    # TypeError: -- SparseVector не может умножаться
    
    v - u
    # TypeError
    
    v.squared_distance(u)
    # 1.0
    
    v.dot(u) / (v.norm(2) * u.norm(2))
    # 0.9939
    ```
    
    Можно задать матрицу:

    ```
    v2 = DenseVector([[1,2], [2,3], [3,4]])
    ```

# Pipeline

Pipeline состоит из двух операций: Transformer и Estimator.

- DataFrame: ML API использует DataFrame из Spark SQL как dataset, в котором могут храниться признакми, метки классов, предсказания
- Transformer: Transformer — это алгоритм, который может преобразовать один DataFrame в другой DataFrame. Например, ML model это Transformer, который преобразует DataFrame с признаками в другой DataFrame с предсказаниями.
- Estimator: Estimator — это алгоритм, который может быть оубчен на DataFrame, чтобы создать Transformer. Например, алгоритм обучения — это Estimator, который обучается на DataFrame и создает модель.
- Pipeline: Pipeline соединяет в цепочку несколько Transformer и Estimator вместе, чтоюы задать ML workflow.
- Parameter: Все Transformer и Estimator теперь имеют общее API для спецификации параметров.

- Построение собственного Transformer-а
    
    ```python
    class ConstTransformer(Transformer):
    	""" 
    	Constant transformer.
    	"""
    	def _transform(self, dataset):
    		return dataset.withColumn("col", f.lit("col"))
    
    df = spark.range(0, 3, numPartition=1)
    transformer = ConstTransformer()
    transformer.transform(df).show()
    +---+------+
    | id|   col|
    +---+------+
    |  0| "col"|
    |  1| "col"|
    |  2| "col"|
    +---+------+
    ```
    
    Как специфицируются параметры transformer'a?
    
    ```python
    from pyspark.ml.param.shared import HasOutputCol
    
    class ConstTransformer(Transformer, HasOutputCol):
    	""" Constant transformer with variable name. """
    	def __init__(self):
    		super(ConstTransformer, self).__init__()
    
    	def _transform(self, dataset):
    		return dataset.withColumn(self.fetOutputCol(), f.lit('col'))
    
    transformer = ConstTransformer()
    transformer.transform(df).show() # выдаст колонку с похим названием
    
    # Но тут плохо то что параметр задаем неявно
    transformer.setOutputCol("col_name")
    transformer.transform(df).show() # будет хорошее название
    ```
    
    ```python
    class ConstTransformer(Transformer, HasOutputCol):
    	""" Constant transformer with variable name. """
    	@keyword_only # чтобы параметры можно было записать только явно
    	def __init__(self, outpitCol=None):
    		super(ConstTransformer, self).__init__()
    		if outputCol is not None:
    			self.setOutputCol(outputCol)
    
    ****	def _transform(self, dataset):
    		return dataset.withColumn(self.fetOutputCol(), f.lit('col'))
    
    transformer = ConstTransformer(outputCol="col_name")
    ```
    
    Transformer с заданными input и output колонками
    
    ```python
    from pyspark.ml.param.shared import HasInputCol
    
    class HashTransformer(Transformer, HasInputCol, HasOutputCol):
    	""" Constant transformer with variable name. """
    	@keyword_only
    	def __init__(self, inputCol=None, outpitCol=None):
    		super(HashTransformer, self).__init__()
    		if inputCol is not None:
    			self.setOutputCol(inputCol)
    		if outputCol is not None:
    			self.setOutputCol(outputCol)
    
    ****	def _transform(self, dataset):
    		return dataset.withColumn(self.fetOutputCol(), 
    															f.md5(f.col(self.getInputCol()).cast('string')))
    
    transformer = HashTransformer(inputCol="hash", outputCol="hash")
    ```
    
    Кастомный параметр [?????]
    
    ```python
    from pyspark.ml.param import Param, Params, TypeConverters
    
    class HashTransformer(Transformer, HasInputCol, HasOutputCol):
        
        algorithm = Param(Params._dummy(), "algorithm",
                          "hash function to use, must be one of (md5|sha1)",
                          typeConverter=TypeConverters.toString)
        
        @keyword_only
        def __init__(self, inputCol=None, outputCol=None, algorithm="md5"):
            super(HashTransformer, self).__init__()
            if inputCol is not None:
                self.setInputCol(inputCol)
            if outputCol is not None:
                self.setOutputCol(outputCol)
            self._set(algorithm=algorithm)
            
        def get_hash_function(self):
            try:
                return getattr(f, self.getOrDefault("algorithm"))
            except AttributeError as e:
                raise ValueError("Unsupported algorithm {}".format(self.getOrDefault("algorithm")))
                
        def setAlgorithm(self, algorithm):
            self._set(algorithm=algorithm)
                
        def _transform(self, dataset):
            hash_col_func = self.get_hash_function()
            return dataset.withColumn(self.getOutputCol(), hash_col_func(f.col(self.getInputCol()).cast("string")))
    
    transformer = HashTransformer(inputCol="id", outputCol="hash", algorithm="lalalal")
    ```
    
    Передадим параметр из одного Transformer'а в другой
    
    ```python
    class HashTransformer(Transformer, HasInputCol, HasOutputCol):
        
        algorithm = Param(Params._dummy(), "algorithm",
                          "hash function to use, must be one of (md5|sha1)",
                          typeConverter=TypeConverters.toString)
        
        @keyword_only
        def __init__(self, inputCol=None, outputCol=None, algorithm="md5"):
            super(HashTransformer, self).__init__()
            if inputCol is not None:
                self.setInputCol(inputCol)
            if outputCol is not None:
                self.setOutputCol(outputCol)
            self._set(algorithm=algorithm)
            
        def get_hash_function(self):
            try:
                return getattr(f, self.getOrDefault("algorithm"))
            except AttributeError as e:
                raise ValueError("Unsupported algorithm {}".format(self.getOrDefault("algorithm")))
                
        def setAlgorithm(self, algorithm):
            self._set(algorithm=algorithm)
        
        def getAlgorithm(self):
            return self.getOrDefault("algorithm")
    
        def _transform(self, dataset):
            hash_col = self.get_hash_function()
            res = dataset.withColumn(self.getOutputCol(), hash_col(f.col(self.getInputCol()).cast("string")))
            return res
    
    transformer1 = HashTransformer(inputCol="id", outputCol="hash1", algorithm="sha1")
    transformer2 = HashTransformer(inputCol="hash1", outputCol="hash2", algorithm=transformer1.getAlgorithm())
    ```
    
    ```python
    # или так:
    from pyspark.ml import Pipeline
    
    pipeline = Pipeline(stages=[
        transformer1, transformer2
    ])
    
    pipeline_model = pipeline.fit(df)
    ```
    
- Построение собственного Estimator-а
    
    ```python
    
    ```

## Модели машинного обучения

```python

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
X, y = make_classification(random_state=5757)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5757)
```

- Логистическая регрессия
    
    ```python
    from pyspark.ml.classification import LogisticRegression
    
    from pyspark.sql.types import StructType, StructField, DoubleType
    from pyspark.ml.linalg import VectorUDT
    
    schema = StructType(fields=[
        StructField("label", DoubleType()),
        StructField("features", VectorUDT()),
    ])
    
    # первое значение -- label
    # второе значение -- вектор фичей
    training_set = spark.createDataFrame([
        (1.0, Vectors.dense([0.0, 1.1, 0.1])),
        (0.0, Vectors.dense([2.0, 1.0, -1.0])),
        (0.0, Vectors.dense([2.0, 1.3, 1.0])),
        (1.0, Vectors.dense([0.0, 1.2, -0.5]))], schema=schema)
    
    lr = LogisticRegression(maxIter=10, regParam=0.01)
    
    lr.params
    ```
    
    - Вывод с параметрами модели
        
        ```
        [Param(parent='LogisticRegression_686ace36c5c3', name='aggregationDepth', doc='suggested depth for treeAggregate (>= 2).'),
         Param(parent='LogisticRegression_686ace36c5c3', name='elasticNetParam', doc='the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.'),
         Param(parent='LogisticRegression_686ace36c5c3', name='family', doc='The name of family which is a description of the label distribution to be used in the model. Supported options: auto, binomial, multinomial'),
         Param(parent='LogisticRegression_686ace36c5c3', name='featuresCol', doc='features column name.'),
         Param(parent='LogisticRegression_686ace36c5c3', name='fitIntercept', doc='whether to fit an intercept term.'),
         Param(parent='LogisticRegression_686ace36c5c3', name='labelCol', doc='label column name.'),
         Param(parent='LogisticRegression_686ace36c5c3', name='lowerBoundsOnCoefficients', doc='The lower bounds on coefficients if fitting under bound constrained optimization. The bound matrix must be compatible with the shape (1, number of features) for binomial regression, or (number of classes, number of features) for multinomial regression.'),
         Param(parent='LogisticRegression_686ace36c5c3', name='lowerBoundsOnIntercepts', doc='The lower bounds on intercepts if fitting under bound constrained optimization. The bounds vector size must beequal with 1 for binomial regression, or the number oflasses for multinomial regression.'),
         Param(parent='LogisticRegression_686ace36c5c3', name='maxIter', doc='max number of iterations (>= 0).'),
         Param(parent='LogisticRegression_686ace36c5c3', name='predictionCol', doc='prediction column name.'),
         Param(parent='LogisticRegression_686ace36c5c3', name='probabilityCol', doc='Column name for predicted class conditional probabilities. Note: Not all models output well-calibrated probability estimates! These probabilities should be treated as confidences, not precise probabilities.'),
         Param(parent='LogisticRegression_686ace36c5c3', name='rawPredictionCol', doc='raw prediction (a.k.a. confidence) column name.'),
         Param(parent='LogisticRegression_686ace36c5c3', name='regParam', doc='regularization parameter (>= 0).'),
         Param(parent='LogisticRegression_686ace36c5c3', name='standardization', doc='whether to standardize the training features before fitting the model.'),
         Param(parent='LogisticRegression_686ace36c5c3', name='threshold', doc='Threshold in binary classification prediction, in range [0, 1]. If threshold and thresholds are both set, they must match.e.g. if threshold is p, then thresholds must be equal to [1-p, p].'),
         Param(parent='LogisticRegression_686ace36c5c3', name='thresholds', doc="Thresholds in multi-class classification to adjust the probability of predicting each class. Array must have length equal to the number of classes, with values > 0, excepting that at most one value may be 0. The class with largest value p/t is predicted, where p is the original probability of that class and t is the class's threshold."),
         Param(parent='LogisticRegression_686ace36c5c3', name='tol', doc='the convergence tolerance for iterative algorithms (>= 0).'),
         Param(parent='LogisticRegression_686ace36c5c3', name='upperBoundsOnCoefficients', doc='The upper bounds on coefficients if fitting under bound constrained optimization. The bound matrix must be compatible with the shape (1, number of features) for binomial regression, or (number of classes, number of features) for multinomial regression.'),
         Param(parent='LogisticRegression_686ace36c5c3', name='upperBoundsOnIntercepts', doc='The upper bounds on intercepts if fitting under bound constrained optimization. The bound vector size must be equal with 1 for binomial regression, or the number of classes for multinomial regression.'),
         Param(parent='LogisticRegression_686ace36c5c3', name='weightCol', doc='weight column name. If this is not set or empty, we treat all instance weights as 1.0.')]
        ```
        
    
    Регуляризационный параметр:
    
    ```python
    lr.getOrDefault("regParam")
    ```
    
    ```python
    lr.getOrDefault("standardization")
    # True -- стандартизация включена
    ```
    
    Обучим модель
    
    ```python
    model = lr.fit(training_set)
    type(model)
    # pyspark.ml.classification.LogisticRegressionModel
    
    model.coefficients
    # DenseVector([-3.1009, 2.6082, -0.3802])
    
    model.interceptVector
    # DenseVector([0.0682])
    
    # Предсказания:
    predict = model.transform(training_set)
    predict.printSchema()
    # Датафрейм немного изменился:
    root
     |-- label: double (nullable = true)
     |-- features: vector (nullable = true)
     |-- rawPrediction: vector (nullable = true)
     |-- probability: vector (nullable = true)
     |-- prediction: double (nullable = false)
    
    training_set.show(1)
    '''
    +-----+-------------+
    |label|     features|
    +-----+-------------+
    |  1.0|[0.0,1.1,0.1]|
    +-----+-------------+
    only showing top 1 row
    '''
    
    predict.show(1, truncate=False, vertical=True)
    '''
    -RECORD 0-------------------------------------------------
     label         | 1.0                                      
     features      | [0.0,1.1,0.1]                            
     rawPrediction | [-2.8991948946380375,2.8991948946380375] 
     probability   | [0.0521933766630071,0.947806623336993]   
     prediction    | 1.0                                      
    only showing top 1 row
    '''
    
    model.getOrDefault("threshold")
    # 0.5
    ```
    
- Классификация токсичных коментариев (логистическая регрессия)
    
    ```python
    from pyspark.sql.types import StringType, IntegerType
    
    schema = StructType([
        StructField("id", StringType()),
        StructField("comment_text", StringType()),
        StructField("toxic", IntegerType()),
        StructField("severe_toxic", IntegerType()),
        StructField("obscene", IntegerType()),
        StructField("threat", IntegerType()),
        StructField("insult", IntegerType()),
        StructField("identity_hate", IntegerType())
    ])
    
    ! hdfs dfs -head /lectures/lecture03/data/train.csv
    
    # Был баг с multiline CSVs, fix в 2.2.0 
    # https://issues.apache.org/jira/browse/SPARK-19610
    dataset = spark.read.csv("/lectures/lecture03/data/train.csv",
                             schema=schema, header=True, multiLine=True, escape='"')
    
    dataset.show(2, vertical=True)
    ```
    
    Определим бинарный target (toxic/non-toxic)
    
    ```python
    from pyspark.sql import functions as f
    
    target = f.when(
        (dataset.toxic == 0) &
        (dataset.severe_toxic == 0) &
        (dataset.obscene == 0) &
        (dataset.threat == 0) &
        (dataset.insult == 0) &
        (dataset.identity_hate == 0),
        0
    ).otherwise(1)
    
    dataset = dataset.withColumn("target", target)
    
    dataset.select("id", "target").show(5)
    '''
    +----------------+------+
    |              id|target|
    +----------------+------+
    |b63771a6dd0bc63c|     0|
    |844520719ae75ca9|     1|
    |1b3a6936a549fb69|     0|
    |6003fc22945ddfb6|     0|
    |63d49f7a1ffb31e8|     0|
    +----------------+------+
    '''
    
    dataset.groupBy("target").count().collect()
    # [Row(target=1, count=16225), Row(target=0, count=143346)]
    
    dataset = dataset.drop("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate").cache()
    ```
    
    Обучим самую простую binary logistic regression
    
    ```python
    from pyspark.ml.feature import Tokenizer, HashingTF
    
    # Преобразуем комметарии в слова
    tokenizer = Tokenizer(inputCol="comment_text", outputCol="words")
    dataset2 = tokenizer.transform(dataset)
    dataset2
    # DataFrame[id: string, comment_text: string, target: int, words: array<string>]
    
    dataset2.take(1)
    '''
    [Row(id='26e1b63617df36b1', comment_text='"\n\n charlie wilson \n\ni didnt notice 
    the music genres that were reverted. However my intention was to revert his alias 
    that you deleted.  His alias a.k.a is actually ""Uncle Charlie"" and needs to be 
    put back and shouldn\'t  have been removed."', target=0, words=['"', '', '', 'charlie',
     'wilson', '', '', 'i', 'didnt', 'notice', 'the', 'music', 'genres', 'that', 'were', 
    'reverted.', 'however', 'my', 'intention', 'was', 'to', 'revert', 'his', 'alias', '
    that', 'you', 'deleted.', '', 'his', 'alias', 'a.k.a', 'is', 'actually', '""uncle', 
    'charlie""', 'and', 'needs', 'to', 'be', 'put', 'back', 'and', "shouldn't", '', 
    'have', 'been', 'removed."'])]
    '''
    
    type(dataset2.take(1))
    # list
    type(dataset2.take(1)[0])
    # pyspark.sql.types.Row
    type(dataset2.take(1)[0].words)
    # list
    ```
    
- Hashing trick vs CountVectorizer
    
    ```python
    df = spark.createDataFrame([
        (0, "PYTHON HIVE HIVE".split(" ")),
        (1, "JAVA JAVA SQL".split(" "))
    ], ["id", "words"])
    df.show(truncate = False)
    '''
    +---+--------------------+
    |id |words               |
    +---+--------------------+
    |0  |[PYTHON, HIVE, HIVE]|
    |1  |[JAVA, JAVA, SQL]   |
    +---+--------------------+
    '''
    from pyspark.ml.feature import CountVectorizer
    cv = CountVectorizer(inputCol="words", outputCol="features")
    model = cv.fit(df)
    result = model.transform(df)
    result.show(truncate=False)
    '''
    +---+--------------------+-------------------+
    |id |words               |features           |
    +---+--------------------+-------------------+
    |0  |[PYTHON, HIVE, HIVE]|(4,[0,3],[2.0,1.0])|
    |1  |[JAVA, JAVA, SQL]   |(4,[1,2],[2.0,1.0])|
    +---+--------------------+-------------------+
    '''
    from pyspark.ml.feature import HashingTF
    ht = HashingTF(inputCol="words", outputCol="features", numFeatures=4)
    result = ht.transform(df)
    result.show(truncate=False)
    '''
    +---+--------------------+-------------------+
    |id |words               |features           |
    +---+--------------------+-------------------+
    |0  |[PYTHON, HIVE, HIVE]|(4,[0],[3.0])      |
    |1  |[JAVA, JAVA, SQL]   |(4,[2,3],[1.0,2.0])|
    +---+--------------------+-------------------+
    '''
    from pyspark.ml.feature import HashingTF
    ht = HashingTF(inputCol="words", outputCol="features", numFeatures=10)
    result = ht.transform(df)
    result.show(truncate=False)
    '''
    +---+--------------------+--------------------+
    |id |words               |features            |
    +---+--------------------+--------------------+
    |0  |[PYTHON, HIVE, HIVE]|(10,[0,2],[2.0,1.0])|
    |1  |[JAVA, JAVA, SQL]   |(10,[2,9],[1.0,2.0])|
    +---+--------------------+--------------------+
    '''
    ```


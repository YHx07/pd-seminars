# YARN UI

Зададим количество экзекьюторов:
```bash
--num-executors 2
```

Запустим код:

```python
vocabulary = ('Apache', 'Spark', 'Hadoop')
numbers = np.random.randint(10, size=10000)
words = np.random.choice(vocabulary, size=10000)
collection = zip(numbers, words)

rdd = spark.parallelize(collection)

rdd.count()
```

![image](https://user-images.githubusercontent.com/36137274/205499704-031c022c-1456-499b-a190-8fd185a1cd7c.png)
Было 2 экзекьютора, поэтому создалось 2 задачи

![image](https://user-images.githubusercontent.com/36137274/205499729-690c0641-d533-45cd-be5d-79de73f10a23.png)

Метрики:

- Locallity Level — где происходят вычисления
- Host — какие сервера были задействованы
- Duration — время

![image](https://user-images.githubusercontent.com/36137274/205499761-2a558602-e697-4e25-9e98-e268cefc6ddf.png)

Task Time (GC Time) == Garbage collection time. Время которое тратится на сборку мусора. Если это порядка 15%, то число подсветится красным. Это значит, что выделено мало памяти на экзекьютор

Партиции в PySpark — это таски. То на сколько tasks разибивается задача.

Будем запускать по шагам и смотреть на YARN UI, если в UI что-то отображается, то заносим в таблицу, если нет, то "-":

<table>
<tr>
  <td> ## </td> <td> Code </td> <td> YARN UI </td>
  </tr>
  <tr>
  <td> 1 </td>
  <td>

  ```python
  raw_logs = spark.textFile('/lectures/lecture01/data/log.txt')
  ```

  </td>
  <td> - </td>
</tr>
<tr>
  <td> 2 </td>
  <td>

  ```python
  raw_logs.take(5)
  ```

  </td>
  <td> <img src="https://user-images.githubusercontent.com/36137274/205500220-138b6df0-2078-47af-912f-f1a64602127d.png" height = 20px > </td>
</tr>
<tr>
  <td> 3 </td>
  <td>
  Вывод:
    
  ```bash
  ['192.168.0.10\tERROR\tWhen production fails in dipsair, whom you gonna call?',
 '192.168.0.39\tINFO\tJust an info message passing by',
 '192.168.0.35\tINFO\tJust an info message passing by',
 '192.168.0.19\tINFO\tJust an info message passing by',
 '192.168.0.23\tERROR\tWhen production fails in dipsair, whom you gonna call?']
  ```

  </td>
  <td> - </td>
</tr>
<tr>
  <td> 4 </td>
  <td>

  ```python
  raw_logs.getNumPartitions()
  ```

  </td>
  <td> - </td>
</tr>
<tr>
  <td> 5 </td>
  <td>

  ```python
  logs = raw_logs.map(lambda x: x.split('\t'))
  ```

  </td>
  <td> - </td>
</tr>
<tr>
  <td> 6 </td>
  <td>

  ```python
  logs.take(5)
  ```
    
  </td>
  <td> <img src="https://user-images.githubusercontent.com/36137274/205501107-e8b9ad48-94ca-4267-a6b1-216849352955.png" height = 20px > </td>
</tr>
<tr>
  <td> 7 </td>
  <td>

  ```python
  logs.flatMap(lambda x: x[2].split()).take(20)
  ```
    
  </td>
  <td> <img src="https://user-images.githubusercontent.com/36137274/205501197-5adde150-cc29-41a4-9e46-6aa955e2e48f.png" height = 20px > </td>
</tr>
<tr>
  <td> 8 </td>
  <td>

  ```python
  words = logs.flatMap(lambda x: x[2].split())
  words.groupBy(lambda x: x).count()
  ```

  </td>
  <td> <img src="https://user-images.githubusercontent.com/36137274/205501228-d042047e-c1eb-4d59-a028-b8debd652635.png" height = 20px > </td>
</tr>
</table>

Что происходило:

Последовательность трансформаций определеяет граф вычислений (DAG). В нем есть партиции и зависимости между партициями. Таким образом Spark имеет всю необходимую информацию для вычисления графа в любой точке и возможных оптимизаций.

![image](https://user-images.githubusercontent.com/36137274/205501283-1f200948-d9cf-48ad-b3d4-8c4ce09da835.png)

В ReduceByKey данные с одним ключа локализуются на одном экзекьютор. Наличие такого DAG позволяет в случае падения какого-то Stage откатиться назад по DAGу и пеерсчитать партицию заново.

Трансформации бывают:

<table>
<tr>
  <td> Узкие </td>
  <td> Широкие </td>
</tr>
<tr>
  <td> <img src="https://user-images.githubusercontent.com/36137274/205501365-f7578174-90f4-4141-a436-615191a4facc.png" height = 300px > 
  Обычные трансформации
  </td>
  <td> <img src="https://user-images.githubusercontent.com/36137274/205501404-c02a683b-28de-4fc8-a718-baee2271baba.png" height = 300px > 
  Возникают, когда экзекьюторы начинают обмениваться данными — это операция shuffle
  </td>
</tr>
</table>

Широкие трансформации разделяют Job на Stage-ы. Между Stage происходит shuffle данных, которого надо избегать. Узкие трансформации объединяются в stage. Stage разделены широкими трансформациями.

### Summary

- Каждое действие (action) инициирует новое задание (job)
- Spark анализирует граф RDD и строит план выполнения
- План выполнения может включать несколько этапов (stages)
- Каждый этап состоит из набора задач (tasks) — выполняется один и тот же код на разных кусочках данных

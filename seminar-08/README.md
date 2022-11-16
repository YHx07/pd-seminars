Форма для отзывов: https://forms.gle/kkcPJtXRsJKmKyNo8. Также, я буду благодарен за Merge Request-ы в этом репозитории.

# Seminar 8

https://gitlab.com/fpmi-atp/pd2022a-supplementary/global/-/blob/main/materials/09-mapreduce_part2.md

В частности по ссылке выше есть материалы по Hadoop Java API.

# MapReduce 2

Подробная схема:

https://hadoop.apache.org/docs/stable/hadoop-mapreduce-client/hadoop-mapreduce-client-core/MapReduceTutorial.html

<img width="1200" alt="image" src="../seminar-07/MapReduce-v3.png">

## Пример на Java

Смотрим код: `velkerr/seminars/pd2020/05_wordcount_java/src/ru/mipt/examples/WordCount.java`

Основной класс `public class WordCount extends Configured implements Tool {...}`:

- Класс WordCount содержит всю логику задачи
- Базовый класс [Configured](https://hadoop.apache.org/docs/r2.4.1/api/org/apache/hadoop/conf/Configured.html) отвечает за возможность получить конфигурацию HDFS (достаточно вызвать `getConf()` внутри `run()`). Это полезно в тех случаях, когда нужно работать с HDFS из программы (например, удалять промежуточные результаты),
- [Tool](https://hadoop.apache.org/docs/stable/api/org/apache/hadoop/util/Tool.html) - интерфейс, содержащий единственный метод `run()`, который (а) парсит аргументы командной строки, (б) производит настройку Job'ы. Выполняется run() на клиенте.

### Метод run(). Важные моменты:

1. Форматы ввода-вывода:

```java
job1.setInputFormatClass(TextInputFormat.class);
job1.setOutputFormatClass(TextOutputFormat.class);
```

Это форматы, в которых пишется результат Job'ы. В Hadoop есть несколько встроенных форматов ввода-вывода. Основные: **TextInputFormat / TextOutputFormat** Результат пишется в стандартный текстовый файл. Плюсы: быстро работает. Минусы: теряет данные о типах ключей-значений. При считывании данных будем иметь пары (`LongWritable offset`, `Text rawString`). rawString нужно парсить.

**KeyValueTextInputFormat / KeyValueTextOutputFormat**
Улучшенная версия предыдущего. Типы ключей-значений хранит только если это простые типы (например, IntWritable, Text).

**SequenceFileInputFormat / SequenceFileTextOutputFormat**
Плюсы: хранит типы ключей-значений.
Минусы: Выходной файл будет записан в формате бинарного файла и прочитать его командой `hdfs dfs -cat` не выйдет.

В итоге:
* в промежуточных job'ах лучше использовать форматы KeyValue или SequenceFile.
* В последней job'е - TextOutputFormat

Любой Input / OutputFormat позволяет задавать сжатие. Это сэкономит затраты на передачу по сети.

```java
SequenceFileOutputFormat.setOutputCompressionType(job, CompressionType.BLOCK);
SequenceFileOutputFormat.setCompressOutput(job, true);
```

CompressionType - механизм сжатия. Бывает `BLOCK` (сжимаем поблочно) и `RECORD` (сжимаем записи раздельно).

*CompressionType:  Тем, кто слушает Java, полезно посмотреть на хорошую реализацию `enum` с полями и методами.*

[Больше](http://timepasstechies.com/input-formats-output-formats-hadoop-mapreduce/) входных и выходных форматов.

2. Задание кол-ва мапперов и редьюсеров.

`job.setNumReduceTasks(8);` - задаем кол-во редьюсеров.
Число мапперов напрямую задать нельзя. Система устанавливает его равным кол-ву сплитов.
* Размер сплита можно изменять (по умолчанию равен размеру блока)
* В логах Hadoop-задачи можно найти строчку: `19/10/14 21:37:44 INFO mapreduce.JobSubmitter: number of splits:2`.

### Мапперы и редьюсеры

В отличие от Hadoop Streaming, цикл обхода сплита писать не надо. Выполняются на нодах.

#### Классы-обёртки
В Hadoop существует 2 интерфейса: `Writable` и `WritableComparable`. Оба они поддерживают сериализацию / десериализацию, нужную для передачи данных между нодами.
* Типы входных-выходных ключей должны реализовывать интерфейс `WritableComparable` (т.к. их придётся сравнивать на этапе сортировки).
* Для значений достаточно `Writable`.
* Классы-обёртки, входящие в поставку Hadoop, реализовывают интерфейс `WritableComparable`.


### Сборка на Ant (https://ant.apache.org/):

Скрипт сборки: `velkerr/seminars/pd2020/05_wordcount_java/build.xml`. Сборка командой `ant` в директории с `build.xml` файлом. В результате появляется папка `jar`. Очистка командой `ant clean`. Вообще Ant --  устаревший сборщик, его заменил `maven`.

Запуск программы: `hadoop jar <путь_к_jar> <полное_имя_главного_класса> <вход> <выход> [другие_аргументы (напр. кол-во редьюсеров)]`

### Сборка на Maven :

Скрипт сборки: `velkerr/seminars/pd2020/05_wordcount_java/pom.xml`. Сборка командой `mvn package` в директории с `pom.xml` файлом. (Кэши maven лежат в директории .m2).

Скрипт запуска в файле `run.sh`:

```bash
#! /usr/bin/env bash

mvn package
hdfs dfs -rm -r -skipTrash wordcount_out 2> /dev/null
# hadoop jar jar/WordCount.jar ru.mipt.examples.WordCount /data/griboedov wordcount_out
yarn jar target/WordCount-1.0.jar ru.mipt.examples.WordCount /data/griboedov wordcount_out
```

### Глобальная сортировка

Смотрим код: `velkerr//seminars/pd2020/10-globalsort$ cd src/ru/mipt/GlobalSorter.java`

Внутри выдачи редьюсера сортировка соблюдается. Как сделать общую сортировку для всех редьюсеров? 
* Лобовое решение: пройтись по всем ключам и явно сказать Partitioner'у, какие ключи попадут на тот или иной редьюсер. Это долго.
* Выход - семплирование. Проходим не все ключи, а выбираем с некоторой вероятностью, затем аппроксиимруем на весь датасет.

Пример кода: `InputSampler.Sampler<LongWritable, Text> sampler = new InputSampler.RandomSampler<>(0.5, 10000, 10);`. 
* 0,5 - вероятность выбора записи:
* максимальное кол-во "выборов"
* максимальное кол-во сплитов.

Как только дошли до границы хотя бы по одному аргументу, семплирование прекращаем.

Такой Sampler подаётся в TotalOrderPartitioner. Подробнее см. "Hadoop. The definitive guide, 4 изд. стр. 287".

### Joins

- Reduce side join -- обычный join.
- Map side join -- mapper зачитывает маленькую таблицу в mapper, join осуществляется на стороне mapper, что экономит ресурсы на sort & merge.
- Bucket side join
 
Смотрим код: `velkerr/seminars/pd2020/11-join/StackExchangeCountHistogramRunner.java`

Решаемая задача: 

По данным stackoverflow посчитайте гистограмму количества вопросов и ответов в зависимости по возрастам пользователей. В случае отсутствия или невалидного возраста пусть будет 0. Выведите ее на печать, сортировка по возрасту (числовая, по возрастанию).
На печать: весь результат, сортировка по возрасту.

* *Входные данные:* посты stackoverflow.com
* *Формат ответа:* age <tab> num_questions <tab> num_answers

##### Описание входных данных.

`/data/stackexchange/posts` - сами посты, записаны в строках, начинающихся с ‘<row’ (можно разбирать строки вручную, без специальных xml-парсеров). Значение полей:

* PostTypeId - ‘1’, если это вопрос; ‘2’, если ответ на вопрос
* Id - идентификатор; если это вопрос, то он откроется тут: http://stackoverflow.com/questions/<Id>
* Score - показатель полезности этого вопроса или ответа
* FavoriteCount - добавления вопроса в избранные
* ParentId - для ответа - идентификатор поста-вопроса
* **OwnerUserId - идентификатор пользователя - автора поста**

`/data/stackexchange/users` - пользователи

* **Id - идентификатор пользователя** (в posts это OwnerUserId)
* Age - возраст (может отсутствовать)
* Reputation - уровень репутации пользователя

Есть несколько семплов датасета (stackexchange100, stackexchange1000 - 100-я и 1000-я часть исходного датасета соответственно).

Обратите внимание на классы `StackExchangeEntry` и `PairWritable`. Это пример написания собственных [Writable](https://hadoop.apache.org/docs/r3.0.1/api/org/apache/hadoop/io/Writable.html)-обёрток. 
  
#### Схема решения: 
*1я Job*
* **Mapper**: строка датасета (users или posts) -> пары таких типов: 
    * (userId, tag, возраст, 0, 0) для пользователя,
    * (userId, tag, 0, 1, 0) для вопроса
    * (userId, tag, 0, 0, 1) для ответа

`tag` разделяет пользователей и посты (например U для пользователей и P для постов).

`userId` это Primary key, по которому будем делать join.

`возраст` храним т.к. именно по нему нужно будет сортировать

Работать со сложными ключами тяжело. В Hadoop Streaming приходится настраивать comparator и partitioner, на Java API удобнее написать свой Writable-класс (с сериализацией). В примере это StackExchangeEntry.java

* **Reducer**: (userId, [(tag, возраст, 0, 0), (tag, 0, 1, 0), (tag, 0, 1, 0), ...]) -> (возраст, кол-во\_вопросов, кол-во\_ответов)

*2я Job*
* Доаггрегирование пар (возраст, кол-во\_вопросов, кол-во\_ответов). WordCount по возрасту (ключ - возраст, значение - вопросы и ответы).
* Сортировка по возрасту.

**СЛОЖНО!!**. Выход: Hive.
  
### Пример где есть всё

Смотрим код: `velkerr/seminars/pd2020/12-avg-page-life`


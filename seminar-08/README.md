# Seminar 8

https://gitlab.com/fpmi-atp/pd2022a-supplementary/global/-/blob/main/materials/09-mapreduce_part2.md

В частности по ссылке выше есть материалы по Hadoop Java API.

# MapReduce 2

Подробная схема:

https://hadoop.apache.org/docs/stable/hadoop-mapreduce-client/hadoop-mapreduce-client-core/MapReduceTutorial.html

<img width="1200" alt="image" src="../seminar-07/MapReduce-v3.png">


https://hadoop.apache.org/docs/r3.0.1/api/org/apache/hadoop/io/Writable.html

### Пример на java

Смотрим код:

`velkerr/seminars/pd2020/05_wordcount_java/src/ru/mipt/examples/WordCount.java`

#### Сборка на Ant (https://ant.apache.org/):

Скрипт сборки: `velkerr/seminars/pd2020/05_wordcount_java/build.xml`.

Сборка командой `ant` в директории с `build.xml` файлом. В результате появляется папка `jar`. Очистка командой `ant clean`. Вообще Ant --  устаревший сборщик, сейчас пользуются `maven`.

#### Сборка на Maven :

Скрипт сборки: `velkerr/seminars/pd2020/05_wordcount_java/pom.xml`.

Сборка командой `mvn package` в директории с `pom.xml` файлом. (Кэши maven лежат в директории .m2).

Запуск в файле `run.sh`:

```bash
#! /usr/bin/env bash

mvn package
hdfs dfs -rm -r -skipTrash wordcount_out 2> /dev/null
# hadoop jar jar/WordCount.jar ru.mipt.examples.WordCount /data/griboedov wordcount_out
yarn jar target/WordCount-1.0.jar ru.mipt.examples.WordCount /data/griboedov wordcount_out
```

### Глобальная сортировка

Смотрим код:

`velkerr//seminars/pd2020/10-globalsort$ cd src/ru/mipt/GlobalSorter.java`


`velkerr/seminars/hadoop-course/02-avg-page-life$`
### Joins

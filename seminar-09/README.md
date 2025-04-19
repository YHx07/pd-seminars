# Seminar 9

https://gitlab.com/fpmi-atp/pd2022a-supplementary/global/-/blob/main/materials/10-hive_part1.md

# Hive 1

Для рабыот с Hadoop удобен сервис Hue. Он будет полезен и для Hive.

### Apache Hue (hadoop user experience)
1. Заходим на кластер по shh, пробрасываем порт `-L 8888:mipt-node03:8888`
2. В HUE заходим, например, как пользователь `hue_user`, пароль `hue_password`

---
Мотивация этого занятия: 

Вопрос: какие СУБД вы знаете? Знаем, например, PostgreSQL, MySQL. Можем ли мы использовать их, вместо что обсуждаем про Hadoop? 
Идея: вместо трудоемкого написания MapReduce для залачи Word Count (сколько раз слово встретилось в тексте) придумаем как положить эти тексты в базу данных,
и затем сделаем `SELECT word, count(*) as word_count FROM table GROUP BY word`. Но на самом деле, СУБД не всегда умеют работать с большим объемом данных, как сервисы в экосистеме Hadoop.
Почему не все СУБД могут работать с большими данными -- дело в их архитектуре, часто СУБД разворачиваются в пределах одной машины и ограничены жестким диском этой машины. 
Не все СУБД умеют работать распредлеенно. 
Например, PostreSQL можно настроить на распределенную работу используя некоторые дополнительные модули,но из коробки так сделать нельзя.
Вообще, если хочется настроить распределенный PostgreSQL, то стоит обратить внимание на Greenplum.

Еще одна идея: 
Мы знаем, как устроено хранение данных в Hadoop -- это HDFS, он горизонтально масштабируется почти до бесконечности. 
Мы знаем, как данные в HDFS обработать -- для этого есть Hadoop Streaming (и вообще парадигма MapReduce).
Мы знаем синтаксис языка запросов SQL. Давайте придумаем как через язык SQL обрабатывать данные в Hadoop. 
-- это как раз Hive

### Архитектура Hive:

![hive-architecture](https://github.com/YHx07/pd-seminars/assets/36137274/98df215d-97a6-4423-990b-1c359fa8d994)
https://cloud.mail.ru/public/Shqp/ukAovqDPh

#### Выполнение SQL запроса:

1. Parser. Парсинг SQL запроса. Проверка синтаксиса (скобки, пкнктуация)
2. Semantic Analyzer — проверка типов данных, приведение типов данных, раскрытие *.
3. Logical Plan Operation — построение дерева выполнения запросовв виде MapReduce job. Оптимизация запросов:
    - Два join-а в одну mapreduce задачу
    - Reduce side join → map side join
    - Group By можно сделать разными способами (часть на map-стадии)
4. Query Plna Generator — подготовка физического плана запроса: где брать данные и на каких нодах выполняется запрос.

Красткий вывод из архитектуры:
1. Простые запросы, типа `SELECT * FROM table` обрабатываются обращением в Metastore.
2. Сложные запросы, в которых есть группировка, join, агрегация, where можно представить в виде последовательности MapReduce задач.

### Компоненты:

1. Driver — Manages the lifecycle of HQL statement
2. Compiler – Compiles HQL into DAG i.e. Directed Acyclic Graph
3. Metastore. Хранит метаданные:
    - Базы данных (database) — пространство имен таблиц
    - Таблицы (tables) — список столбцов, создателя, тип хранилища,десериализатор данных
4. Execution engine

### Пример плана запроса

![hive-explain](https://github.com/YHx07/pd-seminars/assets/36137274/c77c7e02-a068-41cd-a933-1fa1bad8d4d8)
https://cloud.mail.ru/public/xMpn/FbPGiTLya

### Hive. Типы таблиц

- По влиянию на данные
    - External — не можем менять данные, только метаинформацию
    - Managed — можем менять данные
- По времени жизни
    - Permanent — существует всегда
    - Temporary — существует во время Hive сессии

### Hive. Партиционирование

`set hive.exec.dynamic.partition=true;` — динамическое создание партиций. По умолчанию Hive партиционирует статически, т.е. создаётся фиксированное число партиций. Если мы не знаем, сколько у нас уникальных значений в партиционируемой колонке, то не знаем сколько потребуется партиций. Эта опция стоит по умолчанию.

`SET hive.exec.dynamic.partition.mode=nonstrict;` - "нестрогое" партиционирование. При динамич. партиционировании, hive требует чтобы хотя бы 1 партиция была статическая. Снимаем это требование.


### Hive. Написание запроса

1. Можно работать через Apache HUE — UI интерфейс для HIVE.
2. Можно запустить HIVE в терминале (будет открыт интерактивный shell для hive): команда `hive`.
    
    Подготовим .sql файлы:
    
    1. Создать "базу данных":
        
        file.sql:
        
        ```bash
        CREATE DATABASE <YOUR_USER> LOCATION '/user/<YOUR_USER>/test_metastore';
        ```
        
        LOCATION — место где хранятся метаданные
        
        Создаем БД:
        
        ```bash
        hive -f <file.sql>
        ```
        
        - Описание БД
            
            ```bash
            DESCRIBE DATABASE <YOUR_USER>
            ```
            
            ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/fc374b23-63ca-4288-96f4-7ecf5a561929/Untitled.png)
            
            название, где хранится metastore, кто создал.
            
    2. Создать таблицу
        
        file.sql:
        
        ```bash
        USE <YOUR_USER>; #подключились к бд
        
        DROP TABLE IF EXISTS Subnets;
        CREATE EXTERNAL TABLE Subnets (
             ip STRING,
             mask STRING
        ) ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t' 
        STORED AS TEXTFILE 
        LOCATION '/data/subnets/big';
        ```
        
        Запускаем:
        
        ```bash
        hive -f <file.sql>
        ```
        
    
    Можно выполнить сразу конкатенацию запросов:
    
    ```bash
    hive -f <file1.sql> && hive -f <file2.sql>
    ```
    
3. Можно выполнить однострочный SQL запрос:
    
    ```bash
    hive --database <YOUR_USER> -e 'show tables'
    ```
    
4. Настройка ресурсов и названия Jobы:
<img width="516" alt="Screenshot 2023-12-03 at 01 54 25" src="https://github.com/YHx07/pd-seminars/assets/36137274/72ead2e7-5293-4098-a139-d644e46f3b8f">

Просто EXPLAIN выведет план запроса вместо выполнения самого запроса.

EXPLAIN EXTEND — тоже план, но больше статистики. 

5. Парсинг входных данных с помощью регулярных выражений

Какие бывают SERDEPROPERTIES :

https://cwiki.apache.org/confluence/display/Hive/UserGuide

P.S. Если задать сделать ошибку в SERDEPROPERTIES, то создание бд пройдет успешно. Будет ошибка когда будем читать данные. Потому Schema on read.

Можно таким образом предобработать столбец

### Hive. ### Оптимизация

- Партиционирование
    
    [Hive - Dynamic Partition (DP)](https://datacadamia.com/db/hive/dp#mode)
    
    ```bash
    SET hive.exec.dynamic.partition.mode=nonstrict;
    
    USE <YOUR_USER>_test;
    DROP TABLE IF EXISTS SubnetsPart;
    
    CREATE EXTERNAL TABLE SubnetsPart (
        ip STRING
    )
    PARTITIONED BY (mask STRING)
    STORED AS TEXTFILE;
    
    INSERT OVERWRITE TABLE SubnetsPart PARTITION (mask)
    SELECT * FROM Subnets;
    ```
    
    `set hive.exec.dynamic.partition=true;` — динамическое создание партиций. По умолчанию Hive партиционирует статически, т.е. создаётся фиксированное число партиций. Но в данном случае мы не знаем, сколько у нас уникальных значений маски, а значит не знаем сколько потребуется партиций. Эта опция стоит по умолчанию.
    
    `SET hive.exec.dynamic.partition.mode=nonstrict;` - "нестрогое" партиционирование. При динамич. партиционировании, hive требует чтобы хотя бы 1 партиция была статическая. Снимаем это требование.
    
    P.S.
    
    в таблице с партициями второе поле пишем только в PARTITIONED BY
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b3678dd4-0e96-486a-adef-5d5c514c0789/Untitled.png)
    
- Кластеризация
- Семплирование
- Map-side Join

### Join-ы

[LanguageManual JoinOptimization](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+JoinOptimization#LanguageManualJoinOptimization-PriorSupportforMAPJOIN)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/bbb6e934-9585-4f4e-8ccb-9ddf9812f846/Untitled.png)

Сделать mapjoin:

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/60207e6a-1485-4270-b4d2-5e25c9265c13/Untitled.png)

P.S. limit 10 не делает глобальную сортировку

Order by — глобальная сортировка

### Hive streaming

Существует 2 основных способа использования внешних скриптов в Hive Streaming.

Решение 2-мя способами:

- Использование команды
    
    ```sql
    USE example;
    
    SELECT TRANSFORM(ip, mask)
    USING 'cut -d . -f 1' AS ip
    FROM Subnets
    LIMIT 10;
    ```
    
- Подключение внешних скриптов
    
    ```bash
    #! /usr/bin/env bash
    
    cut -d . -f 1
    ```
    
    ```sql
    ADD FILE ./script.sh;
    
    USE example;
    SELECT TRANSFORM(ip, mask)
    USING './script.sh' AS ip2
    FROM SubnetsLIMIT 10;
    ```
    

`TRANSFORM`: выбирает поле, кот. мы будем обрабатывать с помощью streaming.

Как запускать.

Варианта А:
1. В директории лежат: task.sql, script.py
2. вы пишите hive -f task.sql

Вариант В:
Если у вас скачет репозиторий с GitLab, то:
1. В директории лежат: task.sql, script.py, run.sh
2. В run.sh скрипт: hive -f task.sql
2. Вы пишите bash run.sh

Для запуска на GitLab в run.sh: hive -f task.sql + рядом с task.sql лежит файл script.py
### UDF

- Regular UDF: обрабатываем вход построчно,
- UDAF: аггрегация, n строк на вход, 1 на выходе,
- UDTF: 1 строка на вход, таблица (несколько строк и полей) на выходе,
- Window functions: "окно" (несколько строк, *m*) на вход, несколько строк(*n*) на выходе (1 строка для каждого окна). Функции аггрегации и UDAF тоже могут быть использованы в качестве оконных.

**(Regular) User-defined functions**

1. Для реализации UDF нужно создать Java-класс, являющийся наследником класса org.apache.hadoop.hive.ql.exec.UDF.
2. Реализовать в этом классе один или несколько методов evaluate(), в которых будет записана логика UDF.
3. Для сборки нужно подключить ещё один Jar-файл: `/opt/cloudera/parcels/CDH/lib/hive/lib/hive-exec.jar`
4. Для использования UDF в запросе нужно:
    1. добавить собранный Jar-файл в Distributed cache (можно использовать относительный путь): `ADD JAR <path_to_jar>`
    При этом никаких дополнительных Jar-файлов в запросе можно не добавлять т.к. Jar с UDF уже содержит все необходимые коды.
    2. создать функцию на основе Java-класса UDF: `CREATE TEMPORARY FUNCTION <your_udf> AS 'com.your.OwnUDF';`

С UDF:

- много кода,
- только **Java** :(

Без UDF:

- Ещё больше кода (правда на SQL). Пример: `2-sum_udf/query_without_udf.sql`
- Не всегда можно реализовать в 1 запрос => будут подзапросы => будет несколько Job (дольше).

**User-defined table functions (UDTF)**

От обычных UDF данный вид функций отличается тем, что на выходе может быть больше одной записи. Причём столбцов также может быть сгенерировано несколько, т.е. по одной записи на входе мы можем получить целую таблицу. Отсюда и название.

1. Для реализации UDTF нужно создать класс-наследника от org.apache.hadoop.hive.ql.udf.generic.GenericUDTF.
2. Логика UDTF пишется в 3 методах:
    
    а) `initialize()`:
    
    - разбор входных данных (проверка количества аргументов и их типов), сохранение данных в ObjectInspector'ы
    - создание структуры выходной таблицы (названия и типы полей)
    
    б) `process()`: реализация механизма получения выходных данных из входных,
    
    в) `close()`: аналог cleanup() в MapReduce. Обрабатывает то, что не было обработано в `process()`. Здесь не используется т.к. не аггрегируем, используется в основном в UDAF, см. ниже.
    
3. Собираем Jar также, как и в случае с обычными UDF, однако для сборки подключить нужно не 1, а 2 дополнительных Jar:
    
    ```
    /opt/cloudera/parcels/CDH/lib/hive/lib/hive-exec.jar
    /opt/cloudera/parcels/CDH/lib/hive/lib/hive-serde.jar
    
    ```
    

**User-defined aggregation functions (UDAF)**

Позволяют реализовать свои функции наподобие `SUM()`, `COUNT()`, `AVG()`.

**Доп. литература.** [Programming hive](https://www.oreilly.com/library/view/programming-hive/9781449326944/), гл. 13 "Functions" (с. 163).

---

### Features of Apache Hive:

- Hive supports client-application written in any language like Python, Java, PHP, Ruby, and C++.
- It generally uses RDBMS as metadata storage, which significantly reduces the time taken for the semantic check.
- Hive Partitioning and Bucketing improves query performance.
- Hive is fast, scalable, and extensible.
- It supports Online Analytical Processing and is an efficient ETL tool.
- It provides support for **User Defined Function** to support use cases that are not supported by Built-in functions.

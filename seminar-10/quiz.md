### № 1
Вам нужно перемножить две матрицы размером 10.000.000.000 х 500 и 500 х 5. Какой тип join-а лучше всего выбрать?
- Map-side join (*)
- Reduce-side join
- Bucket-side join
- Не имеет значения (любой)
### № 2
Имеются две таблицы: первая размером 20.000.000 строк с колонками A,B,C,D и вторая размером 2.000.000 строк с колонками A,B,E,F,G. Первая таблица партиционирована по столбцам A, B, вторая по столбцу B. Данные в таблицах не отсортированы. Вам нужен INNER JOIN таблиц по столбцу B. Какой тип join-а в Hive лучше всего выбрать?
- Bucket Map Join (*)
- Map Join
- Skew Join
- Sort Merge Bucket Join
### № 3

Имеются две таблицы: первая размером 20.000.000 строк с колонками A,B,C,D и вторая размером 2.000 строк с колонками A,B,E,F,G. Первая таблица партиционирована по столбцам A, B, вторая по столбцу B. Вам нужен INNER JOIN таблиц по столбцу A. Какой тип join-а в Hive лучше всего выбрать?

- Map Join (*)
- Bucket Map Join
- Skew Join
- Sort Merge Bucket Join

### № 4

Имеются две таблицы: первая размером 100.000.000 строк с колонками A,B,C,D и вторая размером 7.000.000 строк с колонками A,B,E,F,G. Первая таблица партиционирована по столбцам A, B, вторая по столбцу B.

Первая таблица содержит информацию о жителях России, в частности столбец А — город проживания. Столбец B — идентификатор компании в которой работает человек. Вторая таблица содержит информацию о компаниях России, где А — город, В — идентификатор компании.

Вам нужен INNER JOIN таблиц по столбцу A. Какой тип join-а в Hive лучше всего выбрать?

- Skew Join (*)
- Map Join
- Bucket Map Join
- Sort Merge Bucket Join

### № 5

Имеются две таблицы: первая размером 20.000.000 строк с колонками A,B,C,D и вторая размером 2.000.000 строк с колонками A,B,E,F,G. Первая таблица партиционирована по столбцам A, B, вторая по столбцу B. Данные в таблицах отсортированы по столбцу В. Вам нужен INNER JOIN таблиц по столбцу B. Какой тип join-а в Hive лучше всего выбрать?

- Sort Merge Bucket Join (*)
- Bucket Map Join
- Map Join
- Skew Join

### № 6

Какие операции над Spark DataFrame могут привести к увеличению числа партиций?

- SortMergeJoin (*)
- Broadcast Hash Join
- Shuffle Hash Join
- Ни один вариант не приводит к увеличению числа партиций

### № 7

Сколько джоб запустит запрос написанный на hive? Размеры таблицы: logs — 1.000.000 записей, Users — 80.000, IPRegions — 8.000, Subnets — 2.500.

```SQL
SET hive.auto.convert.join=false;

SELECT 
		reg AS region,
	  sum(m_cnt) AS male_cnt,
    sum(f_cnt) AS female_cnt
FROM (
		SELECT /*+ MAPJOIN(IPRegions,Users) */
			  IPRegions.region AS reg,
        if(Users.sex='male', count(1), 0) AS m_cnt,
        if(Users.sex='female', count(1), 0) AS f_cnt
		FROM Logs
		JOIN IPRegions ON Logs.ip = IPRegions.ip
		JOIN Users ON Logs.ip = Users.ip
    GROUP BY 
			IPRegions.region, Users.sex
   ) AS sub
GROUP BY reg;
```

- 2 (*)
- 1
- 3
- 4

### № 8

Какой алгоритм join используется по умолчанию в Spark (без оптимизации)

- Sort-merge join (*)
- Shuffle Hash join
- Broadcast join
- Broadcast Nested Loop join

### № 9

Какая асимптотика работы Sort-Merge Join? Таблицы размером n и m:

- O(n * log(n) + m * log(m)) (*)
- O(n^2 + m^2)
- O(n + m)
- O(n * m)

### № 10

Какая асимптотика работы Hash Join? Таблицы размером n и m:

- O(n + m) (*)
- O(max(n, m))
- O(min(n, m))
- O(n * m)

### № 11

Какая асимптотика работы Nested Loop Join? Таблицы размером n и m:

- O(n * m) (*)
- O(n + m)
- O((n * m)^2)
- O(n^2 + m^2)

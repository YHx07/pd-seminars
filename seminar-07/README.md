# Seminar 7

https://gitlab.com/fpmi-atp/pd2022a-supplementary/global/-/blob/main/materials/08-mapreduce_part1.md

# MapReduce 1

<img width="800" alt="image" src="./MapReduce.png">

https://hadoop.apache.org/docs/stable/hadoop-mapreduce-client/hadoop-mapreduce-client-core/MapReduceTutorial.html

<img width="1200" alt="image" src="./MapReduce-v3.png">

Напоминание наш кластер:

<img width="800" alt="image" src="./Cluster.png">

Заходим на кластер: `$ ssh <username>@mipt-client.atp-fivt.org -L 8088:mipt-master:8088 -L 19888:mipt-master:19888 -L 8888:mipt-node03:8888`. Значения XXX см. на почте в письме от automation@atp-fivt.org.
  
В случае проблем с доступами или ключами заходим по логину `hdfsuser` (пароль `hdfsuser`). Это не желательно т.к. у вас в этом случае будет общая очередь на кластере.
 
 Порты:
 - 8088 --
 - 19888 --
 - 8888 --

# Материалы

1. https://www.michael-noll.com/tutorials/writing-an-hadoop-mapreduce-program-in-python/
2. https://github.com/Ebazhanov/linkedin-skill-assessments-quizzes/blob/main/hadoop/hadoop-quiz.md

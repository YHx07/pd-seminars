# Seminar 7

https://gitlab.com/fpmi-atp/pd2022a-supplementary/global/-/blob/main/materials/08-mapreduce_part1.md

# MapReduce 1

<img width="700" alt="image" src="./MapReduce.png">

https://hadoop.apache.org/docs/stable/hadoop-mapreduce-client/hadoop-mapreduce-client-core/MapReduceTutorial.html

<img width="1200" alt="image" src="./MapReduce-v3.png">

При помощи [oozie](https://oozie.apache.org/) или [luigi](https://github.com/spotify/luigi) MapReduce задачи можно объединить в сложные последовательности:

<img width="600" alt="image" src="https://user-images.githubusercontent.com/36137274/198106002-b8536b49-868e-4e40-9d44-a98ed394124a.png">

Напоминание наш кластер:

<img width="400" alt="image" src="./Cluster.png">

Заходим на кластер: `$ ssh <username>@mipt-client.atp-fivt.org -L 8088:mipt-master:8088 -L 19888:mipt-master:19888 -L 8888:mipt-node03:8888`. Значения XXX см. на почте в письме от automation@atp-fivt.org.
  
В случае проблем с доступами или ключами заходим по логину `hdfsuser` (пароль `hdfsuser`). Это не желательно т.к. у вас в этом случае будет общая очередь на кластере.
 
 Порты:
 - 8088 --
 - 19888 --
 - 8888 --

# Ещё про сервисы по работе с hadoop

https://platform.digital.gov.ru/docs/analytics/platform-v-hadoop

# Материалы

1. https://www.michael-noll.com/tutorials/writing-an-hadoop-mapreduce-program-in-python/
2. https://github.com/Ebazhanov/linkedin-skill-assessments-quizzes/blob/main/hadoop/hadoop-quiz.md
3. https://cloud.yandex.ru/services/data-proc
4. [https://sber.ru/legal/data_platform/](https://sber.ru/legal/data_platform/#SDP_Hadoop)

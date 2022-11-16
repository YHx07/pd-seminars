Форма для отзывов: https://forms.gle/kkcPJtXRsJKmKyNo8. Также, я буду благодарен за Merge Request-ы в этом репозитории.

# Seminar 1 

### Параллельные и распределенные вычислительные системы

__Параллельные__:
* многократное ускорение
* высокопроизводительные машины
* *отказоустойчивости нет
* с точки зрения разработчика -- набор взаимодействующих процессов

__Распределенные__:
* большие объёмы данных
* обычные машины
* отказоустойчивость
* с точки зрения разработчика -- одно распределенное вычислительное устройство

### MPI программа на С/С++

Реализация MPI-функций лежит в заголовочном файле ```mpi.h```.
Зона параллельной части программы находится между вызовами функций ```MPI_Init``` и ```MPI_Finalize```.
Чтобы получить количество процессов, используется ```MPI_Comm_size```. Определить id процесса среди N запущенных 
можно через ```MPI_Comm_rank```.

![MPI Reference](pic/mpi_reference.png)

Пример программы, выполняющей ```"Hello, World!"``` на каждом процессе и считающей количество процессов:

```
#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    
    int procid, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &procid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_length;
    MPI_Get_processor_name(processor_name, &name_length);
    
    printf("Hello, World! My id is %d and my processor name is %s\n", procid, processor_name);
    
    if (procid == 0) {
        printf("All processes count: %d\n", num_procs);
    }
    
    MPI_Finalize();
    return 0;
}
```

Пример на C++:

```
#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int procid, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &procid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_length;
    MPI_Get_processor_name(processor_name, &name_length);

    std::cout << "Hello, world! My id is " << procid << " and my processor name is " << processor_name << " out of " << world_size << std::endl;
	
    MPI_Finalize();
    return 0;
}
```

### Компиляция MPI программы на С/С++

Компиляция программ происходит при помощи компиляторов ```mpicc``` и ```mpic++```. Сначала нужно подключить модуль при помощи команды (old)

```[bash]
module add mpi/openmpi4-x86_64
```

В 2022 году: 

```[bash]
module add centos/8/mpi/hpcx-v2.7.0
```

Посмотреть какие модули есть на кластере можно с помощью команды:

```[bash]
module avail
```

После этого ```mpicc``` и ```mpic++``` подгрузятся в `$PATH` и можно:

```[bash]
mpicc <FILE_NAME>.c
```

```[bash]
mpic++ <FILE_NAME>.cpp
```

### Запуск программ MPI локально

_Инструкция ниже подходит для запуска на пользовательских машинах и на клиенте кластера_

Для локального запуска можно использовать скрипт:
```bash
#!/bin/bash
mpiexec -np 4 ./a.out
```

Опция ```-np (-c|-n|-np|--np <arg0>)``` используется для указания количества процессов.

MPI локально может быть установлен для следующих ОС (на кластере уже есть):

Ubuntu: `sudo apt-get install openmpi-bin libopenmpi-dev`

Mac OS: `brew install open-mpi`

### Команды по использованию SLURM

* `sinfo` - посмотреть информацию по нодам кластера
* `sinfo -N -l` - посмотреть информацию по каждой ноде кластера
* `squeue` - посмотреть очередь задач
* `srun <command>` - запустить команду на ноде кластера
* `sbatch <script>` - запустить скрипт на нодах кластера. Каждый скрипт должен начинаться с `#!/bin/bash`.
  После этого должно высветиться сообщение `Submitted batch job <job_id>`, результаты работы попадают в лог-файл `slurm-<job_id>.out`.

### Запуск программ MPI на кластере

Чтобы отправить задачу в очередь на Slurm (ресурсный менеджер кластера):
```bash
sbatch -n <NUM_OF_PROCESSES> ./run.sh
```

или с помощью sbatch-файла ```run_sbatch_config.sh```, который имеет вид:

```bash
#!/usr/bin/env bash
#
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --job-name=my_job
#SBATCH --output=out.txt
#SBATCH --error=error.txt
mpiexec ./a.out
```
```bash
sbatch ./run_sbatch_config.sh
```

### Локальный запуск программ через Docker:

Создаём файл run_docker.sh:
```
CONTAINER_NAME=pd-mpi

docker run -d -e TZ=Europe/Moscow -e OMPI_ALLOW_RUN_AS_ROOT=1 -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
  --volume=`pwd`:/home --name="${CONTAINER_NAME}" ubuntu:18.04 tail -f /dev/null
docker exec -t "${CONTAINER_NAME}" apt update
docker exec -t "${CONTAINER_NAME}" apt upgrade -y
docker exec -t "${CONTAINER_NAME}" apt-get install -y \
  build-essential make vim g++ sudo libomp-dev cmake libopenmpi-dev \
  openmpi-common openmpi-bin libopenmpi-dev openssh-client openssh-server net-tools netcat iptables
```

Запускаем `bash run_docker.sh`, поднимается контейнер:

<img width="1382" alt="image" src="https://user-images.githubusercontent.com/36137274/190561579-356e8c4f-9662-423e-bb4b-7377556b608d.png">

Подключаемся к контейнеру командой:

`docker exec -it pd-mpi /bin/bash`

#### Запуск программ

1. `mpic++ main.cpp `
2. `mpiexec -np 4 --allow-run-as-root ./a.out`

#### Запуск программ с make

1. `cmake . `
2. `make -j9`
3. `mpiexec -n 4 --allow-run-as-root bin/..`


#### Репозиторий с кодом:
https://github.com/akhtyamovpavel/ParallelComputationExamples/tree/master/MPI

### Полезные ссылки:
* [Mануал OpenMPI 4.0](https://www.open-mpi.org/doc/current/)
* [Tutorial по MPI с примерами](https://mpitutorial.com)
* [Примеры работы с sbatch в bash](https://hpc-uit.readthedocs.io/en/latest/jobs/examples.html)
* [Python-клиент для работы с sbatch](https://github.com/luptior/pysbatch)
* [Python-библиотека для разработки MPI программ](https://mpi4py.readthedocs.io/en/stable/intro.html)

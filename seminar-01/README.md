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

Пример программы, выполняющей "Hello, World!" на каждом процессе и считающей количество процессов:

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

Локальный запуск программ через Docker:

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

Запускаем `{bash run_docker.sh}`, поднимается контейнер:

<img width="1382" alt="image" src="https://user-images.githubusercontent.com/36137274/190561579-356e8c4f-9662-423e-bb4b-7377556b608d.png">

Подключаемся к контейнеру командой:

`docker exec -it pd-mpi /bin/bash`

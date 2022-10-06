# Seminar 3

Все программы по CUDA лежат тут: https://github.com/akhtyamovpavel/ParallelComputationExamples/tree/master/CUDA

# CUDA

Хороший сайт с характеристиками видеокарт (google: gtx 1080 specs..):
- https://www.techpowerup.com/gpu-specs/geforce-gtx-1080.c2839

Ещё полезную информацию можно найти в английской википедии в разделе Technical Specification:
- https://en.wikipedia.org/wiki/CUDA. (Например, что warp size всегда был 32)

# Пирамида памяти CPU vs GPU

# Shared memory

Shared memory -- GPU, в отличие от CPU, даёт возможность управлять L1 кэшом. 
На пирамиде памяти видим, что на GPU по сравнению с CPU можно гораздо чаще обращаться к L1 и L2 кэшам. 
В каждом блоке максимальный размер L1 кэша -- 48кб, причём его можно выделить ключевым словом `__shared__`. 
То есть если мы активно используем данные с оперативной памяти внутри одного блока, то данные лучше перекинуть в shared память и затем использовать их из shared памяти.

Чтобы не сломать кэш-линию, внутри warp-а должен быть последовательный доступ.

## Пример 1: ломаем кэш-линию

Работаем с /03-memory-model/01-uncoalesced-access/full_uncoalesce.cu.

Рассмотрим две реализации:

```cu
#define ILP 8

__global__
void add(int n, float* x, float* y, float* z) {
    int tid = threadIdx.x + ILP * blockDim.x * blockIdx.x;
    for (int i = 0; i < ILP; ++i) {
        int current_tid = tid + i * blockDim.x;
        
        z[current_tid] = 2.0f * x[current_tid] + y[current_tid];
    }
}
```

```
__global__
void stupid_add(int n, float* x, float* y, float* z) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int actual_tid = ILP * index;
    for (int i = 0; i < ILP; ++i) {
        int current_tid = actual_tid + i;
        z[current_tid] = 2.0f * x[current_tid] + y[current_tid];
    }
}
```

## Пример 2: 

Работаем с /03-memory-model/02-shared/*_example.cu

Считаем сумму элемента и его соседей слева и справа.

Сравним две реализации:

global:

```cu
__global__ void ComputeTriSum(int n, int* input, int* result) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int result_tmp = 0;

    if (tid > 0) {
        result_tmp = input[tid - 1];
    }
    if (tid + 1 < n) {
        result_tmp = result_tmp + input[tid + 1];    
    }

    result_tmp = result_tmp + input[tid];
    result[tid] = result_tmp;
}
```

shared:

```cu
#define BLOCKSIZE 512

__global__ void ComputeThreeSum(int n, int* input, int* result) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int local_tid = threadIdx.x;
    __shared__ int s_data[BLOCKSIZE]; // unique for every block!

    int tmp = 0;
    s_data[local_tid] = input[tid]; // copy data to shared memory
    
    __syncthreads();

    if (local_tid > 0) {
        tmp = s_data[local_tid - 1];
    } else if (tid > 0) {
        tmp = input[tid - 1];
    }

    if (local_tid + 1 < BLOCKSIZE) {
        tmp = tmp + s_data[local_tid + 1];
    } else if (tid + 1 < n) {
        tmp = tmp + input[tid + 1];    
    }

    tmp = tmp + s_data[local_tid];
    result[tid] = tmp;
}
```

Модификатор `__shared__` говорит, что выделяется память внутри блока. 
Выделяем память по размеру равную `BLOCKSIZE`.

Копируем данные из обычной памяти в shared память. Это происходит на каждом потоке параллельно. Далее нам нужно синхронизировать память внутри блока. 
`__syncthreads()` -- барьер внутри одного блока, то есть ждем все threads внутри блока. Важно: нельзя делать `__syncthreads()` внутри if -- будет deadlock.

Если local_tid > 0, то мы обратимся к данным внутри блока, если = 0, то мы находимся в начале блока и нам надо обратиться к данным из глобальной памяти (инструкция начинает работать как в примере global).

Запустим:

```bash
nvcc global_example.cu -o global_example 
nvcc shared_example.cu -o shared_example
./global_example 
>> 3.96787 elapsed
./shared_example 
>> 4.37818 elapsed
```

В итоге код с shared данными работает медленее, потому что обращений к shared данным в этой программе не много, поэтому выигрыш несущественный.

## Пример 3 (эффективный): 

Сейчас только запустим, разберём в следующий раз: /04-reduction/01-default-sum/main*.cu. 
Программа считает сумму элементов массива.
Тут действительно есть выигрыш за счёт интенсивного использования shared memory.

Запустим:

```bash
nvcc main.cu -o main 
nvcc main_without_shared_memory.cu -o main_without_shared_memory
./main 
>> 0.183968 elapsed
./main_without_shared_memory 
>> 0.206272 elapsed
```

На самом деле должно быть ещё быстрее (06-..). И нужно синхнонизировать потоки!

## Пример 4 (перемножение матриц): 

Работаем с 03.5-matrix-multiplication-example/main.cu.

Тут мтарицы одномерные, но в блоках можно работать и с многомерными случаями. В домашнем задании надо будет к этому коду добавить shared memory. 

## CUBLAS

06-cublas/03-cosine-distance.cu

Есть библиотека BLAS -- классическая библиотека для линейной алгебры (Basic Linear Algebra Subprograms). CUBLAS -- то же, но на CUDA-е.

```
cublasCreate(...) -- вызываем конструктор класса
cublasSetVector(...) -- аллоцируем массив, замена cudaMemCpy
cublasSdot(...) -- вычисление скалярного произведения
cublasSnrm2(...) -- вычисление l2 нормы
```

## PyCuda

07-pycuda/01-simple-ariphmetic.py

Можно писать и на python. Запустим программу:

```bash
python3.6 01-simple-ariphmetic.py 
```

Если не работает, то установить недостающие библиотеки можно использую команду:

```bash
python3.6 -m pip install pycuda --user
```

# Материалы:

- [CUDA в примерах (рус)](https://cloud.mail.ru/public/DCPf/aCk7BnMTJ)
- [Developer.nvidia.com](https://developer.nvidia.com/accelerated-computing-training)
- [Sample Code](https://github.com/nvidia/cuda-samples)
- [Seminar CUDA 1 -- A. Chernetskiy](https://gitlab.com/fpmi-atp/pd2022a-supplementary/chernetskiy/-/blob/main/Seminar_3_CUDA_1.md)
- [Seminar CUDA 2 -- A. Chernetskiy](https://gitlab.com/fpmi-atp/pd2022a-supplementary/chernetskiy/-/blob/main/Seminar_4_CUDA_2.md)
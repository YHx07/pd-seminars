Форма для отзывов: https://forms.gle/kkcPJtXRsJKmKyNo8. Также, я буду благодарен за Merge Request-ы в этом репозитории.

# Seminar 5

Все программы по CUDA лежат тут: https://github.com/akhtyamovpavel/ParallelComputationExamples/tree/master/CUDA

В этом семинаре по сути то же, что на лекции: про задачи Reduction и Scan. 

Примеры:

1. Reduction. Наивный алгоритм. `/04-reduction/01-default-sum`.
2. Reduction. Уменьшим количество warp-итераций. Bank conflict. `/04-reduction/02-another-math`.
3. Reduction. Решение bank conflict-а. `/04-reduction/03-improving-bank-conflicts`.
4. Reduction. Ещё ускорение (Количество блоков меньше в 2 раза). `/04-reduction/03-improving-bank-conflicts`.
5. Reduction. Ещё ускорение (Разворачиваем цикл).`/04-reduction/05-warp-reduce`.
6. Reduction. Ещё ускорение (Сдвиг по маске внутри warp-а). `/04-reduction/06-warp-design-specific`.
7. Scan. Наивный алгоритм: получили data-race. `/05-scan/naive_cuda_scan.cu`.
8. Scan. Уберем data-race. Bank conflict. `/05-scan/correct_scan.cu`.
9. Scan. Решение bank conflict-а. `/05-scan/scan_bank_conflicts.cu`.

..to be done

# CUDA - 3

Рассмотрим 2 задачи:

- Задача Reduction -- агрегация данных, сумма элементов в массиве.
- Задача Scan -- сумма на префиксе, кумулятивная сумма.

### Reduction

- Классическое решение: N операций по регистрам, 1 поток.
- Параллельное решение на CPU: N/C операций по регистрам, C потоков.
- Параллельное решение на GPU: логика другая и рассмотрим её ниже.

#### Наивный алгоритм

/04-reduction/01-default-sum

Проведем подсчет внутри блока таким образом: 

<img width="600" alt="image" src="https://user-images.githubusercontent.com/36137274/197416803-50b31a23-ee07-4bb9-a42a-397a2c818d10.png">

В этом алгоритме каждый поток складывает свои числа: нулевой тред складывает элементы 0 и 1, первый -- 1 и 2 и тд. Итого за первый логический такт вычисляем сумму по 2 элемента, на следующем такте -- по 4 и тд. Время работы такого алогоритма: $\mathcal{O}(\text{Block size})$. Вспомним, что в вычислениях на видеокартах мы имеем дело блоками, которые состоят из warp-ов. Внутри warp вычисления происходят параллельно за один такт времени, внутри блока warp-ы выполняются последовательно. Тогда внутри блока подсчет пройдет за: $\mathcal{O}(\log(\text{Block size}) \cdot \dfrac{\text{Block size}}{\text{Warp size}})$.

Код. Если tid % (2 * s), то сложение:

```cuda
__global__ void Reduce(int* in_data, int* out_data) {
    extern __shared__ int shared_data[];

    unsigned int tid = threadIdx.x;
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    shared_data[tid] = in_data[index];
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_data[blockIdx.x] = shared_data[0];
    }
}
```

Так же тут добавили в параметрах ядра размер необходимой shared памяти:

```cuda
Reduce<<<num_blocks, block_size, sizeof(int) * block_size>>>(d_array, d_blocksum);
```

`extern __shared__ int shared_data[];` -- объявление shared памяти динамического размера. Используется, когда не можем во время компиляции определить требуемый размер shared памяти.

Запустим:

```bash
nvcc main.cu
./a.out
```

Итого:

```bash
1.15072 elapsed
4194304
```

Посчитаем количество warp-ов (warp size = 32) для block size = 256:

Итерация | Количество работающих wapr-ов
---------|------------------------------
1 | 8
2 | 8
3 | 8
4 | 8
5 | 8
6 | 4
7 | 2
8 | 1

Итого 47 warp-итераций. 

#### Уменьшим количество warp-итераций (изменена нумерация потоков):

/04-reduction/02-another-math

<img width="600" alt="image" src="https://user-images.githubusercontent.com/36137274/197417851-2880802a-2007-4534-b887-4ddb4638615a.png">

Используем 127 потоков и сложим по 2 элемента. 

Посчитаем количество warp-ов (warp size = 32) для block size = 256:

Итерация | Количество работающих wapr-ов
---------|------------------------------
1 | 4
2 | 2
3 | 1
4 | 1
5 | 1
6 | 1
7 | 1
8 | 1

Итого 12 warp-итераций. Поэтому ожидаем, что такой алгоритм должен работать быстрее прошлого варианта.

Код. Вычисляем номер индекса по `threadId`:

```cuda
__global__ void Reduce(int* in_data, int* out_data) {
    extern __shared__ int shared_data[];

    unsigned int tid = threadIdx.x;
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    shared_data[tid] = in_data[index];
    __syncthreads();
    
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        int i  = 2 * s * tid;

        if (i < blockDim.x) {
            shared_data[i] += shared_data[i + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_data[blockIdx.x] = shared_data[0];
    }
}
```

Запустим:

```bash
nvcc main.cu
./a.out
```

Итого:

```bash
0.883296 elapsed
4194304
```

*Bank conflict* - Поведение в shared-memory, когда два потока внутри одного warp-а записывают данные внутри разных кеш-линий по одному индексу (по модулю WS). Это приводит к вычислению за два такта вместо одного.

Тут так и происходит:

s = 1:

tid | i  | i + s 
----|----|-------
0   | 0  | 1
1   | 2  | 3

s = 2:

tid | i       | i + s 
----|---------|-------
0   | 2s*0=0  | 2
1   | 2s*1=4  | 6

...

s = 16:

tid | i   | i + s 
----|-----|-------
0   | 0   | 16
1   | 32  | 48
2   | 64  | 80

Пример: очередь в банк в котором есть 32 кабинки. В очереди сидят thread-ы, они идут в кабинку с номером i % 32. Тогда при s=16 нулевой поток, первый поток, второй поток -- все пойдут в нулевую кабинку.

#### Решим bank conflict:

04-reduction/03-improving-bank-conflicts

<img width="600" alt="image" src="https://user-images.githubusercontent.com/36137274/197418586-6b3c409c-90b4-4c5f-b1d1-52245215b9c5.png">

Тут те же 12 warp-итераций, но не ломают кэш-линию. Первый поток складывает себя и 128-го, второй складывает себя и 129-й и тд:

s = 128:

tid | Элементы 
----|---------
0   | 0 + 128
1   | 1 + 129
2   | 2 + 130

s = 64:

tid | Элементы | Фактически складывает
----|----------|----------------------
0   | 0 + 64   | 0 + 64 + 128 + 192
1   | 1 + 65   | 1 + 65 + 129 + 193
2   | 2 + 66   | 2 + 66 + 130 + 194

...

s = 16:

tid | Элементы | Фактически складывает
----|----------|----------------------
0   | 0 + 16   | 0 + 16 + 32 + 64 + ... + 240
1   | 1 + 17   | 1 + 17 + 33 + 65 + ... + 241
2   | 2 + 18   | 2 + 18 + 34 + 66 + ... + 242
... | ...      | ...
15  | 15 + 31  | 15 + 31 + 63 + 79 + ... + 255

Нет пересечений (в примере с банком -- все попали в свою кабинку и поэтому нет очереди внутри кабинки).

Код :

```cuda
__global__ void Reduce(int* in_data, int* out_data) {
    extern __shared__ int shared_data[];

    unsigned int tid = threadIdx.x;
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    shared_data[tid] = in_data[index];
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_data[blockIdx.x] = shared_data[0];
    }
}
```

Запустим:

```bash
nvcc main.cu
./a.out
```

Итого:

```bash
0.566368 elapsed
4194304
```

Буст не большой. Вообще так как shared memory -- быстрая память, то параллельное простаивание в shared memory менее заметно, чем в global memery.

#### Хаки

/04-reduction/03-improving-bank-conflicts. Количество блоков меньше в 2 раза (ILP = 2), shared data формируется по-другому: первое сложение осуществляем на global памяти, а не на shared памяти:

```
__global__ void Reduce(int* in_data, int* out_data) {
    extern __shared__ int shared_data[];

    unsigned int tid = threadIdx.x;
    unsigned int index = blockIdx.x * blockDim.x * 4 + threadIdx.x;

    shared_data[tid] = in_data[index] + in_data[index + blockDim.x] + in_data[index + blockDim.x * 2] + in_data[index + blockDim.x * 3];
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_data[blockIdx.x] = shared_data[0];
    }
}
```

Ускорение за счет обмена одной инструкции копирования на одно сложение на глобальной памяти. Копирование работает чуть дольше сложения, поэтому получаем выигрыш.

Запустим:

```bash
nvcc main.cu
./a.out
```

Итого:

```bash
0.412192 elapsed
4194304
```

/04-reduction/05-warp-reduce. Разворачиваем цикл. 

Когда один warp, то писать цикл `for` не нужно. Просто `A[i] += A[i+32]; A[i] += A[i+16]; A[i] += A[i+8];..`:

```cuda
__device__ void WarpReduce(volatile int* shared_data, int tid) {
    shared_data[tid] += shared_data[tid + 32];
    shared_data[tid] += shared_data[tid + 16];
    shared_data[tid] += shared_data[tid + 8];
    shared_data[tid] += shared_data[tid + 4];
    shared_data[tid] += shared_data[tid + 2];
    shared_data[tid] += shared_data[tid + 1];
}

__global__ void Reduce(int* in_data, int* out_data) {
    extern __shared__ int shared_data[];

    unsigned int tid = threadIdx.x;
    unsigned int index = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    shared_data[tid] = in_data[index] + in_data[index + blockDim.x];
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) {
        WarpReduce(shared_data, tid);
    }
    
    if (tid == 0) {
        out_data[blockIdx.x] = shared_data[0];
    }
}
```

При `volatile` shared память воспринимается как будто внутри одного warp-а, поэтому не надо делать `__syncthreads()`.

Запустим:

```bash
nvcc main.cu
./a.out
```

Итого:

```bash
0.396 elapsed
4194304
```

Работает примерно так же, но кажется скорость стала более устойчивая.

/04-reduction/06-warp-design-specific. Сдвиг по маске внутри warp-а:

`__shfl_down_sync(mask, val, offset)`:
 0 | 1 | 2 | 3 |...|30 |31
---|---|---|---|---|---|--
 1 | 2 | 3 | 4 |...|31 | 0
 
```cuda
__device__ void WarpReduce(volatile int* shared_data, int tid) {
    
    shared_data[tid] += shared_data[tid + 32];
	int val = shared_data[tid];
	val += __shfl_down_sync(-1, val, 16);
	val += __shfl_down_sync(-1, val, 8);
	val += __shfl_down_sync(-1, val, 4);
	val += __shfl_down_sync(-1, val, 2);
	val += __shfl_down_sync(-1, val, 1);
	shared_data[tid] = val;
}

__global__ void Reduce(int* in_data, int* out_data) {
    extern __shared__ int shared_data[];

    unsigned int tid = threadIdx.x;
    unsigned int index = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    shared_data[tid] = in_data[index] + in_data[index + blockDim.x];
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) {
        WarpReduce(shared_data, tid);
    }
    
    if (tid == 0) {
        out_data[blockIdx.x] = shared_data[0];
    }
}
```

Запустим, будет немного ругаться:

```bash
nvcc main.cu
./a.out
```

Итого:

```bash
0.861184 elapsed
4194304
```

Работает примерно так же.

### Задача Scan

https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/scan/doc/scan.pdf

Есть массив `A: A[0], A[1], A[2], ...`. Хотим посчитать:

- A[0]
- A[0] + A[1]
- A[0] + A[1] + A[2]

#### Наивный алгоритм:

05-scan/naive_cuda_scan.cu

<img width="600" alt="image" src="https://user-images.githubusercontent.com/36137274/197419294-80c6c6d7-b656-45d1-b6dc-fff26a300081.png">

Запустим:

```bash
nvcc naive_cuda_scan.cu
./a.out
```

Итого:

```bash
2.31587 elapsed
1024
```

Проблемы: 
- data-race при записи в shared память. Например при вычислении суммы 4-5 информация в 3 может быть перезписана, и там вместо значения 3 будет сумма 2-3. 
- Кроме того проблемы: $\mathcal{O}(N\log N)$ действий, $\mathcal{O}(N\log N)$ warp операций.

#### Уберем data-race. По степеням двойки сохраняем элементы:

05-scan/correct_scan.cu

Первая стадия:

<img width="600" alt="image" src="https://user-images.githubusercontent.com/36137274/197419553-76680c02-09d6-46f9-8af7-dfec878322c2.png">

Вторая стадия:

<img width="600" alt="image" src="https://user-images.githubusercontent.com/36137274/197419642-cb569b4e-68c0-429f-9af4-5a3292c4ace3.png">

Теперь $\mathcal{O}(N)$ действий. На лекции этот алгритм назывался аналогом дерева Фенвика.

Запустим:

```bash
nvcc correct_scan.cu
./a.out
```

Итого:

```bash
1.49805 elapsed
255
```

Проблемы:
- Сложно, тк две стадии
- Банк конфликт

Банк конфликт:

0-й поток вычисляет 0+1
0-й поток вычисляет 1+3
0-й поток вычисляет 3+7
0-й поток вычисляет 7+15
0-й поток вычисляет 15+31
16-й поток вычисляет 32+33

#### Решение: Пропускаем каждый 32-й элемент массива

05-scan/scan_bank_conflicts.cu

Делаем сдвиг, чтобы избежать банк конфликт: 

 0 | 1 | 2 | 3 |...|31 |32 |33 |34 |...|64 |65 |66 |
---|---|---|---|---|---|---|---|---|---|---|---|---|
 0 | 1 | 2 | 3 |...|31 | - |32 |33 |...|63 | - |64 |

# Seminar 4

Все программы по CUDA лежат тут: https://github.com/akhtyamovpavel/ParallelComputationExamples/tree/master/CUDA

# CUDA

На прошлом семинаре видели, что для отправки задач на видеокарту CUDA создаёт блоки подзадач, каждая из которых выполняется на микроядре SM процессора. В специальных параметрах запуска мы определяли, сколько блоков будем создавать (В Grid'e) и сколько подзадач в каждом блоке.

Хороший сайт с характеристиками видеокарт (google: gtx 1080 specs..):
- https://www.techpowerup.com/gpu-specs/geforce-gtx-1080.c2839

Ещё полезную информацию можно найти в английской википедии в разделе Technical Specification:
- https://en.wikipedia.org/wiki/CUDA. (Например, что warp size всегда был 32)

# Доступ к данным, ILP

На лекции говорилось, как строится работа с оперативной памятью GPU. Напомню - микропроцессоры получают данные партиями по варпам - то есть каждый варп (32 микропроцессора) получает подряд идущие данные из массива (a[i+1], a[i+2], ..., a[i+32]). По этой причине приходилось особым образом оптимизировать работу с данными, чтобы потоки действительно обращались к данным, которые к ним пришли.

В качестве эксперимента попробуем ILP - Instruction-level parallelism или параллелизм на уровне инструкций. Что это значит - за счёт знания архитектуры можно получить выигрыш за счет оптимизации.

Попробуем ускорить суммирование 2 массивов. На прошлом семинаре мы производили вычисления на каждом микроядре отдельно. Теперь вместо этого попробуем отдать одному микроядру сразу несколько ячеек. В чём профит - Количество блоков в очереди уменьшается пропорционально тому, сколько ячеек будет обрабатывать микроядро -> Количество инициализаций и подобных накладных расходов уменьшается -> количество обращений в память не изменяется, то есть выигрыш без особого проигрыша (пока количество блоков больше, чем SM процессоров). Конечно, учитывая, что именно расходы на память у нас пока самые большие, то выигрыш будет довольно мал, но наша цель немного в другом.

Итак, основная задача - проверить, действительно ли случайный доступ к памяти внутри варпа ухудщает работу. Для этого сначала возьмём наш код для main из предыдущего семинара, добавим зависимость от ILP (так как каждый поток теперь считает N ячеек, то соответственно, и блоков нам надо в ILP раз меньше)

## Пример 1: ломаем кэш-линию

Работаем с /03-memory-model/01-uncoalesced-access/full_uncoalesce.cu.

main:

```cu
#define ILP 8

// .. функции смотри ниже

int main() {
	int N = 1 << 28;
	size_t size = N * sizeof(float);
	float *x = (float*)malloc(size);
	float *y = (float*)malloc(size);
    
    cudaSetDevice (5);

	float *d_x, *d_y;

	cudaMalloc(&d_x, size);
	cudaMalloc(&d_y, size);


	for (int i = 0; i < N; ++i) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}


	cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);

	int blockSize = 256;

	int numBlocks = (N + blockSize - 1) / blockSize;

    cudaEvent_t start;
    cudaEvent_t stop;

    // Creating event
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    cudaEventRecord(start);
	add<<<numBlocks / ILP, blockSize>>>(N, d_x, d_y);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << milliseconds << " elapsed normal" << std::endl;

	cudaFree(d_x);
	cudaFree(d_y);
	free(x);
	free(y);
	return 0;
}

```

Теперь передаём в add numBlocks/ILP блоков. Рассматрим два ядра - одно правильное и одно не правильное. Идея для порчи скорости - требуется доступ к памяти, которая не совпадает с той, которая на текущий момент пришла к потоку. Мы испортим так: будем стучаться в следующую ячейку вместо оптимальной.

Рассмотрим две реализации:

```cu
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

Как видим -- мы заставляем текущий поток обрабатывать не только текущий элемент, но и ровно последующий. Блоков в ILP раз меньше, но при этом каждый поток обрабатывает сразу несколько ячеек массива.

Теперь сделаем так, чтобы на warp приходили последовательные данные (просто мнимо увеличим в 2 раза блок и будем работать со сдвигом размером в обычный блок).

```cu
__global__
void add(int n, float* x, float* y, float* z) {
    int tid = threadIdx.x + ILP * blockDim.x * blockIdx.x;
    for (int i = 0; i < ILP; ++i) {
        int current_tid = tid + i * blockDim.x;
        
        z[current_tid] = 2.0f * x[current_tid] + y[current_tid];
    }
}
```

Запускам:

```bash
nvcc full_uncoalesce.cu
./a.out
```

Итого:
```
12.226 elapsed normal
42.5975 elapsed stupid
```

# Пирамида памяти CPU vs GPU, Shared memory

<img width="600" alt="image" src="https://user-images.githubusercontent.com/36137274/194471891-cdea4874-84a6-4f13-94d4-b1f72cf4b0c0.png">

<img width="600" alt="image" src="https://user-images.githubusercontent.com/36137274/194471923-9e1f5e94-a1cf-4fe8-87d9-842c380eb027.png">

Видим те же L1 и L2 кэш у видеокарты. Ко второму мы не имеем доступ, а доступ к первому неожиданно (в отличие от CPU) есть.

Сравнение:

<img width="600" alt="image" src="https://user-images.githubusercontent.com/36137274/194471976-18a409d7-f937-4db1-b0d2-823c3cc2c8eb.png">

Смотрим на схему GPU. Как видим, тот путь, который мы вечно просили проделывать всем данным шёл из Device memory до регистров микроядер. И составлял он в среднем около 500-600 тактов времени. Да и ширина шины памяти довольно не большая, из-за чего скорость получаем совсем маленькую. 

А вот на L1 и L2 кэше на GPU быстрее, чем на CPU. 80 тактов на доступ, а скорости исчисляем в терабайтах. В общем мы можем напрямую управлять L1 памятью. Однако, есть пара загвоздок:

- L1 память у каждого SM процессора своя и никак не пересекается с памятью соседнего (в принципе у ядер процессоров в последнее время так же). А это значит, что каждый блок будет видеть и использовать только свою L1 память (помним, что один блок задач ложиться ровно на один SM процессор)
- L1 память организована с матричным доступом по машинным словам (слово, это 128 байт, но в примере только 10 ячеек). Что это значит? Нууу, представим большой склад, шириной 32 секции и длинной... около 2'000 секций. Да, по факту очень вытянутый в длину склад. Так вот, для работы с этим складом у нас есть только 32 клешни, каждая катается по своим рельсам, расположенным вдоль склада на 2'000 метров. То есть в один момент времени нам могут подвезти только 32 ячейки данных, каждая из которых лежала под своим рельсом.

![image](https://user-images.githubusercontent.com/36137274/194472608-5a34c95e-4a5e-4f70-a0c2-c8a282cb3090.png)

В общем, если 2 потока попытаются прочитать данные из 1го слова B3 ячейки и 3го слова B3 ячейки, то им придётся ждать 2 итерации передачи данных, так как клешне придётся 2 раза отправиться за данными. А вот если стучаться в B3 ячейку и B6 ячейку, даже с учётом, что они стоят не в одной сроке, то операция чтения будет одна и данные придут одновременно. В принципе всё, осталось только понять, как проектировать алгоритмы, что бы процесс чтения и записи данных действительно был быстрым и не мешал друг другу.

В целом идея такая: если мы активно используем данные с оперативной памяти внутри одного блока, то данные лучше перекинуть в shared память и затем использовать их из shared памяти.

## Пример 2: 

Работаем с /03-memory-model/02-shared/*_example.cu

Считаем сумму элемента и его соседей слева и справа.

Сравним две реализации:

Только global:

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

Реализация через shared memory:

Используем ключевое слово `__shared__`.  Модификатор `__shared__` говорит, что выделяется память внутри блока. То есть говорм, что данную переменную или массив нужно хранить на L1 кэше, а не в глобальной памяти. Выделяем память по размеру равную `BLOCKSIZE`.

Первоначальный план таков - мы запишем данные для текущей ячейки в shared память, дождёмся записи, а потом будем в лоб брать сумму ещё со следующей ячейкой, в которой так же лежат полезные данные. Главное уточнение - shared память существует только в рамках одного блока. То есть, если процесс находится уже в другом блоке, он не сможет обратиться к памяти предыдущего, у него она будет своя.

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

Тут есть небольшой выигрыш за счёт интенсивного использования shared memory, но небольшой, потому что пример очень искуственный.

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

Вспомним, что Grid и Блоки можно делать многомерными. В домашнем задании надо будет к этому коду добавить shared memory. Используем это в задаче перемножения матриц:

Просто вспомогательный код: первая функция -- генератор диагональных матриц, вторая - печать матрицы по координатам.

```cu
#include <iostream>

#define BLOCK_SIZE 256


void FillMatrix(float* matrix, int height, int width) {
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			if (i == j) {
				matrix[i * width + j] = 1;
			} else {
				matrix[i * width + j] = 0;
			}
		}
	}
}

void PrintMatrix(float *matrix, int height, int width) {

	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			std::cout << i << " " << j << " " << matrix[i * width + j] << "\n";
		}
	}
}
```

Далее, для создания многомерных блоков или гридов используется специальная переменная dim3. Тогда объявления 3-х мерных (или 2-х мерных в нашем случае) грида и блока будет примерно таким:

```cu
dim3 num_blocks(8, 16);
dim3 block_size(16, 16);
```

Где в скобочках указаны размеры и блока и грида. Внимание, для блока действует ограничение, что суммарное количество потоков  в нём не превышает 1024. То есть не получится создать блок размера 32*32*32, так как это 32768 > 1024. Третья размерность так же не может быть больше 64.

Запишем main:

```cu
int main() {

	float *h_A;
	float *h_B;
	float *h_C;

	h_A = new float[128 * 384];
	h_B = new float[384 * 256];
	h_C = new float[128 * 256];

	FillMatrix(h_A, 128, 384);
	FillMatrix(h_B, 384, 256);

	float* d_A;
	float* d_B;
	float* d_C;

	cudaMalloc(&d_A, sizeof(float) * 128 * 384);
	cudaMalloc(&d_B, sizeof(float) * 384 * 256);
	cudaMalloc(&d_C, sizeof(float) * 128 * 256);

    cudaMemcpy(d_A, h_A, sizeof(float) * 128 * 384, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * 384 * 256, cudaMemcpyHostToDevice);

    // kernel call
    dim3 num_blocks(8, 16);
    dim3 block_size(16, 16);

    MatrixMul<<<num_blocks, block_size>>>(d_A, d_B, d_C, 384);

    cudaMemcpy(h_C, d_C, sizeof(float) * 128 * 256, cudaMemcpyDeviceToHost);
    PrintMatrix(h_C, 128, 256);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	delete[] h_A;
	delete[] h_B;
	delete[] h_C;

	return 0;
}
```

Двемерную матрицу записали в виде одномерного массива. Далее как раньше: запускаемся от num_blocks и block_size и работаем как обычно.

Теперь самое интересное - ядро. Так как у нас теперь двумерный объект, то можно этим воспользоваться для удобной "навигации" по матрице. Смотрите. Размер матрицы С 128*256, а размеры блока и грида 8*16 и 16*16 соответственно (как раз 8*16=128, 16*16=256).

То есть мы можем каждую подзадачу в каждом блоке выделить на одну ячейку матрицы:

```cu
__global__
void MatrixMul(float* A, float* B, float* C, int mid_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int height = blockDim.x * gridDim.x;
    int width = blockDim.y * gridDim.y;

    C[i * width + j] = .0f;

    for (int k = 0; k < mid_size; ++k) {
        C[i * width + j] += A[i * mid_size + k] * B[k * width + j];
    }
}
```

Координаты `i, j` - это положение блока в матрице плюс положение подзадачи внутри блока. Тем самым `i,j` являются настоящими координатами элемента, который требуется обработать в матрице. И тогда элемент, который обрабатывается потоком номер (1,2) в блоке номер (2,2) будет:

```cu
i = 2*blockDim + 2
j = 2*blockDim + 1
```

То есть сначала сдвигаемся по блокам (с шагом blockDim), а потом внутри блока по элементам, выбирая нужный.

## CUBLAS

06-cublas/03-cosine-distance.cu

Есть библиотека BLAS -- классическая библиотека для линейной алгебры (Basic Linear Algebra Subprograms). CUBLAS -- то же, но на CUDA-е.

```
cublasCreate(...) -- вызываем конструктор класса,
cublasSetVector(...) -- аллоцируем массив, замена cudaMemcpy,
cublasSdot(...) -- вычисление скалярного произведения,
cublasSnrm2(...) -- вычисление l2 нормы.
```

## PyCuda

07-pycuda/01-simple-ariphmetic.py

Можно писать и на Python. Запустим программу:

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

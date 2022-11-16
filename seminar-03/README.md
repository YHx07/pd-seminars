Форма для отзывов: https://forms.gle/kkcPJtXRsJKmKyNo8. Также, я буду благодарен за Merge Request-ы в этом репозитории.

# Seminar 3

Все программы по CUDA лежат тут: https://github.com/akhtyamovpavel/ParallelComputationExamples/tree/master/CUDA

# CUDA

CUDA -- Compute Unified Device Architecture. Мы будет использовать CUDA Toolkit для ускорения вычислений программ, написанных на C++, на видеокартах Nvidia. Инструкция на [официальном сайте Nvidia](https://developer.nvidia.com/how-to-cuda-c-cpp).

## GPU

GPU -- Graphics Processing Unit -- видеокарта.

Изначально использовалось для графики (рендеринг, обратная трассировка лучей).

Идея: вектор содержит много чисел и все элементры вектора подвергаются одной операции. Видеокарта позволит произвести эти однотипные вычисления за один такт процессора.

CUDA Toolkit можно использовать и для вычислений на TPU -- Tensor Processing Unit. Почитать про сравнение CPU, GPU, TPU можно [на cloud.google](https://cloud.google.com/tpu/docs/intro-to-tpu), или [на Habr-е на русском](https://habr.com/ru/post/422317/), или посмотреть [интерактивную презентацию Google](https://storage.googleapis.com/nexttpu/index.html).

## Кластер с GPU

Подключиться на кластер: `ssh <login>@lorien.atp-fivt.org`. Логин и пароль должны были прийти на почту. В остальном -- смотри инструкцию GPU-сервер АТП.

### Некоторые характеристики кластера:

- RAM: 256 Gb,
- CPU: Intel Xeon Gold 6136 v4, 24 ядра,
- GPU: Nvidia GeForce RTX 2080, 8 видеокарт, видеопамять на каждой 11 Gb.

- ~ 3 Tb HDD и ~500 Gb SSD.

На кластере уставновлена CUDA 10.1.

NVIDIA System Management Interface -- отображение информации обо всех доступных графических процессорах и процессах, использующих их:

```bash
- nvidia-smi
```

### Первые шаги на кластере:

1. [Создать](https://www.ssh.com/academy/ssh/keygen) себе ssh ключ (ssh-keygen),
2. Добавить .ssh/id_rsa.pub к себе на гитхаб в ключи: https://github.com/settings/keys
3. git clone https://github.com/akhtyamovpavel/ParallelComputationExamples

[Небольшой лайфхак](https://www.ssh.com/academy/ssh/copy-id) как входить на кластер и не вводить каждый раз пароль.

## Программирование на CUDA

- Thread
- Warp -- набор потоков, физически отрабатываемых за один такт времени.
- Block -- набор потоков, логически отрабатываемых в одну единицу времени, block-и делятся на warp-ы в физическом исполнении. Программируя мы имеем дело с block-ами, с точки зрения кода warp-ы не видим.
- Grid -- набор из блоков, количество блоков в grid-е задается пользователем.
- Streaming Multiprocessor (SM) -- аналог физических ядер. Один блок обрабатывается одним SM.

![image](https://user-images.githubusercontent.com/36137274/193191711-56f2a262-45b4-45e2-8d83-1b13557cc03c.png)

Пример с лекции: Например всего на видеокарте 2560 ядер и 20 SM => один SM управляет 128 ядрами. Один warp состоит из 32 потоков, поэтому один SM управляет 4 warp-ами.

## Запуск программы на CUDA

### 00-hello-world-single-thread

Программа на C++, пока без CUDA. Тут складываем два вектора x и y: y = x + y, сложение происходит их на одном ядре:

```C++
#include <iostream>
#include <cmath>

void add(int n, float* x, float* y) {
	for (int i = 0; i < n; ++i) {
		y[i] = x[i] + y[i];
	}	
}


int main() {
	int N = 1 << 28;
	float* x = new float[N];
	float* y = new float[N];

	for (int i = 0; i < N; ++i) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	add(N, x, y);
	
	float maxError = 0.0f;
	for (int i = 0; i < N; i++) {
		maxError = fmax(maxError, fabs(y[i]-3.0f));
	}
	std::cout << "Max error: " << maxError << std::endl;
	delete [] x;
	delete [] y;
	return 0;
}
```

### Скомпилируем и замерим время:

```bash
1. g++ main.cpp 
2. time ./a.out
```

Видим, что максимальная ошибка равна нулю. Запомним время -- у меня получилось 3.6сек.

### 01-hello-world-single-cuda-thread

Попробуем CUDA. В чём отличие между программами? Первое -- main.cpp и main.cu, теперь используем CUDA и имеем другой синтаксис:

```cu
#include <iostream>
#include <cmath>

__global__
void add(int n, float* x, float* y) {
	for (int i = 0; i < n; ++i) {
		y[i] = x[i] + y[i];
	}	
}


int main() {
	int N = 1 << 28;
	float *x, *y;
 
 cudaSetDevice(5)

	cudaMallocManaged(&x, N * sizeof(float));
	cudaMallocManaged(&y, N * sizeof(float));


	for (int i = 0; i < N; ++i) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	add<<<1, 1>>>(N, x, y);

	cudaDeviceSynchronize();	
	float maxError = 0.0f;
	for (int i = 0; i < N; i++) {
		maxError = fmax(maxError, fabs(y[i]-3.0f));
	}
	std::cout << "Max error: " << maxError << std::endl;

	cudaFree(x);
	cudaFree(y);
	return 0;
}
```

1. __global__ -- специальный макрос, который показывает, что функция может запускаться как на GPU, так и на CPU. 

Конструкция называется ядром:
```
__global__
void add
```

2. Память выделяется с помощью `cudaMallocManaged(void* array, size_t size)` (есть еще просто `cudaMalloc`, но с ним сложнее работать; `cudaMallocManaged` создает массив сразу и на CPU, и на GPU).

3. Появилось `<<<1, 1>>>` -- это гиперпараметры вызова ядра -- `__global__void add` из начала кода. То есть в `<<<Grid_size, Block_size>>>` указываем гиперпараметры вызова ядра, а в `()` обычные параметры вызова функции. Один block обрабатвается одним ядром видеокарты. По сути именно на моменте `add<<<1, 1>>>(N, x, y)` начинается работа с CUDA -- массивы копируются на видеокарту и затем складываются на ней.

4. `cudaDeviceSynchronize()` -- синхронизация ядер и потоков между собой, по сути это барьер. Так как работаем между GPU и CPU, и так как все операции на GPU асинхронные, то чтобы считать содержимое массивов нам нужно поставить барьер, и убедиться что когда посчитался массив, он перекинулся на CPU.

5. `cudaSetDevice(5)` - указываем на какой видеокарте мы будем выполнять расчёт.

6. `cudaFree(x)` - по аналогии с malloc освобождение памяти на видеокарте после вычислений.
 
### Скомпилируем и замерим время (используем компилятор nvcc -- обертка над gcc):

```bash
1. nvcc main.cu
2. time ./a.out
```

Запустим команду, чтобы посмотреть на работу видеокарт в реальном времени (чтобы выйти: ctrl+C):
```bash
watch -n 0.1 nvidia-smi
``` 

Стало работать медленнее -- у меня 27.7cек.

### Немного про грядки:

![image](https://user-images.githubusercontent.com/36137274/193194317-60ce61d5-f6bd-4432-ae02-9723e7f524ae.png)

Как видите, у нас здесь присутствует какое то огромное количество мелких компонент. Сразу бросается в глаза L2 кэш, но он не нужен нам. Обратим внимание на зелёные кубики. Если присмотреться, то каждый из них подписан как SM процессор. Это по факту - основные ядра видеокарты. Видим что их много.

![image](https://user-images.githubusercontent.com/36137274/193194454-c442c78c-6099-4e5d-932c-2a682d4fb124.png)

Каждый SM процессор так же состоит из 4 блоков и L1 кэша, а каждый блок ещё состоит из 34 компонент: 2 тензорных ядра, 16 для целочисленных вычислений и 16 для вычислений с плавающей точкой. Правда, компоненты для вычислений целых чисел и с плавающей точкой отвечают за разные задачи, так что можно их считать как 1 компоненту. В общем, такие блоки называются варпами (можно увидеть надпись warp scheduler - компонента, которая управляет частями варпа). Эти маленькие блоки внутри варпа - микроядра, которы и проводят вычисления. Можно даже подсчитать сколько их в итоге:

68 SM процессоров * 4 варпа * 16 ядра в каждом = 4352. Причём для целых чисел и ещё столько же для чисел с плавающей точкой. Это точно не сравниться с 8 ядрами вашего процессора. Можем посмотреть что всего у ядер у видеокарты RTX 2080 TI: их 4352. Mupltiprocessor count: 68 => 4352 / 68 = 64 подъядра в каждом мультипроцессоре.  

Вернёмся к понятиям Grid_size и Block_size. У нас есть SM процессор, у которого 64 микроядра для вычислений. И вспоминаем OpenMP, в котором можно было разбивать код на таски, становящиеся в очередь на выполнение потоками. Здесь будем действовать по аналогии - создадим логическую единицу Block - это таска, которая становится в очередь на выполнение SM процессором. То есть Блоки, это логически пакеты задач, которые берут на себя SM процессоры из стека. Такой стек и называется Grid. В общем:

- Block - логический пакет задач, который выполняется одним SM процессором,
- Grid - Стек блоков, из которого они и распределяются по видеокарте.

Блоки и гриды могут быть 2х и 3х мерными. Физически это мало что меняет, но добавляет целый ряд удобств для работы с многомерными данными (а видеокарты обычно работают с 3D пространством).

В итоге Grid_size и Block_size:

- Block_size - количество подзадач внутри блока, которые и будут выполнены на микроядрах SM процессора,
- Grid_size - количество блоков, которые будут созданы планировщиком для вычислений.

Осталось лишь понять, как именно распределяются подзадачи блока по микроядрам SM процессора. И тут всё довольно просто - блок разбивается по группам в 32 подзадачи, а потом каждая группа отправляется в один из варпов на вычисление. Причём суммарное количество подзадач может быть больше количества варпов*32. Они просто встанут в очередь.

### 02-add-threads

Имеем 2 параметра в <<<>>>, которые означают количество потоков, на которое мы распределяем задачу. Давайте пробовать ускорять её:

```cu
#include <iostream>
#include <cmath>

__global__
void add(int n, float* x, float* y) {
	int index = threadIdx.x;                  //Новое: добавили "индекс" пачки
	int stride = blockDim.x;                  //Новое: добавили размер пачки

	for (int i = index; i < n; i += stride) {
		y[i] = x[i] + y[i];
	}	
}


int main() {
	int N = 1 << 28;
	float *x, *y;

	cudaMallocManaged(&x, N * sizeof(float));
	cudaMallocManaged(&y, N * sizeof(float));


	for (int i = 0; i < N; ++i) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	add<<<1, 256>>>(N, x, y);

	cudaDeviceSynchronize();	
	float maxError = 0.0f;
	for (int i = 0; i < N; i++) {
		maxError = fmax(maxError, fabs(y[i]-3.0f));
	}
	std::cout << "Max error: " << maxError << std::endl;

	cudaFree(x);
	cudaFree(y);
	return 0;
}
```

Как видим, тут добавился размер блока в 256 элементов. Что это значит? Что задача add будет выполнена на одном SM процессоре на всех 64 вычислительных микроядра 4 раза (так как подзадач 256, а вычислителей только 64). А теперь так же заметьте - у нас изменилась функция add. Теперь здесь не цикл по всему массиву, а цикл по элементам, номера которых кратны номеру потока в блоке (если что, эти номера логические и измеряются от 0 до block_size-1).

### Скомпилируем и замерим время (используем компилятор nvcc -- обертка над gcc):

```bash
1. nvcc main.cu
2. time ./a.out
```

Стало работать чуть быстрее -- у меня 6.8cек, то есть относительно задачи 01 прирост в 4 раза. Стандартный код на CPU всё ещё быстрее - 4 секунды, но как минимум у нас теперь есть ещё одно вычислительное устройство, которое работает параллельно CPU с той же скоростью, уже не плохо).

### 03-add-blocks

Теперь воспользуемся тем, что у видеокарты не 1 SM процессор, обрабатывающий блок, а 68.

Добавим еще одну абстракцию -- количество блоков в grid: numBlocks = (N + blockSize - 1) / blockSize (сколько блоков надо (с запасом) делим размер массива на размер блока, и добавляем чуть сверху в случае, если оно целиком не влезает). Раньше на этом месте стояла 1, поэтому в единицу времени обрабатывался 1 блок. Теперь в одну единицу времени можем обрабатывать столько блоков, сколько может видеокарта. 


```cu
#include <iostream>
#include <cmath>

__global__
void add(int n, float* x, float* y) {
	int index = blockIdx.x * blockDim.x + threadIdx.x; //Новое: добавили поправки на размер блоков
	int stride = blockDim.x * gridDim.x;               //Новое: добавили поправки на размер блоков

	for (int i = index; i < n; i += stride) {
		y[i] = x[i] + y[i];
	}	
}


int main() {
	int N = 1 << 28;
	float *x, *y;

	cudaMallocManaged(&x, N * sizeof(float));
	cudaMallocManaged(&y, N * sizeof(float));


	for (int i = 0; i < N; ++i) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	int blockSize = 256;

	int numBlocks = (N + blockSize - 1) / blockSize; // Новое: размер блока

	add<<<numBlocks, blockSize>>>(N, x, y);

	cudaDeviceSynchronize();	
	float maxError = 0.0f;
	for (int i = 0; i < N; i++) {
		maxError = fmax(maxError, fabs(y[i]-3.0f));
	}
	std::cout << "Max error: " << maxError << std::endl;

	cudaFree(x);
	cudaFree(y);
	return 0;
}
```

### Скомпилируем и замерим время (используем компилятор nvcc -- обертка над gcc):

```bash
1. nvcc main.cu
2. time ./a.out
```

У меня получилось 8секунд, то есть медленее задачи 02. 

Проблема в том, что мы не эффективно копируем на GPU память. То есть механизмы cudaMallocManaged периодически производят синхронизацию памяти между видеокартой и оперативкой, что сильно замедляет процесс вычислений, так как происходит простаивание.


### Отшлифовка экспериментов

Для проверки гипотезы, что разница маленькая из-за копирования памяти с GPU и на неё, можем провести эксперимент - добавить сложности в вычисления, не зависящей от памяти. Причём всё довольно просто - добавим в ядра огромный цикл, который будет вычислять сумму этих 2 элементов 1'000 раз и посмотрим теперь на время работы.

Для второго примера:
```cu
__global__
void add(int n, float* x, float* y) {
	int index = threadIdx.x;
	int stride = blockDim.x;
	int sum = 0;

	for (int i = index; i < n; i += stride) {
		for (int j = 0; j < 1'000; j++ )
			sum+= x[i] + y[i];

		y[i] = sum;
	}
}
```

Для третьего примера новое ядро:
```cu
__global__
void add(int n, float* x, float* y) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	int sum = 0;

	for (int i = index; i < n; i += stride) {
		for (int j = 0; j < 1'000; j++ )
			sum += x[i] + y[i];

		y[i] = sum;
	}
}
```

В итоге можем заметить, что второй пример замедлился раз в 5-6, в то время как третий пример выдал те-же результаты по времени. Отсюда можно сделать вывод, что нам хватает вычислительных мощностей и замедление происходит в ином месте.

### 04-memcpy

Давайте тогда попробуем решить проблему с копированием памяти, хотя бы частично.

В CUDA есть функция `cudaMalloc`:
- плюсы: Она копирует память сразу всю и не пытается синхронизировать данные в оперативной памяти процессора с данными на видеокарте
- минусы: Придётся напрямую заняться синхронизацией, то есть буквально копировать данные с видеокарты и обратно командами

Для копирования данных нам потребуется функция `cudaMemcpy(Target, Source, size, cudaMemcpyHostToDevice/cudaMemcpyDeviceToHost)`
- target - адрес памяти, куда копируются данные
- source - адрес памяти, откуда копируются данные
- size - количество данных в байтах
- cudaMemcpyHostToDevice/cudaMemcpyDeviceToHost - две константы, которыми определяется направление работы функции

Изменим main, оставив наше ядро аналогичным:

```cu
#include <iostream>
#include <cmath>
#include <cstdio>


__constant__ int device_n;


__global__
void add(int n, float* x, float* y) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
    
//    if (threadIdx.x == 0) {
//        printf("%d %d %d\n", blockIdx.x, gridDim.x, blockDim.x);
//    }

	for (int i = index; i < n; i += stride) {
		y[i] = x[i] + y[i];
	}	
}


int main() {
	int N = 1 << 28;
	size_t size = N * sizeof(float);
	float *h_x = (float*)malloc(size);
	float *h_y = (float*)malloc(size);

	float *d_x, *d_y;

	cudaMalloc(&d_x, size);
	cudaMalloc(&d_y, size);


	for (int i = 0; i < N; ++i) {
		h_x[i] = 1.0f;
		h_y[i] = 2.0f;
	}


	cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);

	int blockSize = 256;

	int numBlocks = (N + blockSize - 1) / blockSize;

	add<<<numBlocks, blockSize>>>(N, d_x, d_y);

	// cudaDeviceSynchronize();	
	cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost);

	float maxError = 0.0f;
	for (int i = 0; i < N; i++) {
		maxError = fmax(maxError, fabs(h_y[i]-3.0f));
	}
	std::cout << "Max error: " << maxError << std::endl;

	cudaFree(d_x);
	cudaFree(d_y);
	free(h_x);
	free(h_y);
	return 0;
}
```

Как видим, одно из главных изменений - теперь мы отдельно выделяем память на CPU и на GPU. Важно - мы выделяем память в байтах, как в принципе и с помощью обычного malloc. А так же, теперь нужно переносить данные из одной памяти в другую, чем мы и занимаемся в 3 вызовах cudaMemcpy. И как обычно, в конце теперь нам придётся освобождать и память CPU и память GPU.

Итак, запускаем, смотрим... и снова никакого эффекта, время не уменьшилось в рамках погрешности. Однако, это было не зря, так как теперь мы можем использовать функцию nvprof. Это профилировщик, который поможет нам оценить время, затраченное каждой функцией.

После компиляции прямо запускаем:

```
nvprof ./a.out 
```

И смотрим на затраченное время. Особенно советую обратить внимание на время, затраченное на копирование данных на GPU и копирование данных обратно. Оно будет примерно 95-98% времени общей работы. Само ядро же будет отрабатывать только около 5 ms (без мусорного цикла для загрузки). В общем, отсюда крайне рекомендую извлечь один важный момент - перед вычислениями стоит копировать данные целиком и стараться не трогать их, так как это относительно дорогая операция.

## Замеры времени

В д/з, придётся считать время работы программы. Можно это делать nvprof, но такое себе. Поэтому мы изучим, как CUDA может замерять время работы программы. Механизм довольно не тривиален, но в целом очень похож на подобные.

Давайте пройдёмся по существующим для этого функциям:

- `cudaEvent_t start` - создаёт переменную, где будет хранится замер времени в определённый момент... собственно времени. В общем это аналог timestamp, который хранит время, прошедшее с некоторого момента (момент всегда для всех одинаковый). Однако хранит она это время по особенному, так как процессы асинхронны, и их тысячи.

- `cudaEventCreate(&start)` - здесь инициализируем переменную событий, тонко намекая CUD'e, что мы хотим сделать.

- `cudaEventRecord(start)` - записывает текущий момент времени в переменную start

- `cudaEventSynchronize(stop)` - вот тут и начинаются отличия. Так как у нас почти все операции и ядра асинхронны, то требуется дождаться, пока все вычисления закончатся. В итоге, после записи временной метки, нам ещё требуется выровнять её по последнему потоку, который закончил вычисления. Именно этим и занимаемся)

- `cudaEventElapsedTime(&milliseconds, start, stop)` - после создания timestamp, нужно превратить их в миллисекунды времени. То есть тут мы записываем в milliseconds время, прошедшее с момента start до момента stop.

### 02-device-specs-benchmarks/00-time-measurement

```cu
#include <iostream>
#include <cmath>

__global__
void add(int n, float* x, float* y) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < n; i += stride) {
		y[i] = x[i] + y[i];
	}	
}


int main() {
	int N = 1 << 28;
	size_t size = N * sizeof(float);
	float *x = (float*)malloc(size);
	float *y = (float*)malloc(size);

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
	add<<<numBlocks, blockSize>>>(N, d_x, d_y);

    // cudaEventRecord(stop);

	cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;

    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << milliseconds << " elapsed" << std::endl;

	cudaFree(d_x);
	cudaFree(d_y);
	free(x);
	free(y);
	return 0;
}
```

```bash
1. nvcc main.cu
2. time ./a.out
```

У меня получилось 355мс -- время работы программы.

### Материалы:

- [CUDA в примерах (рус)](https://cloud.mail.ru/public/DCPf/aCk7BnMTJ)
- [Developer.nvidia.com](https://developer.nvidia.com/accelerated-computing-training)
- [Sample Code](https://github.com/nvidia/cuda-samples)
- [Seminar_CUDA_1 -- Chernetskiy](https://gitlab.com/fpmi-atp/pd2022a-supplementary/chernetskiy/-/blob/main/Seminar_3_CUDA_1.md)

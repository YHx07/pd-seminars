[in progess]

# Суть за счёт чего работает ускорение вычислений

Давно известно, что видеокарты (GPU) гораздо быстрее делают математические вычисления, нежели процессоры (CPU).  Все дело в том, что в этих операциях большое значение играют количество ядер и параллельность расчетов. В видеочипах ядер намного больше, чем в CPU.

Если брать CUDA (архитектура параллельных вычислений, позволяет увеличить вычислительную производительность благодаря использованию графических процессоров фирмы Nvidia), то в одной видеокарте их может быть до 3072  штук!

Эта особенность связана с ролью видеокарты в компьютере – ей необходимо делать множество простых действий параллельно и за очень короткое время.

# Классификация видеокарт

## Линейки

Семейство GeForce -- это семейство является самым основным у компании NVIDIA. 
Ее представители ставятся как в мощные игровые ПК, так и в простенькие офисные ноутбуки. 
Видеокарты из этого семейства удовлетворяют 90% потребностей простых потребителей. 
Остальные семейства созданы для энтузиастов, для профессионалов и корпоративного сегмента или вовсе для достаточно необычных на первый взгляд задач.

TITAN -- подлинейка GeForce. О подлинейке TITAN смотри [ниже](#TITAN)

Семейство Quadro -- это семейство предназначено для профессионального использования.  
Эти карты очень хорошо подойдут для сложных 3D-приложений и вычислительных симуляций. 
На этих картах производится рендеринг настоящих фильмов со спецэффектами. Эти карты не подходят для игр. 
Даже самые дорогие решения будут проигрывать средним игровым видеокартам GeForce. 
Все дело в том, что эти видеоадаптеры рассчитаны на вычисления с плавающей запятой двойной точности (FP 64), а играм достаточно и одинарной (FP 32).

В более точных вычислениях и приложениях, использующих OpenGL* драйвера, Quadro превосходит игровые видеокарты (за исключением некоторых Titan).

Семейство Tesla

Семейство Tesla – это семейство, созданное специально для ускорения математических вычислений. 
Такие карты хорошо справляются с как с FP 32, так и с FP 64 расчетами. 
Их используют в научных центрах и на серверах, ведь на единицу потребленной энергии они сделают больше полезной работы, нежели процессор. 
Интересный факт: в картах этой линейки нет видеовыходов. 
Первая буква означает поколение.

Семейство NVS

Это семейство создано для корпоративного сегмента. 
Раньше оно было частью семейства Quadro и обозначалось также буквами «NVS». 
Эти видеочипы созданы для бизнес-приложений (финансовых, корпоративных), многомониторных решений.

Например, их используют для цифровых информационных панелей. 
Их особенностями являются большое количество портов для подключения дисплеев в некоторых моделях и очень низкая общая стоимость поддержки (ТСО).

Производительностью они не блещут и в них используется DDR3. Тепловыделение не превышает 70 Вт.

Семейство Tegra

Семейство  систем на кристалле (SoC) для мобильных устройств. 
В рамках него были представлены первые двухъядерные и четырехъядерные решения. 
Во времена своего выхода являются топовыми решениями в плане графики, но и в процессорной части дела обстоят довольно хорошо. 
На данный момент у них есть свои разработки ядер Denver, вместо «классических» Cortex.

Есть две версии Tegra K1:

2 ядра Denver
4 ядра Cortex-A15
Tegra K1 был построен на микроархитектуре Kepler, а Tegra X1 на Maxwell.

Преимуществом является то, что есть эксклюзивные проекты и портированные компьютерные игры, сделанные только под устройства на базе Tegra, ввиду их мощности и связей компании. 
У NVIDIA так же есть свои планшеты и портативные косоли, где реализованы некоторые интересные технологии. Например, трансляция игр с ПК на экран своего мобильного устройства на базе Tegra. 
Устройства на базе Tegra являются хорошим подспорьем для мобильного гейминга.

---

## Поколения семейства GeForce

Поколение и микроархитектура видеочипа отражены в его кодовом названии. Так:

- GF – Fermi
- GK – Kepler
- GM – Maxwell
- GP – Pascal

NVIDIA GeForce 400 — линейка графических процессоров, основанная на архитектуре NVIDIA Fermi, первая в арсенале компании, обладающая поддержкой DirectX 11. 
DirectX — это «прослойка» между видеокартой и играми, позволяющая полностью реализовать всю вычислительную мощь компьютера для отрисовки красивой графики.

NVIDIA GF100 — 40-нм графический процессор, первый представитель линейки GeForce 400. 

Дата выпуска	12 апреля 2010 год

### Kepler

Kepler — микроархитектура, созданная для высокопроизводительных вычислений с акцентом на энергоэффективности. 
Тогда как направленностью предыдущей архитектуры, Fermi, была чистая производительность, Kepler рассчитан на энергоэффективность, программируемость и производительность.

Дата выпуска	Апрель 2012

### Maxwell

Maxwell — кодовое название микроархитектуры графических процессоров, разработанной в качестве преемника микроархитектуры Kepler. 
Архитектура Maxwell была введена в более поздних моделях. 
Nvidia для новой архитектуры Maxwell взяла в качестве основы Kepler и доработала её в нескольких областях.

Дата выпуска	Февраль 2014

### Pascal

Pascal — микроархитектура графических процессоров, разрабатываемых NVIDIA в качестве преемника микроархитектуры Maxwell. 
Pascal используется в видеокартах GeForce 10.

Дата выпуска	Май 2016

#### Маркировка

- GT – это буквенное сочетание отражает видеокарты низкого уровня производительности, их нельзя рассматривать как игровые.
- GTX – этим индексом обозначаются видеоадаптеры среднего и высокого уровня, которые хорошо подходят для игр.
- M – мобильная видеокарта (они сильно слабее своих братьев без этой буквы)
- X – маркировка более производительной видеокарты у мобильных решений
- LE – так обозначается версия карты с более низкой тактовой частотой у мобильных адаптеров
- Ti – обозначение более производительной версии у десктопных карт

### TITAN

Это подлинейка GeForce, они имеют индекс GTX. 
Для начала надо разобраться с позиционированием данной линейки. 
Это самые быстрые и дорогие видеокарты на данный момент. 
Но эта цена действительно слишком высока для такого уровня производительности. 
Все дело в том, что так же они позиционируются, как мощные профессиональные видеокарты для математических вычислений и вычислений FP 64 (вычисления с плавающей запятой двойной точности). Это своего рода внедорожник в мире видеокарт – и работать можно и играть. 
Исключением является Titan X, который не хватает с неба звезд в FP 64 – вычислениях и по сути является просто очень дорогой видеокартой с огромным набором видеопамяти.

В этой линейке на начало 2016 года есть только 5 видеокарт и почти все в референсном дизайне (версии от сторонних производителей).

TITAN, TITAN Black Edition и TITAN Z принадлежат Kepler, TITAN X – Maxwell, TITAN X в Pascal (отличия в приставках: у первого полное название NVIDIA GeForce GTX TITAN X, а у второго просто NVIDIA TITAN X).

## Поколения семества Pascal

Первая буква означает микроархитектуру чипа:

ее нет – Fermi
K – Kepler
M – Maxwell

Буквенные индексы есть и в конце названия модели:

M – обозначение мобильной видеокарты
D – другой набор выходов. В случае с K2000 вместо  двух портов  DisplayPort и одного DL-DVI в D-версии стоят два выхода DL-DVI и один mini-DisplayPort.
Цифры указывают положение модели в линейке (больше — лучше).

----
OpenGL переводится как Открытая Графическая Библиотека (Open Graphics Library), это означает, что OpenGL - это открытый и мобильный стандарт. 
Программы, написанные с помощью OpenGL можно переносить практически на любые платформы, получая при этом одинаковый результат, будь это графическая станция или суперкомпьютер.












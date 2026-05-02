# Algorithms for vehicle routing problem with constraints 
## Алгоритмы для задачи маршрутизации транспорта с ограничениями (Fair VRP)

Проект посвящён разработке алгоритмов для задачи маршрутизации транспорта (Vehicle Routing Problem, VRP) с ограничениями в контексте обслуживания парка самокатов.

Исполнители выполняют операции по замене аккумуляторов, ремонту и релокации самокатов. 
Требуется построение маршрутов для нескольких исполнителей, работающих одновременно в пределах одного города.

### Постановка проблемы

В текущей системе маршруты формируются последовательно и жадно: сначала строится маршрут для первого исполнителя, 
затем для второго и так далее. Такой подход приводит к ухудшению качества маршрутов для «последних» исполнителей 
и к неравномерному распределению нагрузки. В результате снижается общая эффективность системы и возникает проблема несправедливого распределения работ.

### Цель работы

Целью проекта является переход от последовательного построения маршрутов к совместной оптимизации всех маршрутов одновременно. 
При построении решения необходимо учитывать всех исполнителей, работающих в городе, а не только тех, кто закреплён за одной базой. 
Возможным расширением является постановка в формате Multi-Depot VRP.

### Ожидаемый результат

В результате должен быть разработан алгоритм совместной маршрутизации, обеспечивающий более равномерное распределение нагрузки и повышение общей эффективности системы,
с возможностью применения в реальных задачах.
Разрабатываемый алгоритм должен быть масштабируемым и работать для десятков исполнителей и тысяч точек. 
Он должен учитывать индивидуальные ограничения исполнителей, такие как вместимость и максимальная длительность маршрута.
Работа предполагает анализ существующих алгоритмов для VRP и Fair VRP, реализацию базовых методов из литературы, 
разработку собственной модификации и проведение экспериментального сравнения на наборах данных. 
Отдельное внимание уделяется анализу полученных результатов и объяснению поведения алгоритмов в различных сценариях.

---

### Установка

Проект использует форк [PyVRP](https://github.com/ksenia274/PyVRP) с изменениями в C++ ядре, поэтому при установке требуется компилятор C++.

#### Linux

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install poetry-core meson ninja docblock
pip install -r requirements.txt --no-build-isolation
```

#### Windows

Требуется [Visual Studio 2022 Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022) с компонентом «Разработка классических приложений на C++».

```bat
python -m venv venv
install_fork.bat
```

---

### Датасеты

**Solomon VRPTW** — стандартный бенчмарк, 56 инстансов (R1/R2/C1/C2/RC1/RC2). Скачивается автоматически через kagglehub.

**Yandex VRP** — реальные инстансы задачи обслуживания самокатов:

```bash
git clone https://github.com/Stephic-Hardy/Fair_VRP_ala_Yandex
```

Клонируйте рядом с этим репозиторием (в `../Fair_VRP_ala_Yandex/`).

---

### Алгоритмы

| Алгоритм | Описание |
|---|---|
| `hgs_simple` | HGS без fairness-постобработки |
| `hgs_rebalance` | HGS + ребалансировка маршрутов (relocate/swap) |
| `hgs_penalty` | HGS с итеративными рестартами и штрафной матрицей расстояний |
| `hgs_adaptive` | HGS + адаптивные веса route\_balance в ILS (форк PyVRP) |
| `alns` | ALNS с fairness-штрафом в целевой функции |

---

### Быстрый старт

```bash
# Одиночный запуск
python main.py run --config configs/runs/hgs_adaptive_yandex.yaml

# Бенчмарк нескольких алгоритмов × нескольких инстансов
python main.py benchmark --config configs/benchmarks/compare_three_methods.yaml

# Визуализация результатов бенчмарка
python main.py visualize results/benchmarks/<name>/

# Сравнение двух бенчмарков
python main.py compare results/benchmarks/bench_a/ results/benchmarks/bench_b/

# Карта маршрутов СПб
python main.py spb-map --load-json ../Fair_VRP_ala_Yandex/vrp_problems/3.json --algorithm hgs_adaptive
```

Override-флаги (`--seed`, `--time`, `--instance`) применяются поверх YAML:

```bash
python main.py run --config configs/runs/hgs_simple_yandex.yaml --seed 7 --time 60 --instance 5
```

---

### Структура проекта

```
algorithms/          # солверы и fairness-утилиты
  factory.py         # build_solver(config) — единая точка входа
  hgs_base.py        # базовый класс всех HGS-солверов
  hgs_solver_*.py    # конкретные HGS-варианты
  alns_solver.py     # ALNS
  algorithm_params.py  # Pydantic-схемы параметров
  solver_result.py   # SolverResult, SolverConfig, SolverDiagnostics

configs/
  global.yaml        # пути, метрики по умолчанию
  runs/              # YAML-конфиги одиночных запусков
  benchmarks/        # YAML-конфиги бенчмарков

data/                # загрузчики инстансов
metrics/             # metrics/fairness.py — FairnessReport, DimensionMetrics

runtime/             # инфраструктура сохранения и CLI
  cli/               # команды run, benchmark, visualize, compare
  global_config.py   # singleton из configs/global.yaml
  run_dir.py         # create_run_dir, save_run, load_run
  serialization.py   # JSON/YAML/CSV

scripts/
  run_spb_map.py     # СПб-инстанс + Folium-карта
  run_hgs_rs.py      # Rectangle Splitting (код коллеги)

visualization/
  plots/             # pareto, dimensions, distribution, category_heatmap
  compose.py         # plot_all(df, output_dir)
  trace_plot.py      # трассировка adaptive-солвера
  map_routes.py      # Folium-карта маршрутов

tests/               # 53 теста, pytest
```

---

### Метрики fairness

Вычисляются независимо по трём измерениям: **расстояние**, **нагрузка**, **число клиентов**.

| Метрика | Формула | Интерпретация |
|---|---|---|
| `worst_ratio` | max / mean | Во сколько раз худший маршрут превышает средний. `1.0` = идеал |
| `gini` | Σᵢⱼ\|xᵢ−xⱼ\| / (2·n·Σxᵢ) | `0.0` = все маршруты одинаковы |
| `cv` | std / mean | Стандартный коэффициент вариации |

Колонки в `metrics.csv`: `dist_worst_ratio`, `dist_gini`, `dist_cv`, `load_worst_ratio`, `load_gini`, `load_cv`, `clients_worst_ratio`, `clients_gini`, `clients_cv`.

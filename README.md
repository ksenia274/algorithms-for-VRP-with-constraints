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
# Системные зависимости (один раз)
sudo apt install python3.12 python3.12-venv build-essential git

# Создать окружение и установить всё
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

`install_fork.bat` активирует окружение MSVC и выполняет `pip install -r requirements.txt --no-build-isolation`.

---

### Датасеты

**Solomon VRPTW** — стандартный бенчмарк, 56 инстансов (R1/R2/C1/C2/RC1/RC2). Скачивается автоматически через kagglehub:
- https://www.kaggle.com/datasets/masud7866/solomon-vrptw-benchmark

**Yandex VRP** — реальные инстансы задачи обслуживания самокатов:
```bash
git clone https://github.com/Stephic-Hardy/Fair_VRP_ala_Yandex
```

---

### Алгоритмы

| `--algorithm` | Описание |
|---|---|
| `hgs_simple` | HGS без fairness-постобработки |
| `hgs_rebalance` | HGS + ребалансировка маршрутов (relocate/swap) для улучшения fairness |
| `hgs_rs` | HGS + rectangle splitting — исследование фронта Парето по (distance, max\_duration) |
| `hgs_penalty` | HGS с итеративными рестартами и штрафной матрицей расстояний |
| `hgs_adaptive` | HGS + адаптивные веса (route\_balance) прямо в ILS через форк PyVRP |
| `alns` | ALNS с fairness-штрафом в целевой функции |

---

### Запуск

#### Одиночный инстанс

```bash
# Yandex
python main.py --algorithm hgs_rebalance --instance ../Fair_VRP_ala_Yandex/vrp_problems/0.json
python main.py --algorithm hgs_penalty   --instance ../Fair_VRP_ala_Yandex/vrp_problems/0.json
python main.py --algorithm hgs_adaptive  --instance ../Fair_VRP_ala_Yandex/vrp_problems/0.json
python main.py --algorithm alns          --instance ../Fair_VRP_ala_Yandex/vrp_problems/0.json

# Solomon
python main.py --algorithm hgs_rebalance --instance R101
```

#### Бенчмарк (все инстансы → CSV)

```bash
# Solomon, hgs_rebalance (по умолчанию)
python main.py benchmark

# Solomon, ALNS
python main.py benchmark --algorithm alns

# Yandex
python main.py benchmark --dataset yandex --yandex-path ../Fair_VRP_ala_Yandex/vrp_problems
```

Результаты сохраняются в `results/fairness_benchmark.csv`.

#### Визуализация результатов бенчмарка

```bash
# Из дефолтного CSV
python main.py visualise

# Из произвольного CSV
python main.py visualise --csv results/fairness_benchmark.csv --output visualization/output
```

Строятся четыре графика в `visualization/output/`:

| Файл | Содержание |
|---|---|
| `01_pareto.png` | Scatter: медианное расстояние vs. медианный `dist_worst_ratio` — компромисс стоимость/равномерность, фронт Парето |
| `02_dimensions.png` | Сгруппированные бары: `worst_ratio` по трём измерениям (расстояние / нагрузка / клиенты) для каждого алгоритма |
| `03_gini_distribution.png` | Скрипичный график: распределение `dist_gini` по инстансам для каждого алгоритма |
| `04_category_heatmap.png` | Тепловая карта: медианный `dist_worst_ratio` по категориям Solomon × алгоритм |

Для сравнения нескольких алгоритмов через CLI:

```bash
python visualization/fairness_charts.py results/hgs.csv results/alns.csv \
    --labels "HGS rebalance" "ALNS" --output visualization/output
```

#### Маршруты на карте Санкт-Петербурга

```bash
# HGS + fairness rebalancing
python scripts/run_spb_hgs.py --load-json ../Fair_VRP_ala_Yandex/vrp_problems/0.json

# Prize-Collecting режим (клиенты необязательны, приоритет по point_scores)
python scripts/run_spb_hgs.py --load-json ../Fair_VRP_ala_Yandex/vrp_problems/0.json --prizes

# HGS + адаптивные веса
python scripts/run_spb_hgs_adaptive.py --load-json ../Fair_VRP_ala_Yandex/vrp_problems/0.json

# HGS + штрафная матрица
python scripts/run_spb_hgs_penalty.py --load-json ../Fair_VRP_ala_Yandex/vrp_problems/0.json

# ALNS
python scripts/run_spb_alns.py --load-json ../Fair_VRP_ala_Yandex/vrp_problems/0.json

# Генерация синтетического инстанса и сохранение
python scripts/run_spb_hgs.py --points 100 --save-json my_problem.json
```

Карта сохраняется в `visualization/map_results/` в формате HTML (Folium).

---

### Параметры

#### `main.py`

| Параметр | По умолчанию | Описание |
|---|---|---|
| `--algorithm` | — | `hgs_simple` / `hgs_rebalance` / `hgs_rs` / `hgs_penalty` / `hgs_adaptive` / `alns` |
| `--instance` | `R101` | Имя Solomon-инстанса или путь к `.json` |
| `--time` | `30` | Лимит времени солвера (с) |
| `--vehicles` | `25` | Число машин |
| `--capacity` | `200` | Вместимость (для Yandex берётся из JSON) |
| `--max-cost-increase` | `5.0` | Допустимый рост стоимости при ребалансировке (%) |
| `--rebalance-iters` | `3000` | Итерации ребалансировки (`hgs_rebalance`) |
| `--alns-iterations` | `25000` | Итерации ALNS |
| `--fairness-weight` | `100.0` | Вес fairness в целевой функции ALNS |
| `--seed` | `42` | Random seed |

#### `main.py benchmark` (дополнительные параметры)

| Параметр | По умолчанию | Описание |
|---|---|---|
| `--algorithm` | `hgs_rebalance` | Алгоритм для прогона |
| `--dataset` | `solomon` | `solomon` или `yandex` |
| `--yandex-path` | `vrp_problems` | Путь к папке с JSON-инстансами |
| `--max-cost-increase` | `8.0` | Допустимый рост стоимости (%) |
| `--rebalance-iters` | `5000` | Итерации ребалансировки |
| `--output` | `results` | Папка для сохранения CSV |

#### `run_spb_hgs.py`

| Параметр | По умолчанию | Описание |
|---|---|---|
| `--load-json` | — | Загрузить готовый JSON-инстанс |
| `--save-json` | — | Сохранить сгенерированный инстанс |
| `--points` | `50` | Число точек при генерации |
| `--time` | `30` | Лимит времени (с) |
| `--vehicles` | `10` | Число машин |
| `--capacity` | из JSON | Вместимость |
| `--output-map` | `map_results.html` | Путь к HTML-файлу карты |
| `--no-fairness` | off | Отключить ребалансировку |
| `--prizes` | off | Prize-Collecting: клиенты необязательны, приоритет по `point_scores` |
| `--seed` | `42` | Random seed |

---

### Метрики fairness

Все метрики вычисляются независимо по трём измерениям: **расстояние**, **нагрузка**, **число клиентов** на маршрут.

| Метрика | Формула | Интерпретация |
|---|---|---|
| **worst_ratio** | max / mean | Во сколько раз худший маршрут превышает средний. `1.0` = идеальный баланс, ≥ 1.0 всегда |
| **gini** | Σᵢⱼ\|xᵢ−xⱼ\| / (2·n·Σxᵢ) | Коэффициент Джини из экономики. `0.0` = все маршруты одинаковы, `1.0` = максимальное неравенство |
| **mean, std, min, max** | — | Базовые описательные статистики по маршрутам |

**Столбцы в CSV** (суффиксы `_before` / `_after`):

```
dist_worst_ratio   load_worst_ratio   clients_worst_ratio
dist_gini          load_gini          clients_gini
```

Плюс служебные: `feasible`, `total_distance`, `num_routes`, `rebalance_moves`, `cost_delta_pct`, `solve_time_s`, `category`.

> Составная оценка (fairness score) и CV намеренно убраны — они вносили произвольные веса и мешали сравнению прогонов. Все выводы строятся на `worst_ratio` и `gini`.

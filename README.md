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

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

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
| `alns` | ALNS с fairness-штрафом в целевой функции |

---

### Запуск

#### Одиночный инстанс

```bash
# HGS без fairness
python main.py --algorithm hgs_simple --instance R101

# HGS + fairness rebalancing (Solomon)
python main.py --algorithm hgs_rebalance --instance R101

# HGS + fairness rebalancing (Yandex)
python main.py --algorithm hgs_rebalance --instance ../Fair_VRP_ala_Yandex/vrp_problems/0.json

# HGS + rectangle splitting
python main.py --algorithm hgs_rs --instance R101

# ALNS с fairness
python main.py --algorithm alns --instance R101
```

#### Бенчмарк (все инстансы → CSV)

```bash
# Solomon, HGS + rebalancing (по умолчанию)
python main.py benchmark

# Solomon, ALNS
python main.py benchmark --algorithm alns

# Yandex, HGS + rebalancing
python main.py benchmark --dataset yandex --yandex-path ../Fair_VRP_ala_Yandex/vrp_problems

# С кастомными параметрами
python main.py benchmark --dataset yandex --yandex-path ../Fair_VRP_ala_Yandex/vrp_problems \
    --time 60 --rebalance-iters 5000 --max-cost-increase 8.0
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

| Файл | Описание |
|---|---|
| `01_gini_before_after.png` | Gini-коэффициент до и после ребалансировки по каждому инстансу |
| `02_category_summary.png` | Средние значения Gini, CV, fairness score и Gini нагрузки по категориям Solomon (R1/R2/C1/C2/RC1/RC2) |
| `03_metric_heatmap.png` | Тепловая карта всех fairness-метрик по инстансам |
| `04_rebalance_moves.png` | Количество применённых ходов ребалансировки по инстансам |

#### Визуализация маршрутов на карте (Санкт-Петербург)

Запуск на реальных данных Yandex с построением интерактивной HTML-карты через Folium:

```bash
# HGS + fairness rebalancing
python run_spb_hgs.py --json ../Fair_VRP_ala_Yandex/vrp_problems/0.json

# ALNS с fairness
python run_spb_alns.py --json ../Fair_VRP_ala_Yandex/vrp_problems/0.json
```

Карта сохраняется в `visualization/map_results/`. Маршруты отображаются цветными линиями, каждая точка обслуживания — кружком соответствующего цвета.

---

### Параметры

#### Для одиночного запуска (`main.py`)

| Параметр | По умолчанию | Описание |
|---|---|---|
| `--algorithm` | — | Алгоритм: `hgs_simple`, `hgs_rebalance`, `hgs_rs`, `alns` |
| `--instance` | `R101` | Имя Solomon-инстанса или путь к `.json` |
| `--time` | `30` | Лимит времени солвера (секунды) |
| `--vehicles` | `25` | Число машин (для Yandex берётся из JSON) |
| `--capacity` | `200` | Вместимость (для Yandex берётся из JSON) |
| `--max-cost-increase` | `5.0` | Допустимый рост стоимости при ребалансировке (%) |
| `--rebalance-iters` | `3000` | Число итераций ребалансировки (для `hgs_rebalance`) |
| `--alns-iterations` | `25000` | Число итераций ALNS |
| `--fairness-weight` | `100.0` | Вес fairness в целевой функции ALNS |
| `--seed` | `42` | Random seed |

#### Для бенчмарка (`main.py benchmark`)

| Параметр | По умолчанию | Описание |
|---|---|---|
| `--algorithm` | `hgs_rebalance` | Алгоритм для прогона |
| `--dataset` | `solomon` | Датасет: `solomon` или `yandex` |
| `--yandex-path` | `vrp_problems` | Путь к папке с JSON-инстансами Yandex |
| `--time` | `30` | Лимит времени на инстанс (секунды) |
| `--max-cost-increase` | `8.0` | Допустимый рост стоимости при ребалансировке (%) |
| `--rebalance-iters` | `5000` | Число итераций ребалансировки |
| `--alns-iterations` | `25000` | Число итераций ALNS |
| `--fairness-weight` | `100.0` | Вес fairness в целевой функции ALNS |
| `--output` | `results` | Папка для сохранения CSV |
| `--seed` | `42` | Random seed |

---

### Метрики fairness

Для каждого решения вычисляются метрики равномерности распределения по маршрутам:

| Метрика | Описание |
|---|---|
| **Gini** | Коэффициент Джини (0 = равенство, 1 = максимальное неравенство) |
| **Jain** | Индекс Джейна (1 = равенство) |
| **CV** | Коэффициент вариации (std / mean) |
| **Fairness score** | Взвешенная сумма CV по расстояниям (0.5), нагрузке (0.3) и числу клиентов (0.2) |

Метрики считаются по расстоянию маршрутов, нагрузке (demand) и числу клиентов на маршрут.

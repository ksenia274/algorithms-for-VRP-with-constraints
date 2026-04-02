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



### Установка

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Датасеты

**Solomon VRPTW** — стандартный бенчмарк, 56 инстансов (R1/R2/C1/C2/RC1/RC2):
- https://www.kaggle.com/datasets/masud7866/solomon-vrptw-benchmark

**Yandex VRP** — реальные инстансы задачи обслуживания самокатов:
```bash
git clone https://github.com/Stephic-Hardy/Fair_VRP_ala_Yandex
```

---

### Запуск

#### HGS на одном инстансе

```bash
# Solomon (по имени инстанса)
python main.py --instance R101

# Solomon с fairness rebalancing
python main.py --instance R101 --fairness

# Yandex (по пути к JSON)
python main.py --instance ../Fair_VRP_ala_Yandex/vrp_problems/0.json

# Yandex с fairness
python main.py --instance ../Fair_VRP_ala_Yandex/vrp_problems/0.json --fairness
```

#### ALNS на одном инстансе

```bash
# Solomon
python main.py --alns --instance R101

# Yandex с fairness
python main.py --alns --fairness --instance ../Fair_VRP_ala_Yandex/vrp_problems/0.json
```

#### Бенчмарк (все инстансы → CSV)

```bash
# Solomon (все 56 инстансов)
python main.py benchmark

# Yandex
python main.py benchmark --dataset yandex --yandex-path ../Fair_VRP_ala_Yandex/vrp_problems

# С кастомными параметрами
python main.py benchmark --dataset yandex --yandex-path ../Fair_VRP_ala_Yandex/vrp_problems \
    --time 60 --rebalance-iters 5000 --max-cost-increase 8.0
```

Результаты сохраняются в `results/fairness_benchmark.csv`.

#### Визуализация

```bash
# Из дефолтного CSV
python main.py visualise

# Из произвольного CSV
python main.py visualise --csv results/fairness_benchmark.csv --output visualization/output
```

#### Параметры

| Параметр | По умолчанию | Описание |
|---|---|---|
| `--instance` | `R101` | Имя Solomon-инстанса или путь к `.json` |
| `--time` | `60` | Лимит времени солвера (секунды) |
| `--vehicles` | `25` | Число машин (для Yandex берётся из JSON) |
| `--capacity` | `200` | Вместимость (для Yandex берётся из JSON) |
| `--fairness` | off | Включить fairness rebalancing |
| `--max-cost-increase` | `5.0` | Допустимый рост стоимости при ребалансировке (%) |
| `--rebalance-iters` | `3000` | Итерации ребалансировки |
| `--seed` | `42` | Random seed |
| `--alns-iterations` | `25000` | Число итераций ALNS |
| `--fairness-weight` | `100.0` | Вес fairness в целевой функции ALNS |
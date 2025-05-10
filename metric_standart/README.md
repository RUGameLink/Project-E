# Инструментарий оценки моделей

Этот инструментарий предоставляет утилиты для оценки моделей нейронных сетей из директории model_save_preset с дополнительными метриками, такими как RMSE, коэффициент детерминации R², и другими, которые могли не быть рассчитаны во время обучения.

## Структура директории

```
metric_standart/
├── model_evaluator.py      # Основной модуль с функциями оценки
├── run_evaluation.py       # Скрипт для запуска оценки моделей
├── analyze_results.py      # Скрипт для анализа и визуализации результатов
├── plots/                  # Директория для сгенерированных графиков
└── README.md               # Этот файл
```

## Требования

- Python 3.7+
- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn

Вы можете установить необходимые пакеты с помощью:

```bash
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn
```

## Использование

### 1. Запуск оценки моделей

Этот скрипт загружает модели из директории model_save_preset, оценивает их с помощью дополнительных метрик и сохраняет результаты в CSV-файл.

```bash
python metric_standart/run_evaluation.py --data data/data.csv --models-dir model_save_preset/models --output metric_standart/extended_model_metrics.csv
```

Параметры:
- `--data`: Путь к файлу данных (по умолчанию: 'data/data.csv')
- `--models-dir`: Директория, содержащая группы моделей (по умолчанию: 'model_save_preset/models')
- `--output`: Выходной файл для метрик (по умолчанию: 'metric_standart/extended_model_metrics.csv')
- `--plots-dir`: Директория для сохранения графиков (по умолчанию: 'metric_standart/plots')
- `--time-step`: Временной шаг для последовательных данных (по умолчанию: 5)
- `--plot-samples`: Количество образцов для визуализации предсказаний (по умолчанию: 100)

### 2. Анализ результатов

Этот скрипт анализирует и визуализирует результаты оценки с помощью различных диаграмм и сравнений.

```bash
python metric_standart/analyze_results.py --metrics-file metric_standart/extended_model_metrics.csv
```

Параметры:
- `--metrics-file`: Путь к файлу с метриками CSV (по умолчанию: 'metric_standart/extended_model_metrics.csv')
- `--output-dir`: Директория для сохранения графиков анализа (по умолчанию: 'metric_standart/plots')
- `--summary-file`: Путь для сохранения итогового отчета (по умолчанию: 'metric_standart/model_summary.txt')

## Вывод

Инструментарий генерирует следующие выходные данные:

1. **Файл расширенных метрик CSV** - Содержит все рассчитанные метрики для каждой модели
2. **Графики сравнения предсказаний** - Визуальное сравнение фактических и предсказанных значений
3. **Графики сравнения метрик** - Столбчатые диаграммы, сравнивающие модели по каждой метрике
4. **Лепестковые диаграммы** - Сравнение лучших моделей по нескольким метрикам
5. **Графики производительности групп** - Диаграммы размаха, показывающие распределение метрик по группам моделей
6. **Итоговый отчет** - Текстовый файл с лучшими моделями по метрикам и статистикой по группам

## Расширенное использование

### Использование основного модуля

Вы можете импортировать функции из модуля model_evaluator в свои собственные Python-скрипты:

```python
from metric_standart.model_evaluator import (
    load_and_prepare_data,
    load_models_from_directory,
    evaluate_models,
    save_evaluation_results
)

# Загрузка и подготовка данных
data = load_and_prepare_data('data/data.csv')

# Загрузка моделей
models = load_models_from_directory('model_save_preset/models')

# Оценка моделей
results = evaluate_models(models, data)

# Сохранение результатов
save_evaluation_results(results, 'my_metrics.csv')
``` 
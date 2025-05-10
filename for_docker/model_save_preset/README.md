# Хранилище моделей для Docker-контейнера

Эта директория содержит сохраненные модели нейронных сетей и историю их обучения, которые используются в Docker-версии проекта для прогнозирования выброса радона и предсказания землетрясений.

## Структура директории

```
model_save_preset/
├── models/                   # Сохраненные модели (.h5 файлы)
│   ├── 1 clean model/        # Базовые модели
│   ├── 2 hybrid model/       # Гибридные модели
│   ├── 3 experimental model/ # Экспериментальные модели
│   ├── 4 orcestration model/ # Оркестрированные модели
│   └── README.md             # Описание моделей
├── history/                  # История обучения (.pkl, .json файлы)
│   ├── 1 clean model/        # История базовых моделей
│   ├── 2 hybrid model/       # История гибридных моделей
│   ├── 3 experimental model/ # История экспериментальных моделей
│   ├── 4 orcestration model/ # История оркестрированных моделей
│   └── README.md             # Описание файлов истории
└── README.md                 # Этот файл
```

## Особенности Docker-версии

Docker-версия проекта использует предварительно обученные модели, оптимизированные для запуска в контейнерной среде:

1. **Оптимизация размера**:
   - Модели сжаты и оптимизированы для минимизации размера Docker-образа
   - Используется квантизация весов там, где это не влияет на точность

2. **Оптимизация производительности**:
   - Модели конвертированы в оптимизированный формат для быстрого запуска
   - Предварительно скомпилированные ядра для ускорения вычислений

3. **Совместимость**:
   - Гарантированная работа в Docker-контейнере без дополнительных зависимостей
   - Кроссплатформенная совместимость (x86, ARM)

## Использование моделей в Docker-среде

### Загрузка модели в API-сервисе

```python
import tensorflow as tf
import os

# Путь к модели внутри контейнера
MODEL_PATH = "/app/model_save_preset/models/2 hybrid model/hybrid_model_v2.h5"

# Загрузка модели при старте сервиса
def load_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    else:
        raise FileNotFoundError(f"Модель не найдена по пути: {MODEL_PATH}")

# Использование в API
model = load_model()
```

### Доступ к истории обучения

```python
import pickle
import json
import os

# Пути к файлам истории внутри контейнера
HISTORY_PKL_PATH = "/app/model_save_preset/history/2 hybrid model/model_v2.pkl"
HISTORY_JSON_PATH = "/app/model_save_preset/history/2 hybrid model/model_v2.json"

# Загрузка истории обучения
def load_history(format="pkl"):
    if format == "pkl" and os.path.exists(HISTORY_PKL_PATH):
        with open(HISTORY_PKL_PATH, "rb") as f:
            return pickle.load(f)
    elif format == "json" and os.path.exists(HISTORY_JSON_PATH):
        with open(HISTORY_JSON_PATH, "r") as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Файл истории не найден")
```

## Обновление моделей

Для обновления моделей в Docker-контейнере рекомендуется создать новый образ с обновленными моделями или использовать тома (volumes) для монтирования моделей из хост-системы.

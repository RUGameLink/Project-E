# Улучшенная модель прогнозирования уровня радона для Google Colab

Этот проект содержит исходные файлы для создания и запуска улучшенных моделей прогнозирования уровня радона на основе данных о температуре и давлении. Проект включает расширенные возможности визуализации с помощью Plotly и усовершенствованные нейронные сети для более точного прогнозирования.

## Преимущества улучшенной модели

1. **Улучшенные архитектуры нейронных сетей**:
   - Добавлен слой BatchNormalization для стабилизации обучения
   - Увеличены размеры скрытых слоев (128/64 нейронов вместо 64/32)
   - Добавлена модель CNN-LSTM для улучшения захвата временных зависимостей
   - Реализована ансамблевая модель, объединяющая LSTM, GRU и CNN подходы
   - Применена L2-регуляризация для предотвращения переобучения

2. **Расширенная аналитика данных**:
   - Декомпозиция временных рядов (тренд, сезонность, остаток)
   - Анализ выбросов и автоматическая их обработка
   - Улучшенный анализ запаздывающих переменных

3. **Усовершенствованные прогнозы**:
   - Метод Монте-Карло для оценки неопределенности прогнозов
   - Визуализация доверительных интервалов прогнозов
   - Анализ важности признаков для интерпретации модели

4. **Интерактивные визуализации с Plotly**:
   - Полностью интерактивные графики с возможностью масштабирования
   - Информативные подсказки при наведении курсора
   - Настраиваемые цветовые схемы и темы оформления

## Структура проекта

Проект содержит следующие файлы:

1. `radon_analysis.py` - код для анализа данных и визуализации
2. `radon_models.py` - улучшенные модели нейронных сетей (LSTM, GRU, BiLSTM, CNN-LSTM, Ensemble)
3. `radon_prediction.py` - код для прогнозирования уровня радона с оценкой неопределенности

## Инструкции по запуску в Google Colab

### Шаг 1: Создание нового блокнота

1. Откройте [Google Colab](https://colab.research.google.com)
2. Создайте новый блокнот

### Шаг 2: Клонирование репозитория и установка зависимостей

```python
# Клонирование репозитория
!git clone https://github.com/yourusername/Project-E.git
%cd Project-E/alt_new_model_gc

# Установка необходимых пакетов
!pip install numpy pandas matplotlib scikit-learn tensorflow seaborn scipy plotly statsmodels
```

### Шаг 3: Импорт необходимых модулей

```python
# Импорт функций из модулей
from radon_analysis import *
from radon_models import *
from radon_prediction import *

# Отображение графиков Plotly в ноутбуке
import plotly.io as pio
pio.renderers.default = 'colab'
```

### Шаг 4: Загрузка данных

#### Вариант 1: Загрузка собственных данных

```python
# Загрузка собственных данных
from google.colab import files
uploaded = files.upload()  # Загрузите CSV-файл с данными

# Получение имени загруженного файла
file_name = list(uploaded.keys())[0]

# Загрузка и предобработка данных
data = load_data(file_name)
```

#### Вариант 2: Создание синтетических данных для тестирования

```python
# Создание образца данных
file_name = create_sample_data(filename='sample_data.csv', periods=1500, frequency='H')

# Загрузка данных
data = load_data(file_name)
```

### Шаг 5: Анализ данных

```python
# Визуализация временных рядов с помощью Plotly
fig_time_series = plot_time_series(data, plot_type='plotly')
fig_time_series.show()

# Анализ корреляций
fig_corr, corr_matrix = analyze_correlations(data, plot_type='plotly')
fig_corr.show()

# Визуализация взаимосвязей
fig_scatter = plot_scatter_relationships(data, plot_type='plotly')
fig_scatter.show()

# Анализ статистики
stats, fig_hist = analyze_statistics(data, plot_type='plotly')
fig_hist.show()

# Сезонная декомпозиция для радона
radon_col = [col for col in data.columns if 'radon' in col.lower()][0]
decomposition, fig_decompose = perform_seasonal_decomposition(data, radon_col, period=24, plot_type='plotly')
fig_decompose.show()

# Анализ запаздывающих переменных
corr_lag, fig_lag = perform_lag_analysis(data, lag_days=7, plot_type='plotly')
fig_lag.show()
```

### Шаг 6: Подготовка данных и обучение моделей

```python
# Подготовка данных с увеличенной длиной последовательности
X, y, X_train, X_test, y_train, y_test, scaler = prepare_data(
    data, 
    seq_length=10,  # Увеличенная длина последовательности
    test_size=0.2, 
    random_state=42,
    scaler_type='minmax'  # Также доступен 'standard'
)

# Обучение моделей с расширенным набором типов
models, histories, best_model, saved_paths = train_models(
    X_train, 
    y_train, 
    epochs=100,  # Увеличено количество эпох
    batch_size=32, 
    validation_split=0.2, 
    save_models=True,
    model_types=['lstm', 'gru', 'bidirectional', 'cnn_lstm', 'ensemble'],
    l2_reg=0.001  # Регуляризация для предотвращения переобучения
)

# Вывод путей сохраненных моделей
print("\nСохраненные модели и истории обучения:")
for model_type, paths in saved_paths.items():
    if model_type != 'best':
        print(f"{model_type.upper()}:")
        print(f"  Модель: {paths['model']}")
        print(f"  История: {paths['history']}")
    else:
        print(f"Лучшая модель: {paths}")

# Оценка моделей
results = evaluate_all_models(models, X_test, y_test, scaler)
```

### Шаг 7: Прогнозирование с оценкой неопределенности

```python
# Прогнозирование с использованием лучшей модели и оценка неопределенности
predictions, future_predictions, prediction_bounds = make_predictions(
    best_model, 
    data, 
    seq_length=10, 
    future_steps=48,  # 2 дня вперед (при почасовых данных)
    use_monte_carlo=True,  # Включение метода Монте-Карло
    num_simulations=50  # Количество симуляций
)

# Интерактивная визуализация результатов
fig = plot_predictions(
    data, 
    predictions, 
    future_predictions, 
    prediction_bounds,
    plot_title="Прогнозирование уровня радона с оценкой неопределенности",
    plot_theme='plotly_white'
)
fig.show()

# Анализ важности признаков
fig_importance = plot_feature_importance(best_model, data, seq_length=10)
fig_importance.show()
```

### Шаг 8: Загрузка сохраненной модели для прогнозирования

```python
# Получение пути к лучшей модели
best_model_path = saved_paths['best']

# Создание комплексного прогноза с визуализацией и анализом
predictions, future_predictions, prediction_bounds, fig_pred, fig_importance = predict_radon_levels(
    best_model_path, 
    data, 
    use_monte_carlo=True,
    plot_theme='plotly_white'
)

# Отображение результатов
fig_pred.show()
fig_importance.show()
```

## Примеры графиков

Ниже приведены примеры визуализаций, которые можно получить с помощью улучшенной модели:

1. **Временные ряды с интерактивными элементами управления**
2. **Интерактивная тепловая карта корреляций**
3. **Прогноз с доверительными интервалами**
4. **График важности признаков для интерпретации прогнозов**
5. **Сезонная декомпозиция временного ряда**

## Расширенные возможности

### Настройка скользящего среднего

Для данных с высоким уровнем шума можно применить скользящее среднее:

```python
# Применение скользящего среднего к временному ряду
window_size = 12  # размер окна для среднего (в часах)
radon_col = [col for col in data.columns if 'radon' in col.lower()][0]
data[f"{radon_col}_MA"] = data[radon_col].rolling(window=window_size, center=True).mean()

# Использование сглаженных данных для обучения
data_smoothed = data.dropna().copy()
data_smoothed[radon_col] = data_smoothed[f"{radon_col}_MA"]
data_smoothed = data_smoothed.drop(columns=[f"{radon_col}_MA"])

# Теперь можно использовать data_smoothed для обучения модели
X, y, X_train, X_test, y_train, y_test, scaler = prepare_data(data_smoothed)
```

### Оптимизация гиперпараметров

Для поиска оптимальных гиперпараметров модели можно использовать Keras Tuner:

```python
# Установка Keras Tuner
!pip install -q keras-tuner

import keras_tuner as kt

# Определение модели для настройки
def build_model(hp):
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = Sequential()
    
    # Настройка количества нейронов в первом слое LSTM
    lstm_units_1 = hp.Int('lstm_units_1', min_value=32, max_value=256, step=32)
    model.add(LSTM(lstm_units_1, input_shape=input_shape, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)))
    
    # Настройка количества нейронов во втором слое LSTM
    lstm_units_2 = hp.Int('lstm_units_2', min_value=16, max_value=128, step=16)
    model.add(LSTM(lstm_units_2))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)))
    
    # Настройка количества нейронов в плотном слое
    dense_units = hp.Int('dense_units', min_value=8, max_value=64, step=8)
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(1))
    
    # Настройка скорости обучения
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    
    return model

# Инициализация тюнера
tuner = kt.Hyperband(
    build_model,
    objective='val_loss',
    max_epochs=50,
    factor=3,
    directory='keras_tuner',
    project_name='radon_tuning'
)

# Поиск оптимальных гиперпараметров
tuner.search(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
])

# Получение лучшей модели
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.hypermodel.build(best_hp)

print("Лучшие гиперпараметры:")
for param, value in best_hp.values.items():
    print(f"{param}: {value}")

# Обучение лучшей модели
best_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
])
```

## Заключение

Улучшенная модель прогнозирования уровня радона включает множество усовершенствований, которые позволяют получить более точные прогнозы и лучше понять факторы, влияющие на уровень радона. Благодаря реализации методов оценки неопределенности и расширенной визуализации, модель не только предсказывает значения, но и помогает интерпретировать результаты и оценить их надежность. 
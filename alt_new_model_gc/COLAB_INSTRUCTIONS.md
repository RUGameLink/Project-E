# Инструкция по созданию и настройке блокнота Google Colab

Следуйте этим шагам, чтобы создать и настроить блокнот Google Colab для использования улучшенной модели прогнозирования уровня радона.

## Шаг 1: Создайте новый блокнот в Google Colab

1. Откройте [Google Colab](https://colab.research.google.com/)
2. Создайте новый блокнот: `Файл -> Новый блокнот`

## Шаг 2: Добавьте код для клонирования репозитория и установки зависимостей

Вставьте следующий код в первую ячейку блокнота и выполните его:

```python
# Клонирование репозитория
!git clone https://github.com/yourusername/Project-E.git
%cd Project-E/alt_new_model_gc

# Установка необходимых пакетов
!pip install numpy pandas matplotlib scikit-learn tensorflow seaborn scipy plotly statsmodels
```

## Шаг 3: Импортируйте необходимые модули

Добавьте новую ячейку с следующим кодом:

```python
# Импорт функций из модулей
from radon_analysis import *
from radon_models import *
from radon_prediction import *

# Отображение графиков Plotly в ноутбуке
import plotly.io as pio
pio.renderers.default = 'colab'

# Для воспроизводимости результатов
import numpy as np
import tensorflow as tf
np.random.seed(42)
tf.random.set_seed(42)
```

## Шаг 4: Создайте или загрузите данные

### Вариант 1: Создание синтетических данных

```python
# Создание образца данных
file_name = create_sample_data(filename='sample_data.csv', periods=1500, frequency='H')

# Загрузка данных
data = load_data(file_name)
```

### Вариант 2: Загрузка собственных данных

```python
# Загрузка собственных данных
from google.colab import files
uploaded = files.upload()  # Загрузите CSV-файл с данными

# Получение имени загруженного файла
file_name = list(uploaded.keys())[0]

# Загрузка и предобработка данных
data = load_data(file_name)
```

## Шаг 5: Проведите анализ данных

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

## Шаг 6: Подготовка данных и обучение моделей

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
# Для экономии времени можно уменьшить количество эпох или выбрать только некоторые типы моделей
models, histories, best_model, saved_paths = train_models(
    X_train, 
    y_train, 
    epochs=50,  # Для демонстрации уменьшим до 50 эпох
    batch_size=32, 
    validation_split=0.2, 
    save_models=True,
    model_types=['lstm', 'gru', 'bidirectional'],  # Базовые модели
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

## Шаг 7: Расширенные модели (опционально)

```python
# Обучение продвинутых моделей (CNN-LSTM и ансамблевая модель)
advanced_models, advanced_histories, advanced_best_model, advanced_saved_paths = train_models(
    X_train, 
    y_train, 
    epochs=50,
    batch_size=32, 
    validation_split=0.2, 
    save_models=True,
    model_types=['cnn_lstm', 'ensemble'],
    l2_reg=0.001
)

# Объединение результатов
models.update(advanced_models)
histories.update(advanced_histories)
saved_paths.update(advanced_saved_paths)

# Выбор лучшей модели среди всех
best_model_type = min(
    [(model_type, min(histories[model_type].history['val_loss'])) for model_type in models.keys()],
    key=lambda x: x[1]
)[0]
best_model = models[best_model_type]
print(f"Лучшая модель: {best_model_type.upper()}")

# Повторная оценка всех моделей
results = evaluate_all_models(models, X_test, y_test, scaler)
```

## Шаг 8: Прогнозирование с оценкой неопределенности

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

## Шаг 9: Использование сохраненной модели

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

## Советы по работе в Google Colab

1. **Сохранение файлов**: Colab автоматически очищает файлы после отключения сессии. Если вы хотите сохранить обученные модели или результаты, загрузите их или сохраните на Google Drive:

   ```python
   # Сохранение файлов на Google Drive
   from google.colab import drive
   drive.mount('/content/drive')
   
   # Копирование файлов на Google Drive
   !cp -r saved_models /content/drive/MyDrive/radon_models
   ```

2. **Использование GPU**: Для ускорения обучения моделей включите GPU в Colab:
   `Изменить -> Настройки блокнота -> Ускоритель -> GPU`

3. **Сохранение блокнота**: Не забудьте сохранить блокнот на Google Drive или скачать его:
   `Файл -> Сохранить копию на Google Drive` или `Файл -> Скачать -> .ipynb`

4. **Предотвращение отключения сессии**: Colab отключает сессию после некоторого времени бездействия. Чтобы этого избежать, выполните:

   ```javascript
   // Выполните в JavaScript-консоли браузера
   function ClickConnect(){
     console.log("Clicking connect button"); 
     document.querySelector("colab-connect-button").click()
   }
   setInterval(ClickConnect, 60000)
   ``` 
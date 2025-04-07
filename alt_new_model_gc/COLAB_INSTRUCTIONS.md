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

## Сохранение и загрузка моделей через Google Drive

В обновленной версии нашего кода добавлена возможность автоматического сохранения моделей на Google Drive. Это позволяет сохранять результаты работы даже при отключении сессии Colab.

### Сохранение моделей на Google Drive

```python
# Подключение Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Импорт необходимых модулей
from radon_models import train_models, prepare_data

# Подготовка данных
X, y, X_train, X_test, y_train, y_test, scaler = prepare_data(data, seq_length=10)

# Обучение моделей с сохранением на Google Drive
save_dir = "/content/drive/MyDrive/saved_models"
models, histories, best_model, saved_paths = train_models(
    X_train, 
    y_train, 
    epochs=100, 
    batch_size=32, 
    save_models=True,
    model_types=['lstm', 'gru'],  # можно указать только нужные модели
    save_dir=save_dir  # путь для сохранения на Google Drive
)

print("Пути к сохраненным моделям:", saved_paths)
```

### Загрузка моделей с Google Drive и прогнозирование

```python
# Подключение Google Drive (если еще не подключен)
from google.colab import drive
drive.mount('/content/drive')

# Импорт функции прогнозирования
from radon_prediction import predict_radon_levels

# Путь к модели на Google Drive
model_path = "/content/drive/MyDrive/saved_models/model_lstm_best_20240422_1045.h5"

# Если модель не найдена по указанному пути, 
# функция автоматически попытается найти модели на Google Drive
predictions, future_predictions, prediction_bounds, fig, fig_importance = predict_radon_levels(
    model_path, 
    data, 
    use_monte_carlo=True
)

# Вывод интерактивных графиков
fig.show()
fig_importance.show()
```

### Советы по работе с Google Drive

1. **Монтирование диска**: При первом использовании потребуется авторизация.
2. **Пути к файлам**: Используйте путь `/content/drive/MyDrive/...` для доступа к файлам.
3. **Сохранение структуры**: Функции создадут директорию `saved_models` автоматически.
4. **Отключение сессии**: Даже если сессия Colab завершится, модели останутся сохраненными на вашем Google Drive.
5. **Совместное использование**: Вы можете открыть доступ к папке с моделями другим пользователям через интерфейс Google Drive.

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

## Использование улучшенных визуализаций и HTML-отчетов

В новой версии кода добавлена возможность создания интерактивных визуализаций с помощью Plotly и сохранения результатов в виде HTML-отчетов.

### Создание интерактивных визуализаций

```python
# Подключение Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Загрузка данных
data = load_data('your_data.csv')

# Получение пути к модели
model_path = "/content/drive/MyDrive/saved_models/model_lstm_best_20240422_1045.h5"

# Создание прогнозов с визуализацией в стиле Plotly
predictions, future_predictions, prediction_bounds, fig_pred, fig_importance = predict_radon_levels(
    model_path, 
    data, 
    use_monte_carlo=True,
    plot_theme='plotly_white'  # Доступны разные темы: 'plotly', 'plotly_dark', 'ggplot2' и др.
)

# Отображение графиков в Colab
fig_pred.show()
fig_importance.show()
```

### Создание и сохранение HTML-отчета с визуализациями

```python
# Создание полного отчета с графиками и сохранение в HTML
predictions, future_predictions, prediction_bounds, fig_pred, fig_importance, html_path = predict_radon_levels(
    model_path, 
    data, 
    use_monte_carlo=True,
    plot_theme='plotly_white',
    save_html=True,  # Включение сохранения в HTML
    html_filename='radon_prediction_report.html'  # Имя выходного файла
)

print(f"Отчет сохранен в: {html_path}")
```

HTML-отчет будет автоматически сохранен на Google Drive в папке `/content/drive/MyDrive/radon_reports/` и будет доступен для скачивания. Отчет включает следующие разделы:

1. **Временные ряды признаков** - График с данными радона, температуры и давления
2. **Прогнозирование уровня радона** - График с фактическими значениями, прогнозами и доверительными интервалами
3. **Влияние признаков на прогноз** - График, показывающий важность разных временных шагов для прогноза
4. **Распределение значений радона** - Гистограмма распределения значений уровня радона
5. **Корреляция между признаками** - Тепловая карта корреляций между признаками

### Создание пользовательских отчетов

Вы также можете создать собственный HTML-отчет с нужным набором графиков:

```python
# Импорт необходимого модуля
from radon_prediction import save_plots_to_html

# Создание графиков
fig1 = plot_time_series(data, plot_type='plotly')
fig2 = plot_predictions(data, predictions, future_predictions, prediction_bounds)
fig3 = plot_feature_importance(model, data)

# Объединение графиков в словарь с заголовками разделов
plots_dict = {
    "Временные ряды": fig1,
    "Прогнозы уровня радона": fig2,
    "Важность признаков": fig3
}

# Сохранение в HTML с пользовательским заголовком
html_path = save_plots_to_html(
    plots_dict,
    filename='my_custom_report.html',
    title='Мой отчет по анализу радона'
)
```

Это позволяет создавать полноценные интерактивные отчеты, которые можно легко передавать коллегам или интегрировать в веб-приложения.

## Исправление импортов для TensorFlow 2.x

В последних версиях TensorFlow/Keras произошли изменения в структуре импортов. При возникновении ошибки вида:

```
ImportError: cannot import name 'mean_squared_error' from 'tensorflow.keras.losses'
```

Необходимо исправить импорты в файлах проекта:

1. **Для radon_models.py**:
   ```python
   # Замените импорт
   from tensorflow.keras.losses import mean_squared_error
   
   # На этот вариант
   import tensorflow.keras.losses as klosses
   ```

2. **Для функции create_model**:
   ```python
   # Замените
   model.compile(optimizer=Adam(learning_rate=0.001), loss=mean_squared_error)
   
   # На этот вариант
   model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
   ```

3. **Для функции predict_radon_levels в radon_prediction.py**:
   ```python
   # Замените
   from tensorflow.keras.losses import mean_squared_error
   from tensorflow.keras.metrics import mean_squared_error as mse
   
   model = load_model(model_path, custom_objects={
       'mse': mse,
       'mean_squared_error': mean_squared_error
   })
   
   # На этот вариант
   model = load_model(model_path, compile=True)
   ```

Эти изменения обеспечат совместимость с последними версиями TensorFlow/Keras и устранят ошибки импорта. 
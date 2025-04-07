# Radon Prediction Model для Google Colab

Этот проект содержит исходные файлы для создания и запуска моделей прогнозирования уровня радона на основе данных о температуре и давлении. Проект оптимизирован для запуска в Google Colab.

## Структура проекта

Для создания блокнота Colab используйте следующие исходные файлы:

1. `radon_analysis.py` - код для анализа данных и визуализации
2. `radon_models.py` - код моделей нейронных сетей (LSTM, GRU, BiLSTM)
3. `radon_prediction.py` - код для прогнозирования уровня радона

## Инструкции по использованию в Google Colab

1. Создайте новый блокнот в Google Colab
2. Добавьте следующий код для установки необходимых пакетов:

```python
# Install required packages
!pip install numpy pandas matplotlib scikit-learn tensorflow seaborn scipy
```

3. Загрузите исходные файлы из GitHub или загрузите их напрямую в Colab:

```python
# Если исходные файлы находятся в репозитории GitHub
!git clone https://github.com/yourusername/Project-E.git
!cp Project-E/alt_model_gc/*.py ./

# Если вы загружаете файлы вручную
from google.colab import files
uploaded = files.upload()  # Загрузите radon_analysis.py, radon_models.py и radon_prediction.py
```

4. Импортируйте функции из загруженных файлов:

```python
from radon_analysis import *
from radon_models import *
from radon_prediction import *
```

5. Загрузите данные:

```python
# Загрузка собственных данных
from google.colab import files
uploaded = files.upload()  # Загрузите CSV-файл с данными

# Загрузка и предобработка данных
data = load_data(list(uploaded.keys())[0])
```

6. Выполните анализ данных:

```python
# Визуализация временных рядов
plot_time_series(data)

# Анализ корреляций
corr_matrix = analyze_correlations(data)

# Построение диаграмм рассеяния
plot_scatter_relationships(data)

# Анализ статистики
analyze_statistics(data)

# Анализ лагов
lag_correlations = perform_lag_analysis(data)
```

7. Обучите модели:

```python
# Подготовка данных
X, y, X_train, X_test, y_train, y_test, scaler = prepare_data(data)

# Обучение моделей
models, histories, best_model = train_models(X_train, y_train)

# Оценка моделей
evaluate_all_models(models, X_test, y_test, scaler)
```

8. Сделайте прогнозы:

```python
# Прогнозирование
predictions, future_predictions = make_predictions(best_model, data, X_sequences, scaler)

# Визуализация результатов
plot_predictions(data, predictions, future_predictions)
```

## Создание полного блокнота

Для простоты использования рекомендуется скопировать содержимое всех трех файлов в один блокнот Colab и выполнять его последовательно. Таким образом, вы сможете:

1. Установить необходимые пакеты
2. Загрузить и проанализировать данные 
3. Создать и обучить модели
4. Сделать прогнозы
5. Визуализировать результаты

## Пример использования с образцом данных

Если у вас нет собственных данных, вы можете использовать генератор образцов:

```python
# Создание образца данных
dates = pd.date_range(start='2023-01-01', periods=1000, freq='H')
temperature = 20 + 5 * np.sin(np.arange(1000) * 2 * np.pi / 24) + np.random.normal(0, 1, 1000)
pressure = 1013 + 10 * np.sin(np.arange(1000) * 2 * np.pi / (24 * 7)) + np.random.normal(0, 3, 1000)
radon = 20 + 0.5 * temperature + 0.05 * pressure + 10 * np.sin(np.arange(1000) * 2 * np.pi / (24 * 3)) + np.random.normal(0, 5, 1000)

# Создание DataFrame
df = pd.DataFrame({
    'Datetime': dates.strftime('%d.%m.%Y %H:%M'),
    'Radon (Bq.m3)': radon,
    'Temperature (¡C)': temperature.astype(str).str.replace('.', ','),
    'Pressure (mBar)': pressure.astype(str).str.replace('.', ',')
})

# Сохранение в CSV
df.to_csv('sample_data.csv', sep=';', index=False)

# Загрузка данных
data = load_data('sample_data.csv')
``` 
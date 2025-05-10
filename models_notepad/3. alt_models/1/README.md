# Модель прогнозирования уровня радона для Google Colab

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
# Установка необходимых пакетов
!pip install numpy pandas matplotlib scikit-learn tensorflow seaborn scipy plotly
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

# Обучение моделей (с сохранением)
models, histories, best_model, saved_paths = train_models(X_train, y_train)

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
evaluate_all_models(models, X_test, y_test, scaler)
```

8. Сделайте прогнозы:

```python
# Прогнозирование
predictions, future_predictions = make_predictions(best_model, data)

# Интерактивная визуализация результатов с помощью Plotly
fig = plot_predictions(data, predictions, future_predictions)
fig.show()

# Или для статической визуализации
plot_predictions_static(data, predictions, future_predictions)
```

9. Загрузка сохраненной модели для прогнозирования:

```python
# Загрузка сохраненной модели
from tensorflow.keras.models import load_model
saved_model_path = "saved_models/model_lstm_best_20230605_1415.h5"  # Укажите путь к вашей сохраненной модели
loaded_model = load_model(saved_model_path)

# Прогнозирование с использованием загруженной модели
predictions, future_predictions = make_predictions(loaded_model, data)

# Визуализация результатов
fig = plot_predictions(data, predictions, future_predictions)
fig.show()
```

## Сохранение моделей и истории обучения

Проект теперь поддерживает автоматическое сохранение моделей и истории их обучения. При обучении каждой модели (LSTM, GRU, BiLSTM) происходит следующее:

1. Все модели сохраняются в формате `.h5` в директории `saved_models` с именем в формате: `model_{rnn_name}_{date}_{time}.h5`
   - `rnn_name`: тип модели (lstm, gru, bidirectional)
   - `date`: дата сохранения в формате ГГГГММДД
   - `time`: время сохранения в формате ЧЧММ

2. История обучения каждой модели сохраняется в формате JSON в файлах вида: `history_{rnn_name}_{date}_{time}.json`

3. Дополнительно, лучшая модель (с наименьшей ошибкой на валидационном наборе) сохраняется с отметкой `best` в имени файла.

Для отключения автоматического сохранения, можно передать параметр `save_models=False` в функцию `train_models`.

## Интерактивная визуализация с Plotly

В проект добавлена поддержка интерактивных графиков с помощью библиотеки Plotly. Функция `plot_predictions` теперь создает интерактивный график с возможностями:

- Наведение курсора для просмотра точных значений
- Масштабирование и перемещение по графику
- Ползунок для выбора временного диапазона
- Кнопки для выбора периода (1 день, 1 неделя, 1 месяц)
- Сохранение графика в различных форматах

Для статической визуализации по-прежнему доступна функция `plot_predictions_static`.

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
"""
Модуль прогнозирования уровня радона

Этот модуль содержит функции для прогнозирования уровня радона с использованием обученных моделей.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model


def create_sequences_for_prediction(data, seq_length):
    """Создание последовательностей для прогнозирования временных рядов."""
    X = []
    for i in range(len(data) - seq_length + 1):
        X.append(data[i:(i + seq_length), 1:])  # Только температура и давление
    return np.array(X)


def make_future_prediction(model, last_sequence, scaler, steps=24):
    """Создание прогнозов на будущие временные шаги."""
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(steps):
        # Изменение формы для прогнозирования
        input_seq = current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1])
        
        # Прогнозирование следующего значения
        pred = model.predict(input_seq, verbose=0)
        
        # Создание объединенного массива для обратного масштабирования (радон, температура, давление)
        dummy = np.zeros((1, 3))
        dummy[0, 0] = pred[0, 0]  # Прогнозируемый уровень радона
        
        # Получение последних значений температуры и давления
        # Это упрощение - в реальности нужно также прогнозировать эти значения
        dummy[0, 1:] = current_sequence[-1]
        
        # Обратное преобразование масштаба
        unscaled_pred = scaler.inverse_transform(dummy)[0, 0]
        future_predictions.append(unscaled_pred)
        
        # Обновление последовательности путем удаления первой и добавления прогнозируемой строки
        # Создание новой строки с прогнозируемым радоном (установлен на 0, так как мы используем только температуру и давление)
        # и последними значениями температуры и давления
        new_row = np.zeros(current_sequence.shape[1])
        new_row[:] = current_sequence[-1]  # Копирование последних значений температуры и давления
        
        # Удаление первой строки и добавление новой
        current_sequence = np.vstack([current_sequence[1:], new_row])
    
    return future_predictions


def plot_predictions(data, predictions, future_predictions=None):
    """Визуализация прогнозов с использованием Plotly."""
    # Подготовка данных для визуализации
    dates = data.index[-len(predictions):]
    actual = data['Radon (Bq.m3)'].iloc[-len(predictions):].values
    
    # Создание фигуры с двумя y-осями
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Добавление линии фактических значений
    fig.add_trace(
        go.Scatter(x=dates, y=actual, name="Фактический уровень радона", line=dict(color='blue')),
        secondary_y=False,
    )
    
    # Добавление линии прогнозов
    fig.add_trace(
        go.Scatter(x=dates, y=predictions, name="Прогнозируемый уровень радона", line=dict(color='red')),
        secondary_y=False,
    )
    
    # Добавление температуры
    fig.add_trace(
        go.Scatter(x=dates, y=data['Temperature (°C)'].iloc[-len(predictions):].values, name="Температура (°C)", 
                  line=dict(color='orange', dash='dot')),
        secondary_y=True,
    )
    
    # Добавление давления
    fig.add_trace(
        go.Scatter(x=dates, y=data['Pressure (mBar)'].iloc[-len(predictions):].values / 10, name="Давление / 10 (mBar)", 
                  line=dict(color='green', dash='dot')),
        secondary_y=True,
    )
    
    # Если есть прогнозы на будущее, добавляем их
    if future_predictions is not None and len(future_predictions) > 0:
        # Создаем даты для будущих прогнозов
        last_date = dates[-1]
        
        # Исправление для предотвращения ошибки с Timestamp
        if isinstance(last_date, pd.Timestamp):
            # Определим частоту данных из существующего индекса
            freq = pd.infer_freq(data.index)
            if freq is None:
                # Если не удалось определить частоту, предположим часовой интервал
                freq = 'H'
            
            # Создаем новые даты с правильным интервалом
            future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), 
                                         periods=len(future_predictions), 
                                         freq=freq)
        else:
            # Для неиндексированных данных
            future_dates = [last_date + i + 1 for i in range(len(future_predictions))]
        
        # Добавляем будущие прогнозы на график
        fig.add_trace(
            go.Scatter(x=future_dates, y=future_predictions, name="Прогноз на будущее", 
                      line=dict(color='purple', dash='dash')),
            secondary_y=False,
        )
    
    # Обновление макета
    fig.update_layout(
        title="Прогнозирование уровня радона",
        xaxis_title="Дата",
        legend=dict(y=0.99, x=0.01, orientation="h"),
        hovermode="x unified",
        height=600,
    )
    
    # Настройка осей Y
    fig.update_yaxes(title_text="Уровень радона (Бк/м³)", secondary_y=False)
    fig.update_yaxes(title_text="Значение", secondary_y=True)
    
    # Добавление ползунка и кнопок для выбора диапазона
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1 день", step="day", stepmode="backward"),
                    dict(count=7, label="1 неделя", step="day", stepmode="backward"),
                    dict(count=1, label="1 месяц", step="month", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    
    return fig


def plot_predictions_static(data, predictions, future_predictions=None):
    """Статическая визуализация прогнозов с использованием Matplotlib."""
    # Подготовка данных для визуализации
    dates = data.index[-len(predictions):]
    actual = data['Radon (Bq.m3)'].iloc[-len(predictions):].values
    
    plt.figure(figsize=(14, 8))
    
    # Построение графиков на основной оси Y
    plt.plot(dates, actual, label='Фактический уровень радона', color='blue')
    plt.plot(dates, predictions, label='Прогнозируемый уровень радона', color='red')
    
    # Если есть прогнозы на будущее, добавляем их
    if future_predictions is not None and len(future_predictions) > 0:
        # Создаем даты для будущих прогнозов
        last_date = dates[-1]
        
        # Исправление для предотвращения ошибки с Timestamp
        if isinstance(last_date, pd.Timestamp):
            # Определим частоту данных из существующего индекса
            freq = pd.infer_freq(data.index)
            if freq is None:
                # Если не удалось определить частоту, предположим часовой интервал
                freq = 'H'
            
            # Создаем новые даты с правильным интервалом
            future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), 
                                         periods=len(future_predictions), 
                                         freq=freq)
        else:
            # Для неиндексированных данных
            future_dates = [last_date + i + 1 for i in range(len(future_predictions))]
        
        plt.plot(future_dates, future_predictions, label='Прогноз на будущее', color='purple', linestyle='--')
    
    plt.title('Прогнозирование уровня радона')
    plt.xlabel('Дата')
    plt.ylabel('Уровень радона (Бк/м³)')
    plt.legend(loc='upper left')
    plt.grid(True)
    
    # Создание второй оси Y для температуры и давления
    ax2 = plt.gca().twinx()
    ax2.plot(dates, data['Temperature (°C)'].iloc[-len(predictions):].values, 
             label='Температура (°C)', color='orange', linestyle=':')
    ax2.plot(dates, data['Pressure (mBar)'].iloc[-len(predictions):].values / 10, 
             label='Давление / 10 (mBar)', color='green', linestyle=':')
    
    # Настройка второй оси Y
    ax2.set_ylabel('Значение')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('radon_prediction_result.png', dpi=300)
    plt.show()


def make_predictions(model, data, seq_length=5, future_steps=24):
    """
    Генерирует прогнозы и будущие прогнозы на основе обученной модели.
    
    Args:
        model: Обученная модель Keras
        data: DataFrame с данными, содержащий колонки 'Radon (Bq.m3)', 'Temperature (°C)', 'Pressure (mBar)'
        seq_length: Длина входной последовательности (должна соответствовать обученной модели)
        future_steps: Количество шагов для прогноза в будущее
        
    Returns:
        predictions: Массив прогнозов для имеющихся данных
        future_predictions: Массив прогнозов на future_steps шагов вперед
    """
    # Подготовка данных
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Radon (Bq.m3)', 'Temperature (°C)', 'Pressure (mBar)']].values)
    
    # Создание последовательностей для прогнозирования
    X, y = create_sequences(scaled_data, seq_length)
    
    # Прогнозирование
    scaled_predictions = model.predict(X)
    
    # Обратное преобразование прогнозов
    dummy_array = np.zeros((len(scaled_predictions), 3))
    dummy_array[:, 0] = scaled_predictions.flatten()
    predictions = scaler.inverse_transform(dummy_array)[:, 0]
    
    # Генерация прогнозов на будущее
    future_predictions = make_future_prediction(model, scaled_data, scaler, seq_length, future_steps)
    
    print(f"Создано {len(predictions)} прогнозов для существующих данных")
    print(f"Создано {len(future_predictions)} прогнозов на будущее")
    
    return predictions, future_predictions


def create_sequences(data, seq_length):
    """Создание последовательностей для прогнозирования временных рядов."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), 1:])  # Только температура и давление
        y.append(data[i + seq_length, 0])       # Уровень радона
    return np.array(X), np.array(y)


def make_future_prediction(model, scaled_data, scaler, seq_length, future_steps):
    """
    Генерирует прогнозы на несколько шагов вперед.
    
    Args:
        model: Обученная модель
        scaled_data: Масштабированные данные
        scaler: Объект MinMaxScaler для обратного преобразования
        seq_length: Длина входной последовательности
        future_steps: Количество шагов для прогноза в будущее
        
    Returns:
        future_predictions: Массив прогнозов на future_steps шагов вперед
    """
    # Последняя доступная последовательность
    last_sequence = scaled_data[-seq_length:].copy()
    
    # Массив для будущих прогнозов
    future_predictions = []
    
    # Генерация прогнозов
    current_sequence = last_sequence.copy()
    
    for _ in range(future_steps):
        # Подготовка входных данных для модели (только температура и давление)
        X_future = current_sequence[:, 1:].reshape(1, seq_length, 2)
        
        # Прогнозирование
        next_pred = model.predict(X_future, verbose=0)[0][0]
        
        # Создание строки с прогнозом (копируем последние значения температуры и давления)
        next_point = np.zeros(3)
        next_point[0] = next_pred
        next_point[1:] = current_sequence[-1, 1:]  # Используем последние значения темп. и давления
        
        # Обратное преобразование прогноза
        dummy_array = np.zeros((1, 3))
        dummy_array[0] = next_point
        next_point_inverse = scaler.inverse_transform(dummy_array)[0, 0]
        
        # Добавление прогноза в результаты
        future_predictions.append(next_point_inverse)
        
        # Обновление последовательности для следующего шага
        # Удаляем первую точку и добавляем новую в конец
        new_seq = np.zeros_like(current_sequence)
        new_seq[:-1] = current_sequence[1:]
        new_seq[-1] = next_point
        current_sequence = new_seq
    
    return np.array(future_predictions)


def predict_radon_levels(model_path, data):
    """Загрузка модели и создание прогнозов."""
    try:
        # Загрузка модели
        from tensorflow.keras.models import load_model
        model = load_model(model_path)
        
        # Создание прогнозов
        predictions, future_predictions = make_predictions(model, data)
        
        # Создание интерактивного графика
        fig = plot_predictions(data, predictions, future_predictions)
        
        return predictions, future_predictions, fig
    except Exception as e:
        print(f"Ошибка при прогнозировании: {e}")
        return None, None, None 
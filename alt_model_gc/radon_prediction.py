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


def plot_predictions(data, predictions, future_predictions=None, n_past=100):
    """Интерактивная визуализация фактических данных и прогнозов с помощью Plotly."""
    # Создание фигуры
    fig = go.Figure()
    
    # Отображение фактических значений
    fig.add_trace(go.Scatter(
        x=data.index[-n_past:], 
        y=data['Radon (Bq.m3)'].values[-n_past:],
        mode='lines',
        name='Фактический уровень радона',
        line=dict(color='blue', width=2)
    ))
    
    # Отображение прогнозируемых значений
    if predictions is not None:
        fig.add_trace(go.Scatter(
            x=data.index[-len(predictions):], 
            y=predictions,
            mode='lines',
            name='Прогнозируемый уровень радона',
            line=dict(color='red', width=2)
        ))
    
    # Отображение будущих прогнозов
    if future_predictions is not None:
        # Создание будущих дат
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date, periods=len(future_predictions)+1, freq='H')[1:]
        
        fig.add_trace(go.Scatter(
            x=future_dates, 
            y=future_predictions,
            mode='lines+markers',
            name='Прогноз на будущее',
            line=dict(color='green', width=2, dash='dash')
        ))
        
        # Добавление вертикальной линии для разделения фактических данных и прогнозов
        fig.add_vline(
            x=last_date, 
            line=dict(color='black', width=1, dash='dash'),
            annotation_text="Будущее",
            annotation_position="top right"
        )
    
    # Обновление макета
    fig.update_layout(
        title='Прогнозирование уровня радона',
        xaxis_title='Дата',
        yaxis_title='Уровень радона (Бк/м³)',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template='plotly_white',
        height=600,
        width=1000
    )
    
    # Добавление ползунка диапазона
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1д", step="day", stepmode="backward"),
                    dict(count=7, label="1н", step="day", stepmode="backward"),
                    dict(count=1, label="1м", step="month", stepmode="backward"),
                    dict(step="all", label="Все")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    
    # Возвращение фигуры
    return fig


def plot_predictions_static(data, predictions, future_predictions=None, n_past=100):
    """Статическая визуализация фактических данных и прогнозов с помощью Matplotlib."""
    plt.figure(figsize=(14, 7))
    
    # Отображение фактических значений
    plt.plot(data.index[-n_past:], data['Radon (Bq.m3)'].values[-n_past:], 
             'b-', label='Фактический уровень радона')
    
    # Отображение прогнозируемых значений
    if predictions is not None:
        plt.plot(data.index[-len(predictions):], predictions, 
                 'r-', label='Прогнозируемый уровень радона')
    
    # Отображение будущих прогнозов
    if future_predictions is not None:
        # Создание будущих дат
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date, periods=len(future_predictions)+1, freq='H')[1:]
        
        plt.plot(future_dates, future_predictions, 'g--', label='Прогноз на будущее')
        
        # Добавление вертикальной линии для разделения фактических данных и прогнозов
        plt.axvline(x=last_date, color='k', linestyle='--')
        plt.text(last_date, plt.ylim()[1]*0.9, 'Будущее', ha='right')
        plt.text(future_dates[0], plt.ylim()[1]*0.9, 'Будущее', ha='left')
    
    plt.title('Прогнозирование уровня радона')
    plt.xlabel('Дата')
    plt.ylabel('Уровень радона (Бк/м³)')
    plt.legend()
    plt.tight_layout()
    plt.show()


def make_predictions(model, data, seq_length=5):
    """Создание прогнозов на основе существующих данных и прогнозов на будущее."""
    # Масштабирование данных
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Создание последовательностей для прогнозирования
    X_sequences = create_sequences_for_prediction(scaled_data, seq_length)
    print(f"Создано {len(X_sequences)} последовательностей для прогнозирования")
    
    # Прогнозирование на всех последовательностях
    print("Выполнение прогнозирования...")
    predictions = model.predict(X_sequences)
    
    # Обратное преобразование масштаба прогнозов
    print("Преобразование прогнозов обратно в исходный масштаб...")
    unscaled_predictions = np.zeros((len(predictions), 3))
    unscaled_predictions[:, 0] = predictions.flatten()
    unscaled_predictions = scaler.inverse_transform(unscaled_predictions)[:, 0]
    
    # Прогнозирование на будущее (следующие 24 часа)
    print("Создание прогнозов на будущее...")
    future_pred = make_future_prediction(
        model, 
        X_sequences[-1], 
        scaler, 
        steps=24
    )
    
    return unscaled_predictions, future_pred


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
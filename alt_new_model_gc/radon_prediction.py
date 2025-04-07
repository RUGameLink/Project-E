"""
Модуль прогнозирования уровня радона

Этот модуль содержит функции для прогнозирования уровня радона с использованием обученных моделей.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import load_model


def create_sequences_for_prediction(data, seq_length):
    """Создание последовательностей для прогнозирования временных рядов."""
    X = []
    for i in range(len(data) - seq_length + 1):
        X.append(data[i:(i + seq_length), 1:])  # Только температура и давление
    return np.array(X)


def make_future_prediction(model, last_sequence, scaler, steps=48, use_monte_carlo=False, num_simulations=50):
    """
    Создание прогнозов на будущие временные шаги с возможностью применения метода Монте-Карло.
    
    Args:
        model: Обученная модель
        last_sequence: Последняя последовательность данных
        scaler: Масштабировщик для обратного преобразования
        steps: Количество шагов для прогнозирования
        use_monte_carlo: Использовать ли метод Монте-Карло для оценки неопределенности
        num_simulations: Количество симуляций для метода Монте-Карло
        
    Returns:
        future_predictions: Прогнозируемые значения
        prediction_bounds: Границы доверительного интервала (если use_monte_carlo=True)
    """
    future_predictions = []
    prediction_bounds = None
    
    if use_monte_carlo:
        # Метод Монте-Карло для оценки неопределенности
        all_simulations = []
        
        for _ in range(num_simulations):
            current_sequence = last_sequence.copy()
            simulation_predictions = []
            
            for _ in range(steps):
                # Изменение формы для прогнозирования
                input_seq = current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1])
                
                # Прогнозирование следующего значения
                pred = model.predict(input_seq, verbose=0)
                
                # Добавление небольшого шума для симуляции вариативности
                noise = np.random.normal(0, 0.02, pred.shape)
                pred = pred + noise
                
                # Создание объединенного массива для обратного масштабирования
                dummy = np.zeros((1, 3))
                dummy[0, 0] = pred[0, 0]
                dummy[0, 1:] = current_sequence[-1]
                
                # Обратное преобразование масштаба
                unscaled_pred = scaler.inverse_transform(dummy)[0, 0]
                simulation_predictions.append(unscaled_pred)
                
                # Обновление последовательности
                new_row = np.zeros(current_sequence.shape[1])
                new_row[:] = current_sequence[-1]
                current_sequence = np.vstack([current_sequence[1:], new_row])
            
            all_simulations.append(simulation_predictions)
        
        # Преобразование в numpy массив
        all_simulations = np.array(all_simulations)
        
        # Расчет среднего и границ доверительного интервала (95%)
        mean_predictions = np.mean(all_simulations, axis=0)
        lower_bound = np.percentile(all_simulations, 2.5, axis=0)
        upper_bound = np.percentile(all_simulations, 97.5, axis=0)
        
        future_predictions = mean_predictions
        prediction_bounds = (lower_bound, upper_bound)
    else:
        # Стандартное прогнозирование
        current_sequence = last_sequence.copy()
        
        for _ in range(steps):
            # Изменение формы для прогнозирования
            input_seq = current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1])
            
            # Прогнозирование следующего значения
            pred = model.predict(input_seq, verbose=0)
            
            # Создание объединенного массива для обратного масштабирования
            dummy = np.zeros((1, 3))
            dummy[0, 0] = pred[0, 0]  # Прогнозируемый уровень радона
            
            # Получение последних значений температуры и давления
            dummy[0, 1:] = current_sequence[-1]
            
            # Обратное преобразование масштаба
            unscaled_pred = scaler.inverse_transform(dummy)[0, 0]
            future_predictions.append(unscaled_pred)
            
            # Обновление последовательности
            new_row = np.zeros(current_sequence.shape[1])
            new_row[:] = current_sequence[-1]
            current_sequence = np.vstack([current_sequence[1:], new_row])
    
    return future_predictions, prediction_bounds


def plot_predictions(data, predictions, future_predictions=None, prediction_bounds=None, 
                    radon_col=None, temp_col=None, pressure_col=None, plot_title="Прогнозирование уровня радона",
                    plot_theme='plotly', include_components=True):
    """
    Интерактивная визуализация прогнозов с использованием Plotly.
    
    Args:
        data: DataFrame с данными
        predictions: Прогнозы для существующих данных
        future_predictions: Прогнозы на будущее (опционально)
        prediction_bounds: Границы доверительного интервала (опционально)
        radon_col: Название колонки с данными о радоне
        temp_col: Название колонки с данными о температуре
        pressure_col: Название колонки с данными о давлении
        plot_title: Заголовок графика
        plot_theme: Тема оформления графика ('plotly', 'plotly_white', 'plotly_dark', 'ggplot2', и т.д.)
        include_components: Включать ли компоненты влияния на график
        
    Returns:
        fig: Объект графика Plotly
    """
    # Определение колонок, если они не указаны
    if radon_col is None or temp_col is None or pressure_col is None:
        columns = data.columns
        
        # Определение названий колонок для радона, температуры и давления
        if radon_col is None:
            for col in columns:
                if 'radon' in col.lower():
                    radon_col = col
                    break
            if radon_col is None:
                radon_col = columns[0]  # По умолчанию первая колонка
        
        if temp_col is None:
            for col in columns:
                if 'temp' in col.lower():
                    temp_col = col
                    break
            if temp_col is None and len(columns) > 1:
                temp_col = columns[1]  # По умолчанию вторая колонка
        
        if pressure_col is None:
            for col in columns:
                if 'pressure' in col.lower() or 'давлен' in col.lower():
                    pressure_col = col
                    break
            if pressure_col is None and len(columns) > 2:
                pressure_col = columns[2]  # По умолчанию третья колонка
    
    # Подготовка данных для визуализации
    dates = data.index[-len(predictions):]
    actual = data[radon_col].iloc[-len(predictions):].values
    
    # Установка темы
    if plot_theme:
        template = plot_theme
    else:
        template = "plotly"
    
    # Создание фигуры с двумя y-осями
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Добавление линии фактических значений
    fig.add_trace(
        go.Scatter(x=dates, y=actual, name="Фактический уровень радона", 
                  line=dict(color='blue', width=2)),
        secondary_y=False,
    )
    
    # Добавление линии прогнозов
    fig.add_trace(
        go.Scatter(x=dates, y=predictions, name="Прогнозируемый уровень радона", 
                  line=dict(color='red', width=2)),
        secondary_y=False,
    )
    
    # Добавление компонентов влияния если требуется
    if include_components:
        # Добавление температуры, если колонка существует
        if temp_col in data.columns:
            fig.add_trace(
                go.Scatter(x=dates, y=data[temp_col].iloc[-len(predictions):].values, 
                          name=f"{temp_col}", line=dict(color='orange', dash='dot')),
                secondary_y=True,
            )
        
        # Добавление давления, если колонка существует
        if pressure_col in data.columns:
            # Делим на 10 для лучшего масштабирования на графике
            fig.add_trace(
                go.Scatter(x=dates, y=data[pressure_col].iloc[-len(predictions):].values / 10, 
                          name=f"{pressure_col} / 10", line=dict(color='green', dash='dot')),
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
                      line=dict(color='purple', width=2.5, dash='dash')),
            secondary_y=False,
        )
        
        # Если есть границы доверительного интервала, добавляем их
        if prediction_bounds is not None:
            lower_bound, upper_bound = prediction_bounds
            
            # Добавляем заливку для доверительного интервала
            fig.add_trace(
                go.Scatter(
                    x=list(future_dates) + list(future_dates)[::-1],
                    y=list(upper_bound) + list(lower_bound)[::-1],
                    fill='toself',
                    fillcolor='rgba(128, 0, 128, 0.2)',
                    line=dict(color='rgba(255, 255, 255, 0)'),
                    hoverinfo='skip',
                    showlegend=True,
                    name='95% доверительный интервал'
                ),
                secondary_y=False,
            )
    
    # Обновление макета
    fig.update_layout(
        title=dict(
            text=plot_title,
            font=dict(size=20)
        ),
        xaxis_title="Дата и время",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        ),
        hovermode="x unified",
        height=600,
        template=template,
        margin=dict(l=60, r=40, t=80, b=60)
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
                    dict(step="all", label="Весь период")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    
    return fig


def plot_predictions_static(data, predictions, future_predictions=None, prediction_bounds=None,
                           radon_col=None, temp_col=None, pressure_col=None):
    """Статическая визуализация прогнозов с использованием Matplotlib."""
    # Определение колонок, если они не указаны
    if radon_col is None or temp_col is None or pressure_col is None:
        columns = data.columns
        
        # Определение названий колонок для радона, температуры и давления
        if radon_col is None:
            for col in columns:
                if 'radon' in col.lower():
                    radon_col = col
                    break
            if radon_col is None:
                radon_col = columns[0]  # По умолчанию первая колонка
        
        if temp_col is None:
            for col in columns:
                if 'temp' in col.lower():
                    temp_col = col
                    break
            if temp_col is None and len(columns) > 1:
                temp_col = columns[1]  # По умолчанию вторая колонка
        
        if pressure_col is None:
            for col in columns:
                if 'pressure' in col.lower() or 'давлен' in col.lower():
                    pressure_col = col
                    break
            if pressure_col is None and len(columns) > 2:
                pressure_col = columns[2]  # По умолчанию третья колонка
    
    # Подготовка данных для визуализации
    dates = data.index[-len(predictions):]
    actual = data[radon_col].iloc[-len(predictions):].values
    
    # Создание фигуры с двумя осями y
    fig, ax1 = plt.subplots(figsize=(14, 8))
    ax2 = ax1.twinx()
    
    # Построение графиков на основной оси Y
    ax1.plot(dates, actual, label='Фактический уровень радона', color='blue', linewidth=2)
    ax1.plot(dates, predictions, label='Прогнозируемый уровень радона', color='red', linewidth=2)
    
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
        ax1.plot(future_dates, future_predictions, label='Прогноз на будущее', 
                color='purple', linestyle='--', linewidth=2.5)
        
        # Если есть границы доверительного интервала, добавляем их
        if prediction_bounds is not None:
            lower_bound, upper_bound = prediction_bounds
            ax1.fill_between(future_dates, lower_bound, upper_bound, 
                           color='purple', alpha=0.2, label='95% доверительный интервал')
    
    # Добавление температуры и давления на второй оси Y
    if temp_col in data.columns:
        ax2.plot(dates, data[temp_col].iloc[-len(predictions):].values, 
                label=f'{temp_col}', color='orange', linestyle=':', linewidth=1.5)
    
    if pressure_col in data.columns:
        ax2.plot(dates, data[pressure_col].iloc[-len(predictions):].values / 10, 
                label=f'{pressure_col} / 10', color='green', linestyle=':', linewidth=1.5)
    
    # Настройка осей и легенды
    ax1.set_xlabel('Дата и время', fontsize=12)
    ax1.set_ylabel('Уровень радона (Бк/м³)', fontsize=12)
    ax2.set_ylabel('Значение', fontsize=12)
    
    # Объединение легенд с обеих осей
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', 
             bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=10)
    
    # Настройка сетки и заголовка
    ax1.grid(True, alpha=0.3)
    plt.title('Прогнозирование уровня радона', fontsize=16)
    plt.tight_layout()
    
    return fig


def make_predictions(model, data, seq_length=10, future_steps=48, use_monte_carlo=False, num_simulations=50):
    """
    Генерирует прогнозы и будущие прогнозы на основе обученной модели.
    
    Args:
        model: Обученная модель Keras
        data: DataFrame с данными
        seq_length: Длина входной последовательности (должна соответствовать обученной модели)
        future_steps: Количество шагов для прогноза в будущее
        use_monte_carlo: Использовать ли метод Монте-Карло для оценки неопределенности
        num_simulations: Количество симуляций для метода Монте-Карло
        
    Returns:
        predictions: Массив прогнозов для имеющихся данных
        future_predictions: Массив прогнозов на future_steps шагов вперед
        prediction_bounds: Границы доверительного интервала (если use_monte_carlo=True)
    """
    # Проверка и нормализация названий колонок
    columns = data.columns
    
    # Определение названий колонок для радона, температуры и давления
    radon_col = None
    temp_col = None
    pressure_col = None
    
    # Поиск подходящих колонок
    for col in columns:
        col_lower = col.lower()
        if 'radon' in col_lower:
            radon_col = col
        elif 'temp' in col_lower:
            temp_col = col
        elif 'pressure' in col_lower or 'давлен' in col_lower:
            pressure_col = col
    
    # Проверка наличия необходимых колонок
    if not all([radon_col, temp_col, pressure_col]):
        missing = []
        if not radon_col: missing.append("радона")
        if not temp_col: missing.append("температуры")
        if not pressure_col: missing.append("давления")
        raise ValueError(f"Не найдены колонки для: {', '.join(missing)}. "
                        f"Доступные колонки: {', '.join(columns)}")
    
    # Подготовка данных
    data_array = data[[radon_col, temp_col, pressure_col]].values
    
    # Масштабирование данных
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_array)
    
    # Создание последовательностей для прогнозирования
    X = create_sequences_for_prediction(scaled_data, seq_length)
    
    # Прогнозирование
    predictions = model.predict(X)
    
    # Обратное масштабирование
    predictions_unscaled = np.zeros((len(predictions), 3))
    predictions_unscaled[:, 0] = predictions.flatten()
    predictions_unscaled = scaler.inverse_transform(predictions_unscaled)[:, 0]
    
    # Создание прогноза на будущее
    last_sequence = scaled_data[-seq_length:]
    future_sequence = np.zeros((seq_length, 2))
    for i in range(seq_length):
        future_sequence[i] = last_sequence[i, 1:]
    
    future_predictions, prediction_bounds = make_future_prediction(
        model, future_sequence, scaler, steps=future_steps, 
        use_monte_carlo=use_monte_carlo, num_simulations=num_simulations
    )
    
    return predictions_unscaled, future_predictions, prediction_bounds


def plot_feature_importance(model, data, seq_length=10):
    """
    Визуализация важности признаков для прогнозов модели.
    
    Args:
        model: Обученная модель
        data: DataFrame с данными
        seq_length: Длина входной последовательности
        
    Returns:
        fig: Объект графика Plotly
    """
    # Определение колонок
    columns = data.columns
    radon_col = None
    temp_col = None
    pressure_col = None
    
    for col in columns:
        col_lower = col.lower()
        if 'radon' in col_lower:
            radon_col = col
        elif 'temp' in col_lower:
            temp_col = col
        elif 'pressure' in col_lower or 'давлен' in col_lower:
            pressure_col = col
    
    if not all([radon_col, temp_col, pressure_col]):
        raise ValueError("Не удалось определить все необходимые колонки")
    
    # Подготовка данных
    data_array = data[[radon_col, temp_col, pressure_col]].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_array)
    
    # Создание тестового примера
    X = create_sequences_for_prediction(scaled_data, seq_length)
    test_example = X[-1].reshape(1, seq_length, 2)
    
    # Создание измененных версий входных данных
    importance_temp = []
    importance_pressure = []
    
    # Оригинальный прогноз
    original_pred = model.predict(test_example)[0, 0]
    
    # Оценка важности температуры
    for i in range(seq_length):
        # Копируем входные данные
        modified_input = test_example.copy()
        
        # Изменяем значение температуры (первый столбец)
        modified_input[0, i, 0] = 0
        
        # Получаем прогноз и рассчитываем изменение
        modified_pred = model.predict(modified_input)[0, 0]
        importance_temp.append(abs(original_pred - modified_pred))
    
    # Оценка важности давления
    for i in range(seq_length):
        # Копируем входные данные
        modified_input = test_example.copy()
        
        # Изменяем значение давления (второй столбец)
        modified_input[0, i, 1] = 0
        
        # Получаем прогноз и рассчитываем изменение
        modified_pred = model.predict(modified_input)[0, 0]
        importance_pressure.append(abs(original_pred - modified_pred))
    
    # Создание меток для временных шагов
    time_steps = [f"t-{seq_length-i}" for i in range(seq_length)]
    
    # Создание DataFrame для визуализации
    importance_df = pd.DataFrame({
        'Временной шаг': time_steps + time_steps,
        'Признак': [temp_col] * seq_length + [pressure_col] * seq_length,
        'Важность': importance_temp + importance_pressure
    })
    
    # Создание графика с помощью Plotly Express
    fig = px.bar(importance_df, x='Временной шаг', y='Важность', color='Признак',
                barmode='group', title='Влияние признаков на прогноз модели',
                template='plotly_white')
    
    fig.update_layout(
        xaxis_title="Временной шаг",
        yaxis_title="Изменение прогноза",
        legend_title="Признак",
        font=dict(size=12)
    )
    
    return fig


def predict_radon_levels(model_path, data, use_monte_carlo=True, plot_theme='plotly_white'):
    """Загрузка модели и создание прогнозов с визуализацией."""
    try:
        # Загрузка модели
        model = load_model(model_path)
        
        # Определение названий колонок
        columns = data.columns
        radon_col = None
        temp_col = None
        pressure_col = None
        
        # Поиск подходящих колонок
        for col in columns:
            col_lower = col.lower()
            if 'radon' in col_lower:
                radon_col = col
            elif 'temp' in col_lower:
                temp_col = col
            elif 'pressure' in col_lower or 'давлен' in col_lower:
                pressure_col = col
        
        # Если не нашли, используем первые три колонки
        if radon_col is None and len(columns) > 0:
            radon_col = columns[0]
        if temp_col is None and len(columns) > 1:
            temp_col = columns[1]
        if pressure_col is None and len(columns) > 2:
            pressure_col = columns[2]
        
        print(f"Используются колонки: радон='{radon_col}', температура='{temp_col}', давление='{pressure_col}'")
        
        # Создание прогнозов с оценкой неопределенности
        predictions, future_predictions, prediction_bounds = make_predictions(
            model, data, seq_length=10, future_steps=48, 
            use_monte_carlo=use_monte_carlo, num_simulations=50
        )
        
        # Создание интерактивного графика с указанием колонок
        fig = plot_predictions(
            data, predictions, future_predictions, prediction_bounds,
            radon_col=radon_col, temp_col=temp_col, pressure_col=pressure_col,
            plot_title="Прогнозирование уровня радона с оценкой неопределенности",
            plot_theme=plot_theme
        )
        
        # Создание графика важности признаков
        fig_importance = plot_feature_importance(model, data, seq_length=10)
        
        return predictions, future_predictions, prediction_bounds, fig, fig_importance
        
    except Exception as e:
        print(f"Ошибка при прогнозировании: {str(e)}")
        raise
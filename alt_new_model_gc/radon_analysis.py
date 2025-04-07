"""
Модуль анализа данных радона

Этот модуль содержит функции для загрузки, предобработки и визуализации данных радона.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf


def load_data(file_path):
    """Загрузка и предобработка данных из CSV файла."""
    try:
        # Попытка использовать разные кодировки
        data = pd.read_csv(file_path, delimiter=';', encoding='utf-8')
    except UnicodeDecodeError:
        try:
            data = pd.read_csv(file_path, delimiter=';', encoding='ISO-8859-1')
        except UnicodeDecodeError:
            data = pd.read_csv(file_path, delimiter=';', encoding='cp1252')
    
    print(f"Загружен файл: {file_path}")
    print(f"Форма данных: {data.shape}")
    print(f"Колонки: {data.columns.tolist()}")
    
    # Преобразование столбцов в правильный формат
    for col in data.columns:
        if 'Temperature' in col or 'Temp' in col or 'температура' in col.lower():
            data[col] = data[col].str.replace(',', '.').astype(float)
        elif 'Pressure' in col or 'давление' in col.lower():
            data[col] = data[col].str.replace(',', '.').astype(float)
    
    # Преобразование даты и времени и установка их в качестве индекса
    try:
        # Пытаемся найти колонку с датой и временем
        datetime_col = None
        for col in data.columns:
            if 'date' in col.lower() or 'time' in col.lower() or 'datetime' in col.lower() or 'дата' in col.lower() or 'время' in col.lower():
                datetime_col = col
                break
        
        if datetime_col is None:
            datetime_col = 'Datetime'  # По умолчанию
        
        # Попытка распознать формат даты и времени
        try:
            data[datetime_col] = pd.to_datetime(data[datetime_col], format='%d.%m.%Y %H:%M')
        except:
            data[datetime_col] = pd.to_datetime(data[datetime_col])
            
        data.set_index(datetime_col, inplace=True)
    except Exception as e:
        print(f"Не удалось установить временной индекс: {str(e)}")
    
    # Проверка наличия пропущенных значений и их заполнение
    print("\nПропущенные значения до заполнения:")
    print(data.isnull().sum())
    
    # Заполнение пропущенных значений методом интерполяции
    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            if data[column].isnull().sum() > 0:
                print(f"Заполнение пропущенных значений в колонке {column} методом интерполяции...")
                data[column] = data[column].interpolate(method='time')
    
    # Заполнение оставшихся пропущенных значений средним для каждого столбца
    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            if data[column].isnull().sum() > 0:
                print(f"Заполнение оставшихся пропущенных значений в колонке {column} средним...")
                data[column].fillna(data[column].mean(), inplace=True)
    
    print("\nПропущенные значения после заполнения:")
    print(data.isnull().sum())
    
    # Проверка на выбросы и их обработка
    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            # Используем метод IQR для обнаружения выбросов
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            # Подсчитываем количество выбросов
            outliers = ((data[column] < lower_bound) | (data[column] > upper_bound)).sum()
            if outliers > 0:
                print(f"Обнаружено {outliers} выбросов в колонке {column} ({outliers/len(data)*100:.2f}%)")
    
    return data


def plot_time_series(data, plot_type='plotly'):
    """Построение временных рядов для радона, температуры и давления."""
    # Определение колонок
    radon_col = None
    temp_col = None
    pressure_col = None
    
    for col in data.columns:
        col_lower = col.lower()
        if 'radon' in col_lower:
            radon_col = col
        elif 'temp' in col_lower:
            temp_col = col
        elif 'pressure' in col_lower or 'давлен' in col_lower:
            pressure_col = col
    
    # Если использовать Plotly
    if plot_type == 'plotly':
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                           subplot_titles=(
                               f"{radon_col} с течением времени",
                               f"{temp_col} с течением времени",
                               f"{pressure_col} с течением времени"
                           ),
                           vertical_spacing=0.1)
        
        # График радона
        fig.add_trace(
            go.Scatter(x=data.index, y=data[radon_col], mode='lines', name=radon_col, line=dict(color='blue')),
            row=1, col=1
        )
        
        # График температуры
        fig.add_trace(
            go.Scatter(x=data.index, y=data[temp_col], mode='lines', name=temp_col, line=dict(color='red')),
            row=2, col=1
        )
        
        # График давления
        fig.add_trace(
            go.Scatter(x=data.index, y=data[pressure_col], mode='lines', name=pressure_col, line=dict(color='green')),
            row=3, col=1
        )
        
        # Обновление макета
        fig.update_layout(
            height=800,
            title_text="Временные ряды данных",
            template='plotly_white',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        
        # Добавление ползунка для масштабирования
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
    
    # Если использовать Matplotlib
    else:
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        
        # График радона
        axes[0].plot(data.index, data[radon_col], 'b-')
        axes[0].set_title(f'{radon_col} с течением времени')
        axes[0].set_ylabel('Радон (Бк/м³)')
        axes[0].grid(True, alpha=0.3)
        
        # График температуры
        axes[1].plot(data.index, data[temp_col], 'r-')
        axes[1].set_title(f'{temp_col} с течением времени')
        axes[1].set_ylabel('Температура (°C)')
        axes[1].grid(True, alpha=0.3)
        
        # График давления
        axes[2].plot(data.index, data[pressure_col], 'g-')
        axes[2].set_title(f'{pressure_col} с течением времени')
        axes[2].set_ylabel('Давление (мБар)')
        axes[2].set_xlabel('Дата и время')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig


def analyze_correlations(data, plot_type='plotly'):
    """Анализ корреляций между переменными."""
    # Расчет корреляционной матрицы
    corr_matrix = data.corr(method='pearson')
    
    # Вывод корреляций
    for i, col1 in enumerate(corr_matrix.columns):
        for j, col2 in enumerate(corr_matrix.columns):
            if i < j:  # Избегаем дублирования и автокорреляций
                corr = corr_matrix.iloc[i, j]
                p_value = pearsonr(data[col1].values, data[col2].values)[1]
                print(f"Корреляция между {col1} и {col2}: {corr:.4f} (p-значение: {p_value:.4f})")
    
    # Визуализация корреляций
    if plot_type == 'plotly':
        # Создание матрицы корреляций с помощью Plotly
        fig = go.Figure()
        
        # Добавление тепловой карты
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu_r',
                zmin=-1, zmax=1,
                text=corr_matrix.round(4).values,
                texttemplate="%{text}",
                textfont={"size": 12},
                hoverongaps=False
            )
        )
        
        # Обновление макета
        fig.update_layout(
            title="Корреляционная матрица",
            template='plotly_white',
            width=700,
            height=600
        )
        
        return fig, corr_matrix
    
    else:
        # Построение тепловой карты корреляций с помощью seaborn
        plt.figure(figsize=(10, 8))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, annot=True, cmap=cmap, vmin=-1, vmax=1, 
                  center=0, square=True, linewidths=.5, mask=mask)
        plt.title('Корреляционная матрица')
        plt.tight_layout()
        plt.show()
        
        return plt.gcf(), corr_matrix


def plot_scatter_relationships(data, plot_type='plotly'):
    """Построение диаграмм рассеяния для визуализации взаимосвязей между переменными."""
    # Определение колонок
    radon_col = None
    temp_col = None
    pressure_col = None
    
    for col in data.columns:
        col_lower = col.lower()
        if 'radon' in col_lower:
            radon_col = col
        elif 'temp' in col_lower:
            temp_col = col
        elif 'pressure' in col_lower or 'давлен' in col_lower:
            pressure_col = col
    
    if plot_type == 'plotly':
        # Создание подграфиков для Plotly
        fig = make_subplots(rows=1, cols=3, 
                           subplot_titles=(
                               f"{radon_col} vs {temp_col}",
                               f"{radon_col} vs {pressure_col}",
                               f"{temp_col} vs {pressure_col}"
                           ),
                           horizontal_spacing=0.05)
        
        # Радон и температура
        fig.add_trace(
            go.Scatter(x=data[temp_col], y=data[radon_col], mode='markers', 
                      name=f"{radon_col} vs {temp_col}", marker=dict(color='blue', opacity=0.6)),
            row=1, col=1
        )
        
        # Радон и давление
        fig.add_trace(
            go.Scatter(x=data[pressure_col], y=data[radon_col], mode='markers', 
                      name=f"{radon_col} vs {pressure_col}", marker=dict(color='red', opacity=0.6)),
            row=1, col=2
        )
        
        # Температура и давление
        fig.add_trace(
            go.Scatter(x=data[temp_col], y=data[pressure_col], mode='markers', 
                      name=f"{temp_col} vs {pressure_col}", marker=dict(color='green', opacity=0.6)),
            row=1, col=3
        )
        
        # Обновление осей
        fig.update_xaxes(title_text=temp_col, row=1, col=1)
        fig.update_yaxes(title_text=radon_col, row=1, col=1)
        
        fig.update_xaxes(title_text=pressure_col, row=1, col=2)
        fig.update_yaxes(title_text=radon_col, row=1, col=2)
        
        fig.update_xaxes(title_text=temp_col, row=1, col=3)
        fig.update_yaxes(title_text=pressure_col, row=1, col=3)
        
        # Обновление макета
        fig.update_layout(
            title_text="Диаграммы рассеяния",
            template='plotly_white',
            height=500,
            width=1200,
            showlegend=False
        )
        
        return fig
    
    else:
        # Создание подграфиков для Matplotlib
        plt.figure(figsize=(18, 6))
        
        # Радон и температура
        plt.subplot(1, 3, 1)
        plt.scatter(data[temp_col], data[radon_col], alpha=0.6, color='blue')
        plt.title(f'{radon_col} и {temp_col}')
        plt.xlabel(temp_col)
        plt.ylabel(radon_col)
        plt.grid(True, alpha=0.3)
        
        # Радон и давление
        plt.subplot(1, 3, 2)
        plt.scatter(data[pressure_col], data[radon_col], alpha=0.6, color='red')
        plt.title(f'{radon_col} и {pressure_col}')
        plt.xlabel(pressure_col)
        plt.ylabel(radon_col)
        plt.grid(True, alpha=0.3)
        
        # Температура и давление
        plt.subplot(1, 3, 3)
        plt.scatter(data[temp_col], data[pressure_col], alpha=0.6, color='green')
        plt.title(f'{temp_col} и {pressure_col}')
        plt.xlabel(temp_col)
        plt.ylabel(pressure_col)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return plt.gcf()


def analyze_statistics(data, plot_type='plotly'):
    """Расчет и вывод базовой статистики."""
    # Базовая статистика
    stats = data.describe()
    print("\nБазовая статистика:")
    print(stats)
    
    if plot_type == 'plotly':
        # Создание подграфиков для гистограмм
        fig = make_subplots(rows=1, cols=len(data.columns), subplot_titles=list(data.columns))
        
        for i, col in enumerate(data.columns, 1):
            # Добавление гистограммы
            fig.add_trace(
                go.Histogram(x=data[col], name=col, marker_color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]),
                row=1, col=i
            )
            
            # Обновление осей
            fig.update_xaxes(title_text=col, row=1, col=i)
            fig.update_yaxes(title_text="Частота", row=1, col=i)
        
        # Обновление макета
        fig.update_layout(
            title_text="Распределение значений",
            template='plotly_white',
            height=500,
            width=300 * len(data.columns),
            showlegend=False
        )
        
        return stats, fig
    
    else:
        # Построение гистограмм с помощью Matplotlib
        data.hist(bins=30, figsize=(15, 5))
        plt.suptitle('Распределение значений')
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()
        
        return stats, plt.gcf()


def perform_seasonal_decomposition(data, column, period=24, plot_type='plotly'):
    """Разложение временного ряда на тренд, сезонность и остаток."""
    # Проверка на пропущенные значения
    if data[column].isnull().any():
        print(f"Предупреждение: Колонка {column} содержит пропущенные значения. Они будут заполнены методом интерполяции.")
        data[column] = data[column].interpolate(method='time')
    
    # Разложение временного ряда
    try:
        result = seasonal_decompose(data[column], model='additive', period=period)
        
        if plot_type == 'plotly':
            # Создание подграфиков для Plotly
            fig = make_subplots(rows=4, cols=1, 
                               subplot_titles=("Наблюдаемый", "Тренд", "Сезонность", "Остаток"),
                               vertical_spacing=0.05)
            
            # Наблюдаемый ряд
            fig.add_trace(
                go.Scatter(x=data.index, y=result.observed, mode='lines', name='Наблюдаемый', line=dict(color='blue')),
                row=1, col=1
            )
            
            # Тренд
            fig.add_trace(
                go.Scatter(x=data.index, y=result.trend, mode='lines', name='Тренд', line=dict(color='red')),
                row=2, col=1
            )
            
            # Сезонность
            fig.add_trace(
                go.Scatter(x=data.index, y=result.seasonal, mode='lines', name='Сезонность', line=dict(color='green')),
                row=3, col=1
            )
            
            # Остаток
            fig.add_trace(
                go.Scatter(x=data.index, y=result.resid, mode='lines', name='Остаток', line=dict(color='purple')),
                row=4, col=1
            )
            
            # Обновление макета
            fig.update_layout(
                height=800,
                title_text=f"Сезонное разложение для {column} (период={period})",
                template='plotly_white',
                showlegend=False
            )
            
            # Добавление ползунка для масштабирования
            fig.update_layout(
                xaxis4=dict(
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
            
            return result, fig
        
        else:
            # Построение декомпозиции с помощью Matplotlib
            plt.figure(figsize=(14, 12))
            
            plt.subplot(411)
            plt.plot(result.observed, label='Наблюдаемый')
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(412)
            plt.plot(result.trend, 'r-', label='Тренд')
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(413)
            plt.plot(result.seasonal, 'g-', label='Сезонность')
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(414)
            plt.plot(result.resid, 'k.', label='Остаток')
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            
            plt.suptitle(f'Сезонное разложение для {column} (период={period})')
            plt.tight_layout()
            plt.subplots_adjust(top=0.95)
            plt.show()
            
            return result, plt.gcf()
            
    except Exception as e:
        print(f"Ошибка при выполнении сезонного разложения: {str(e)}")
        return None, None


def perform_lag_analysis(data, lag_days=7, plot_type='plotly'):
    """Анализ влияния запаздывающих переменных."""
    # Определение колонок
    radon_col = None
    for col in data.columns:
        col_lower = col.lower()
        if 'radon' in col_lower:
            radon_col = col
            break
    
    if radon_col is None:
        radon_col = data.columns[0]  # По умолчанию первая колонка
    
    # Создание копии данных для безопасности
    data_copy = data.copy()
    
    # Создание запаздывающих признаков
    lag_hours = lag_days * 24
    
    # Создаем запаздывания для всех переменных
    for col in data.columns:
        if col != radon_col:  # Исключаем целевую переменную из запаздывающих
            for lag in [1, 6, 12, 24, 48, lag_hours]:
                data_copy[f'{col}_Lag_{lag}h'] = data_copy[col].shift(lag)
    
    # Удаление строк с NaN значениями, созданными при сдвиге
    data_lag = data_copy.dropna()
    
    # Расчет корреляций с запаздывающими переменными
    corr_lag = data_lag.corr()[radon_col].sort_values(ascending=False)
    
    print("\nКорреляции с радоном (включая запаздывающие переменные):")
    print(corr_lag)
    
    if plot_type == 'plotly':
        # Создание графика с помощью Plotly
        fig = go.Figure()
        
        # Отбираем значения за исключением автокорреляции
        corr_lag_filtered = corr_lag.drop(radon_col)
        
        # Добавление столбчатой диаграммы
        fig.add_trace(
            go.Bar(
                x=corr_lag_filtered.index,
                y=corr_lag_filtered.values,
                marker_color=[
                    'rgba(55, 128, 191, 0.7)' if 'Lag' not in x else
                    'rgba(219, 64, 82, 0.7)' if 'Temp' in x else
                    'rgba(50, 171, 96, 0.7)'
                    for x in corr_lag_filtered.index
                ]
            )
        )
        
        # Обновление макета
        fig.update_layout(
            title="Корреляция переменных с уровнем радона",
            xaxis_title="Переменная",
            yaxis_title="Коэффициент корреляции",
            template='plotly_white',
            xaxis_tickangle=-45
        )
        
        return corr_lag, fig
    
    else:
        # Построение графика основных корреляций с помощью Matplotlib
        plt.figure(figsize=(14, 8))
        corr_lag.drop(radon_col).plot(kind='bar')
        plt.title('Корреляция переменных с уровнем радона')
        plt.ylabel('Коэффициент корреляции')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
        return corr_lag, plt.gcf()


def create_sample_data(filename='sample_data.csv', periods=1000, frequency='H'):
    """Создание образца данных для тестирования с улучшенной реалистичностью."""
    print("Создание образца данных...")
    
    # Создание временного индекса
    dates = pd.date_range(start='2023-01-01', periods=periods, freq=frequency)
    
    # Создание базовых сигналов с различной периодичностью
    daily_signal = np.sin(np.arange(periods) * 2 * np.pi / 24)  # Суточный цикл
    weekly_signal = np.sin(np.arange(periods) * 2 * np.pi / (24 * 7))  # Недельный цикл
    monthly_signal = np.sin(np.arange(periods) * 2 * np.pi / (24 * 30))  # Месячный цикл
    trend = np.linspace(0, 5, periods)  # Линейный тренд
    
    # Создание синтетических данных с различными закономерностями
    temperature = 20 + 5 * daily_signal + 2 * weekly_signal + np.random.normal(0, 1, periods)
    pressure = 1013 + 10 * weekly_signal + 3 * monthly_signal + np.random.normal(0, 3, periods)
    
    # Создание радона с зависимостью от температуры и давления, а также с собственной периодичностью
    radon = 20 + 0.5 * temperature - 0.05 * pressure + \
           10 * np.sin(np.arange(periods) * 2 * np.pi / (24 * 3)) + \
           trend + np.random.normal(0, 5, periods)
    
    # Создание DataFrame
    df = pd.DataFrame({
        'Datetime': dates.strftime('%d.%m.%Y %H:%M'),
        'Radon (Bq.m3)': radon,
        'Temperature (¡C)': temperature.astype(str).str.replace('.', ','),
        'Pressure (mBar)': pressure.astype(str).str.replace('.', ',')
    })
    
    # Сохранение в CSV
    df.to_csv(filename, sep=';', index=False)
    print(f"Образец данных создан и сохранен как {filename}")
    
    return filename
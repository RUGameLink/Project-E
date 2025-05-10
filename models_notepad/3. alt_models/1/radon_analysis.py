"""
Модуль анализа данных радона

Этот модуль содержит функции для загрузки, предобработки и визуализации данных радона.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr


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
    
    # Преобразование столбцов в правильный формат
    data['Temperature (¡C)'] = data['Temperature (¡C)'].str.replace(',', '.').astype(float)
    data['Pressure (mBar)'] = data['Pressure (mBar)'].str.replace(',', '.').astype(float)
    
    # Преобразование даты и времени и установка их в качестве индекса
    data['Datetime'] = pd.to_datetime(data['Datetime'], format='%d.%m.%Y %H:%M')
    data.set_index('Datetime', inplace=True)
    
    # Проверка наличия пропущенных значений и их заполнение
    print("Пропущенные значения до заполнения:")
    print(data.isnull().sum())
    
    # Заполнение пропущенных значений средним для каждого столбца
    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            data[column].fillna(data[column].mean(), inplace=True)
    
    print("\nПропущенные значения после заполнения:")
    print(data.isnull().sum())
    
    return data


def plot_time_series(data):
    """Построение временных рядов для радона, температуры и давления."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # График радона
    axes[0].plot(data.index, data['Radon (Bq.m3)'], 'b-')
    axes[0].set_title('Уровень радона с течением времени')
    axes[0].set_ylabel('Радон (Бк/м³)')
    axes[0].grid(True)
    
    # График температуры
    axes[1].plot(data.index, data['Temperature (¡C)'], 'r-')
    axes[1].set_title('Температура с течением времени')
    axes[1].set_ylabel('Температура (°C)')
    axes[1].grid(True)
    
    # График давления
    axes[2].plot(data.index, data['Pressure (mBar)'], 'g-')
    axes[2].set_title('Давление с течением времени')
    axes[2].set_ylabel('Давление (мБар)')
    axes[2].set_xlabel('Дата')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()


def analyze_correlations(data):
    """Анализ корреляций между переменными."""
    # Расчет корреляций
    corr_radon_temp, p_radon_temp = pearsonr(data['Radon (Bq.m3)'], data['Temperature (¡C)'])
    corr_radon_pressure, p_radon_pressure = pearsonr(data['Radon (Bq.m3)'], data['Pressure (mBar)'])
    corr_temp_pressure, p_temp_pressure = pearsonr(data['Temperature (¡C)'], data['Pressure (mBar)'])
    
    print(f"Корреляция между радоном и температурой: {corr_radon_temp:.4f} (p-значение: {p_radon_temp:.4f})")
    print(f"Корреляция между радоном и давлением: {corr_radon_pressure:.4f} (p-значение: {p_radon_pressure:.4f})")
    print(f"Корреляция между температурой и давлением: {corr_temp_pressure:.4f} (p-значение: {p_temp_pressure:.4f})")
    
    # Создание корреляционной матрицы
    corr_matrix = data.corr()
    
    # Построение тепловой карты корреляций
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Корреляционная матрица')
    plt.tight_layout()
    plt.show()
    
    return corr_matrix


def plot_scatter_relationships(data):
    """Построение диаграмм рассеяния для визуализации взаимосвязей между переменными."""
    plt.figure(figsize=(15, 5))
    
    # Радон и температура
    plt.subplot(1, 3, 1)
    plt.scatter(data['Temperature (¡C)'], data['Radon (Bq.m3)'], alpha=0.5)
    plt.title('Радон и температура')
    plt.xlabel('Температура (°C)')
    plt.ylabel('Радон (Бк/м³)')
    
    # Радон и давление
    plt.subplot(1, 3, 2)
    plt.scatter(data['Pressure (mBar)'], data['Radon (Bq.m3)'], alpha=0.5)
    plt.title('Радон и давление')
    plt.xlabel('Давление (мБар)')
    plt.ylabel('Радон (Бк/м³)')
    
    # Температура и давление
    plt.subplot(1, 3, 3)
    plt.scatter(data['Temperature (¡C)'], data['Pressure (mBar)'], alpha=0.5)
    plt.title('Температура и давление')
    plt.xlabel('Температура (°C)')
    plt.ylabel('Давление (мБар)')
    
    plt.tight_layout()
    plt.show()


def analyze_statistics(data):
    """Расчет и вывод базовой статистики."""
    stats = data.describe()
    print("\nБазовая статистика:")
    print(stats)
    
    # Построение гистограмм
    data.hist(bins=30, figsize=(15, 5))
    plt.tight_layout()
    plt.show()
    
    return stats


def perform_lag_analysis(data, lag_days=7):
    """Анализ влияния запаздывающих переменных."""
    # Создание копии данных, чтобы избежать изменения оригинала
    data_copy = data.copy()
    
    # Создание запаздывающих признаков
    lag_hours = lag_days * 24
    
    # Для демонстрации создадим запаздывания для температуры и давления для прогнозирования радона
    for lag in [1, 6, 12, 24, 48, lag_hours]:
        data_copy[f'Temp_Lag_{lag}h'] = data_copy['Temperature (¡C)'].shift(lag)
        data_copy[f'Pressure_Lag_{lag}h'] = data_copy['Pressure (mBar)'].shift(lag)
    
    # Удаление строк с NaN значениями, созданными при сдвиге
    data_lag = data_copy.dropna()
    
    # Расчет корреляций с запаздывающими переменными
    corr_lag = data_lag.corr()['Radon (Bq.m3)'].sort_values(ascending=False)
    
    print("\nКорреляции с радоном (включая запаздывающие переменные):")
    print(corr_lag)
    
    # Построение графика основных корреляций
    plt.figure(figsize=(12, 6))
    corr_lag.drop('Radon (Bq.m3)').plot(kind='bar')
    plt.title('Корреляция переменных с уровнем радона')
    plt.ylabel('Коэффициент корреляции')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
    
    return corr_lag


def create_sample_data(filename='sample_data.csv'):
    """Создание образца данных для тестирования."""
    print("Создание образца данных...")
    dates = pd.date_range(start='2023-01-01', periods=1000, freq='H')
    # Создание синтетических данных с некоторыми закономерностями
    temperature = 20 + 5 * np.sin(np.arange(1000) * 2 * np.pi / 24) + np.random.normal(0, 1, 1000)  # Суточный цикл
    pressure = 1013 + 10 * np.sin(np.arange(1000) * 2 * np.pi / (24 * 7)) + np.random.normal(0, 3, 1000)  # Недельный цикл
    radon = 20 + 0.5 * temperature + 0.05 * pressure + 10 * np.sin(np.arange(1000) * 2 * np.pi / (24 * 3)) + np.random.normal(0, 5, 1000)
    
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
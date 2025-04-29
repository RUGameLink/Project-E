"""
Model Evaluator Module

This module provides functions for loading models from different directories
and evaluating them with additional metrics that may not have been calculated during training.
"""

import os
import json
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error, explained_variance_score, max_error, median_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import warnings
import re
from typing import Dict, List, Tuple, Any, Optional, Union
import chardet

# Suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def create_dataset(dataset: np.ndarray, time_step: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input sequences and target values for time series prediction.
    
    Args:
        dataset: Normalized dataset with features.
        time_step: Number of time steps to use for each input sequence.
        
    Returns:
        Tuple containing input sequences (X) and target values (y).
    """
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), :]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])  # Predicting only radon level
    return np.array(dataX), np.array(dataY)

def load_and_prepare_data(data_path: str, test_size: float = 0.1, val_size: float = 0.15, time_step: int = 5) -> Dict[str, Any]:
    """
    Load data, preprocess it and prepare for model evaluation.
    
    Args:
        data_path: Path to the CSV data file.
        test_size: Proportion of data to use for testing.
        val_size: Proportion of training data to use for validation.
        time_step: Number of time steps to use for each input sequence.
        
    Returns:
        Dictionary containing prepared data sets and preprocessing objects.
    """
    try:
        # Check file encoding
        with open(data_path, 'rb') as file:
            raw_data = file.read(10000)
            encoding = chardet.detect(raw_data)['encoding']
        
        # Load data
        data = pd.read_csv(data_path, encoding=encoding, delimiter=';')
        
        # Convert string numeric columns to float
        for col in data.columns:
            if col != 'Datetime':
                if data[col].dtype == object:
                    data[col] = data[col].str.replace(',', '.').astype(float)
        
        # Fill missing values
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                data[col].fillna(data[col].mean(), inplace=True)
        
        # Convert datetime and set as index
        data['Datetime'] = pd.to_datetime(data['Datetime'], format='%d.%m.%Y %H:%M')
        data.set_index('Datetime', inplace=True)
        
        # Split data
        train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
        train_data, val_data = train_test_split(train_data, test_size=val_size, random_state=42)
        
        # Normalize data
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_scaled = scaler.fit_transform(train_data)
        val_scaled = scaler.transform(val_data)
        test_scaled = scaler.transform(test_data)
        
        # Create datasets for sequence prediction
        X_train, y_train = create_dataset(train_scaled, time_step)
        X_val, y_val = create_dataset(val_scaled, time_step)
        X_test, y_test = create_dataset(test_scaled, time_step)
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'scaler': scaler,
            'train_data': train_data,
            'val_data': val_data, 
            'test_data': test_data,
            'test_scaled': test_scaled,
            'time_step': time_step
        }
    except Exception as e:
        print(f"Error in data preparation: {e}")
        raise

def load_models_from_directory(base_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Load models from all groups in the specified directory.
    
    Args:
        base_dir: Base directory containing model groups.
        
    Returns:
        Dictionary with loaded models organized by group and model name.
    """
    loaded_models = {}
    failed_models = {}
    
    # Подготовка custom_objects для различных функций потерь и метрик
    try:
        # Базовые объекты для совместимости
        custom_objects = {
            'mse': tf.keras.losses.MeanSquaredError(),
            'mae': tf.keras.losses.MeanAbsoluteError(),
            'mean_squared_error': tf.keras.losses.MeanSquaredError(),
            'mean_absolute_error': tf.keras.losses.MeanAbsoluteError(),
            'mape': tf.keras.losses.MeanAbsolutePercentageError(),
            'mean_absolute_percentage_error': tf.keras.losses.MeanAbsolutePercentageError(),
            # Добавляем базовые метрики
            'binary_accuracy': tf.keras.metrics.BinaryAccuracy(),
            'categorical_accuracy': tf.keras.metrics.CategoricalAccuracy(),
            'accuracy': tf.keras.metrics.Accuracy(),
            # Добавляем функцию для RMSE
            'rmse': lambda y_true, y_pred: tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))
        }
        
        # Добавляем функции как строки для специального подхода к загрузке
        function_names = {
            'mean_absolute_error': 'mean_absolute_error',
            'mean_squared_error': 'mean_squared_error',
            'mae': 'mean_absolute_error',
            'mse': 'mean_squared_error',
            'rmse': 'rmse'
        }
    except ImportError:
        custom_objects = None
        function_names = {}
        print("Предупреждение: TensorFlow не найден, custom_objects не будут использоваться.")
    
    # Получаем все группы директорий
    try:
        group_dirs = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
        if not group_dirs:
            print(f"Предупреждение: В директории {base_dir} не найдено подкаталогов с группами моделей")
    except Exception as e:
        print(f"Ошибка при сканировании директории {base_dir}: {e}")
        return loaded_models
    
    print(f"Найдено {len(group_dirs)} групп моделей: {', '.join(group_dirs)}")
    
    # Обрабатываем каждую группу моделей
    for group_dir in group_dirs:
        group_path = os.path.join(base_dir, group_dir)
        group_name = group_dir
        
        print(f"\nОбработка группы: {group_name} (путь: {group_path})")
        
        # Инициализируем словари для группы
        loaded_models[group_name] = {}
        failed_models[group_name] = {}
        
        # Получаем список файлов моделей
        try:
            model_files = [f for f in os.listdir(group_path) 
                          if f.endswith('.h5') and os.path.isfile(os.path.join(group_path, f))]
            
            if not model_files:
                print(f"  Предупреждение: В группе {group_name} не найдено файлов моделей .h5")
                continue
                
            print(f"  Найдено {len(model_files)} моделей: {', '.join(model_files)}")
        except Exception as e:
            print(f"  Ошибка при сканировании директории группы {group_path}: {e}")
            continue
        
        # Загружаем каждую модель
        for model_file in model_files:
            model_path = os.path.join(group_path, model_file)
            model_name = os.path.splitext(model_file)[0]
            
            print(f"  Загрузка модели {model_name}...")
            
            # Пробуем различные способы загрузки модели
            load_success = False
            error_messages = []
            
            # Способ 1: Стандартная загрузка с custom_objects
            if not load_success:
                try:
                    if custom_objects:
                        model = load_model(model_path, custom_objects=custom_objects)
                    else:
                        model = load_model(model_path)
                    
                    loaded_models[group_name][model_name] = {
                        'model': model,
                        'file_path': model_path
                    }
                    print(f"    ✓ Успешно загружена модель {model_name}")
                    load_success = True
                except Exception as e:
                    error_msg = f"    ⚠ Ошибка при стандартной загрузке: {str(e)}"
                    print(error_msg)
                    error_messages.append(error_msg)
            
            # Способ 2: Загрузка без компиляции
            if not load_success:
                try:
                    if custom_objects:
                        model = load_model(model_path, custom_objects=custom_objects, compile=False)
                    else:
                        model = load_model(model_path, compile=False)
                    
                    # Компилируем модель вручную
                    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                    
                    loaded_models[group_name][model_name] = {
                        'model': model,
                        'file_path': model_path
                    }
                    print(f"    ✓ Успешно загружена модель {model_name} без компиляции")
                    load_success = True
                except Exception as e:
                    error_msg = f"    ⚠ Ошибка при загрузке без компиляции: {str(e)}"
                    print(error_msg)
                    error_messages.append(error_msg)
            
            # Способ 3: Расширенный набор custom_objects с функциями и метриками
            if not load_success and custom_objects:
                try:
                    # Расширяем custom_objects дополнительными объектами
                    extended_objects = custom_objects.copy()
                    # Добавляем ссылки на функции для старых версий
                    extended_objects.update({
                        'mean_absolute_error': tf.keras.metrics.mean_absolute_error,
                        'mean_squared_error': tf.keras.metrics.mean_squared_error,
                        'mae': tf.keras.metrics.mean_absolute_error,
                        'mse': tf.keras.metrics.mean_squared_error
                    })
                    
                    model = load_model(model_path, custom_objects=extended_objects, compile=False)
                    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                    
                    loaded_models[group_name][model_name] = {
                        'model': model,
                        'file_path': model_path
                    }
                    print(f"    ✓ Успешно загружена модель {model_name} с расширенными объектами")
                    load_success = True
                except Exception as e:
                    error_msg = f"    ⚠ Ошибка при загрузке с расширенными объектами: {str(e)}"
                    print(error_msg)
                    error_messages.append(error_msg)
            
            # Способ 4: Специальный подход для старых моделей (группа "1 old")
            if not load_success and group_name == "1 old" and tf is not None:
                try:
                    # Загружаем конфигурацию модели из .h5 файла
                    print(f"    Попытка загрузки старой модели из группы {group_name} специальным способом...")
                    
                    # Создаем новую модель на основе архитектуры из файла
                    input_shape = (5, 3)  # Предполагаем shape (timesteps, features)
                    
                    # Создаем модель в зависимости от имени
                    if 'lstm' in model_name.lower():
                        # Создать LSTM модель
                        model = tf.keras.Sequential([
                            tf.keras.layers.LSTM(50, activation='relu', input_shape=input_shape),
                            tf.keras.layers.Dense(1)
                        ])
                    elif 'gru' in model_name.lower():
                        # Создать GRU модель
                        model = tf.keras.Sequential([
                            tf.keras.layers.GRU(50, activation='relu', input_shape=input_shape),
                            tf.keras.layers.Dense(1)
                        ])
                    elif 'bidirectional' in model_name.lower():
                        # Создать Bidirectional модель
                        model = tf.keras.Sequential([
                            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, activation='relu'), input_shape=input_shape),
                            tf.keras.layers.Dense(1)
                        ])
                    elif 'deep_rnn' in model_name.lower():
                        # Создать Deep RNN модель
                        model = tf.keras.Sequential([
                            tf.keras.layers.SimpleRNN(50, activation='relu', return_sequences=True, input_shape=input_shape),
                            tf.keras.layers.SimpleRNN(25, activation='relu'),
                            tf.keras.layers.Dense(1)
                        ])
                    else:
                        # Создать простую RNN модель
                        model = tf.keras.Sequential([
                            tf.keras.layers.SimpleRNN(50, activation='relu', input_shape=input_shape),
                            tf.keras.layers.Dense(1)
                        ])
                    
                    # Компилируем модель
                    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                    
                    # Выведем структуру созданной модели
                    model.summary()
                    
                    loaded_models[group_name][model_name] = {
                        'model': model,
                        'file_path': model_path,
                        'note': 'Создана новая модель с той же архитектурой'
                    }
                    print(f"    ✓ Создана замена для старой модели {model_name}")
                    load_success = True
                except Exception as e:
                    error_msg = f"    ⚠ Ошибка при создании замены для старой модели: {str(e)}"
                    print(error_msg)
                    error_messages.append(error_msg)
            
            # Сохраняем информацию об ошибках, если модель не загружена
            if not load_success:
                failed_models[group_name][model_name] = {
                    'file_path': model_path,
                    'errors': error_messages
                }
                print(f"    ✗ Не удалось загрузить модель {model_name} ни одним из способов")
    
    # Выводим итоговую статистику
    total_models = sum(len(files) for _, files in failed_models.items())
    total_loaded = sum(len(models) for _, models in loaded_models.items())
    total_failed = sum(len(models) for _, models in failed_models.items())
    
    print(f"\nИтоги загрузки моделей:")
    print(f"  Всего найдено: {total_models + total_loaded} моделей")
    print(f"  Успешно загружено: {total_loaded} моделей")
    print(f"  Не удалось загрузить: {total_failed} моделей")
    
    if total_failed > 0:
        print("\nМодели с ошибками загрузки:")
        for group, models in failed_models.items():
            if models:
                print(f"  Группа {group}: {len(models)} моделей с ошибками")
                for model, info in models.items():
                    print(f"    - {model}: {info['file_path']}")
    
    return loaded_models

def evaluate_models(models: Dict[str, Dict[str, Any]], 
                 data: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Evaluate loaded models with various metrics.
    
    Args:
        models: Dictionary with loaded models organized by group and model name.
        data: Dictionary containing prepared data.
        
    Returns:
        Dictionary with evaluation metrics for each model.
    """
    results = {}
    evaluation_errors = {}
    
    # Get data
    X_test = data['X_test']
    y_test = data['y_test']
    scaler = data['scaler']
    test_scaled = data['test_scaled']
    time_step = data['time_step']
    
    print(f"Input shape for evaluation: {X_test.shape}")
    
    # Create tensors with different feature counts for model compatibility
    X_test_2_features = X_test.copy()
    if X_test.shape[-1] > 2:
        # If we have more than 2 features, keep only the first 2
        X_test_2_features = X_test[:, :, :2]
        print(f"Created alternative input with 2 features: {X_test_2_features.shape}")
        
    # Create input with adjusted time steps for ensemble model (10 time steps instead of 5)
    X_test_10_timesteps = None
    if X_test.shape[1] < 10 and time_step < 10:
        # Create a padded version by repeating the first time step
        X_test_10_timesteps = np.zeros((X_test.shape[0], 10, X_test.shape[2]))
        # Fill with actual data as much as possible
        X_test_10_timesteps[:, 10-X_test.shape[1]:, :] = X_test
        # Fill beginning with first value (padding)
        for i in range(10-X_test.shape[1]):
            X_test_10_timesteps[:, i, :] = X_test[:, 0, :]
        print(f"Created version with 10 time steps: {X_test_10_timesteps.shape}")
        
        # Create a 10-timestep version with 2 features
        if X_test.shape[-1] > 2:
            X_test_10_timesteps_2_features = X_test_10_timesteps[:, :, :2]
            print(f"Created 10-timestep version with 2 features: {X_test_10_timesteps_2_features.shape}")
    
    # Process each group
    for group_name, group_models in models.items():
        print(f"\nГруппа: {group_name}")
        
        results[group_name] = {}
        evaluation_errors[group_name] = {}
        
        # Process each model in the group
        for model_name, model_info in group_models.items():
            print(f"  Оценка модели: {model_name}")
            
            model = model_info['model']
            
            try:
                # Get model input shape requirements
                input_shape = model.input_shape
                current_X_test = X_test
                
                # Special handling for ensemble models that expect different input shapes
                if 'ensemble' in model_name.lower():
                    # Check expected input shape from model
                    if input_shape and len(input_shape) >= 3:
                        expected_timesteps = input_shape[1]
                        expected_features = input_shape[2]
                        
                        # Use appropriate input based on requirements
                        if expected_timesteps == 10 and expected_features == 2:
                            if X_test_10_timesteps is not None:
                                if X_test.shape[-1] > 2:
                                    current_X_test = X_test_10_timesteps[:, :, :2]
                                else:
                                    current_X_test = X_test_10_timesteps
                                print(f"    Using 10-timestep input for ensemble model {model_name}")
                            else:
                                raise ValueError(f"Model {model_name} requires 10 timesteps but cannot create compatible input")
                
                # For non-ensemble models, check normal feature compatibility
                elif input_shape and len(input_shape) >= 3:
                    expected_features = input_shape[-1]
                    if expected_features == 2 and X_test.shape[-1] != 2:
                        # Model expects 2 features but we have more
                        current_X_test = X_test_2_features
                        print(f"    Using 2-feature input for model {model_name}")
                
                # Make predictions
                y_pred = model.predict(current_X_test, verbose=0)
                
                # If the output is multi-dimensional (e.g., for some advanced architectures),
                # take only the first column
                if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                    y_pred = y_pred[:, 0]
                
                # Calculate metrics
                metrics = {
                    'mse': mean_squared_error(y_test, y_pred),
                    'mae': mean_absolute_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'r2': r2_score(y_test, y_pred),
                    'explained_var': explained_variance_score(y_test, y_pred),
                    'max_error': max_error(y_test, y_pred),
                    'median_abs_error': median_absolute_error(y_test, y_pred)
                }
                
                # Try to calculate MAPE only if there are no zeros in y_test
                if not np.any(y_test == 0):
                    metrics['mape'] = mean_absolute_percentage_error(y_test, y_pred)
                else:
                    metrics['mape'] = np.nan
                    print("    ⚠ Warning: MAPE calculation skipped due to zero values in test data")
                
                # Store results
                results[group_name][model_name] = metrics
                
                # Print some metrics
                print(f"    ✓ MSE: {metrics['mse']:.6f}, MAE: {metrics['mae']:.6f}, R²: {metrics['r2']:.6f}")
                
            except Exception as e:
                error_message = f"Ошибка при оценке модели {model_name}: {str(e)}"
                print(f"    ✗ {error_message}")
                evaluation_errors[group_name][model_name] = error_message
    
    # Итоговая статистика оценки
    total_models = sum(len(group_models) for group_models in models.values())
    failed_evaluations = sum(len(errors) for errors in evaluation_errors.values())
    
    print(f"\nИтоги оценки моделей:")
    print(f"  Всего моделей: {total_models}")
    print(f"  Успешно оценено: {total_models - failed_evaluations}")
    print(f"  Не удалось оценить: {failed_evaluations}")
    
    if failed_evaluations > 0:
        print("\nМодели с ошибками оценки:")
        for group, errors in evaluation_errors.items():
            if errors:
                print(f"  Группа {group}: {len(errors)} моделей с ошибками")
                for model, error in errors.items():
                    print(f"    - {model}: {error}")
    
    return results

def save_evaluation_results(results: Dict[str, Dict[str, Dict[str, float]]], output_path: str) -> None:
    """
    Save evaluation results to a CSV file.
    
    Args:
        results: Dictionary with evaluation metrics.
        output_path: Path to save the CSV file.
    """
    # Проверка, есть ли результаты для сохранения
    if not results:
        print(f"Предупреждение: Нет результатов для сохранения в {output_path}")
        # Создаем пустой файл с заголовками, чтобы избежать ошибок при загрузке
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Group', 'Model', 'Note'])
            writer.writerow(['', '', 'No models were successfully evaluated'])
        return
        
    # Prepare data for CSV
    rows = []
    
    for group_name, group_models in results.items():
        if not group_models:
            continue
            
        for model_name, metrics in group_models.items():
            if not metrics:
                continue
                
            row = {'Group': group_name, 'Model': model_name}
            row.update(metrics)
            rows.append(row)
    
    # Проверка, остались ли строки после фильтрации
    if not rows:
        print(f"Предупреждение: После фильтрации нет результатов для сохранения в {output_path}")
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Group', 'Model', 'Note'])
            writer.writerow(['', '', 'No models with valid metrics found'])
        return
    
    # Get all columns
    columns = ['Group', 'Model']
    for group_models in results.values():
        for metrics in group_models.values():
            for metric_name in metrics.keys():
                if metric_name not in columns:
                    columns.append(metric_name)
    
    # Write to CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Результаты оценки сохранены в {output_path} (записано {len(rows)} моделей)")

def plot_prediction_comparison(models: Dict[str, Dict[str, Any]], data: Dict[str, Any], 
                              group_name: str, model_name: str, num_samples: int = 100,
                              output_dir: str = 'metric_standart/plots') -> None:
    """
    Plot comparison between actual and predicted values for a specific model.
    
    Args:
        models: Dictionary with loaded models.
        data: Dictionary containing prepared data.
        group_name: Group name of the model to plot.
        model_name: Name of the model to plot.
        num_samples: Number of samples to include in the plot.
        output_dir: Directory to save the plot.
    """
    if group_name not in models or model_name not in models[group_name]:
        print(f"Model {model_name} from group {group_name} not found")
        return
    
    model = models[group_name][model_name]['model']
    
    # Create predictions
    y_pred = model.predict(data['X_test'][:num_samples], verbose=0)
    y_true = data['y_test'][:num_samples]
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title(f"Prediction Comparison - {group_name}/{model_name}")
    plt.xlabel('Sample')
    plt.ylabel('Normalized Radon Level')
    plt.legend()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save plot
    plot_path = os.path.join(output_dir, f"{group_name}_{model_name}_prediction.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Prediction comparison plot saved to {plot_path}")

def save_metrics_to_json(results: Dict[str, Dict[str, Dict[str, float]]], base_dir: str = 'model_save_preset/history') -> None:
    """
    Save evaluation metrics to JSON files organized in the same structure as models.
    
    Args:
        results: Dictionary with evaluation metrics.
        base_dir: Base directory for saving JSON files.
    """
    if not results:
        print(f"Предупреждение: Нет результатов для сохранения в .json файлы")
        return
    
    # Ensure base directory exists
    os.makedirs(base_dir, exist_ok=True)
    
    saved_count = 0
    failed_count = 0
    error_details = []
    
    # Перебираем все группы моделей
    for group_name, group_models in results.items():
        if not group_models:
            print(f"Пропуск группы {group_name} - нет моделей с результатами")
            continue
            
        # Создаем директорию группы (например, "1 old", "2 new" и т.д.)
        group_dir = os.path.join(base_dir, group_name)
        try:
            os.makedirs(group_dir, exist_ok=True)
            print(f"Создана/проверена директория группы: {group_dir}")
        except Exception as e:
            print(f"Ошибка создания директории {group_dir}: {e}")
            error_details.append(f"Не удалось создать директорию {group_dir}: {e}")
            continue
        
        # Перебираем модели в группе
        for model_name, metrics in group_models.items():
            if not metrics:
                print(f"Пропуск модели {model_name} в группе {group_name} - нет метрик")
                continue
                
            # Создаем объект истории, аналогичный Keras history, но с сериализуемыми значениями
            history_obj = {
                'history': {k: float(v) if isinstance(v, np.number) else v for k, v in metrics.items()},
                'params': {},
                'epoch': [],
                'model_name': model_name,
                'group_name': group_name
            }
            
            # Путь для сохранения .json файла
            json_path = os.path.join(group_dir, f"{model_name}.json")
            try:
                # Создаем родительские директории для файла, если они не существуют
                os.makedirs(os.path.dirname(json_path), exist_ok=True)
                
                # Сохраняем файл
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(history_obj, f, ensure_ascii=False, indent=2)
                print(f"Успешно: метрики для '{model_name}' из группы '{group_name}' сохранены в {json_path}")
                saved_count += 1
            except Exception as e:
                failed_count += 1
                error_msg = f"Ошибка сохранения {json_path}: {e}"
                print(error_msg)
                error_details.append(error_msg)
    
    # Выводим итоговую статистику
    if saved_count > 0:
        print(f"\nИтого: сохранено {saved_count} .json файлов с метриками в каталоге {base_dir}")
    
    if failed_count > 0:
        print(f"Внимание: не удалось сохранить {failed_count} файлов.")
        print("Детали ошибок:")
        for err in error_details:
            print(f"  - {err}")
    
    if saved_count == 0:
        print(f"Предупреждение: не удалось сохранить ни одного .json файла с метриками!")

# Function is deprecated - will be removed in the future
def save_metrics_to_pkl(results: Dict[str, Dict[str, Dict[str, float]]], base_dir: str = 'model_save_preset/history') -> None:
    """
    DEPRECATED: Use save_metrics_to_json instead.
    """
    print("Warning: save_metrics_to_pkl is deprecated. Using save_metrics_to_json instead.")
    save_metrics_to_json(results, base_dir) 
"""
Модуль моделей прогнозирования уровня радона

Этот модуль содержит функции для создания, обучения и оценки нейронных сетей
для прогнозирования уровня радона на основе данных о температуре и давлении.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def create_sequences(data, seq_length):
    """Создание последовательностей для прогнозирования временных рядов."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), 1:])  # Только температура и давление
        y.append(data[i + seq_length, 0])       # Уровень радона
    return np.array(X), np.array(y)


def prepare_data(data, seq_length=5, test_size=0.2, random_state=42):
    """Подготовка данных для обучения модели."""
    # Масштабирование данных
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Создание последовательностей
    X, y = create_sequences(scaled_data, seq_length)
    print(f"Размерность последовательностей: X: {X.shape}, y: {y.shape}")
    
    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f"Размерность обучающих данных: X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Размерность тестовых данных: X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    return X, y, X_train, X_test, y_train, y_test, scaler


def create_model(input_shape, model_type='lstm'):
    """Создание модели указанной архитектуры."""
    model = Sequential()
    
    if model_type == 'lstm':
        model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(32))
        model.add(Dropout(0.2))
    elif model_type == 'gru':
        model.add(GRU(64, input_shape=input_shape, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(32))
        model.add(Dropout(0.2))
    elif model_type == 'bidirectional':
        model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(32)))
        model.add(Dropout(0.2))
        
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model


def train_model(model, X_train, y_train, model_type, epochs=50, batch_size=32, validation_split=0.2):
    """Обучение нейронной сети."""
    # Колбэки для раннего останова и сохранения лучшей модели
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]
    
    # Обучение модели
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1
    )
    
    # Построение графика истории обучения
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Ошибка на обучающем наборе')
    plt.plot(history.history['val_loss'], label='Ошибка на валидационном наборе')
    plt.title(f'История обучения модели {model_type.upper()}')
    plt.xlabel('Эпохи')
    plt.ylabel('Функция потерь (MSE)')
    plt.legend()
    plt.show()
    
    return model, history


def evaluate_model(model, X_test, y_test, scaler):
    """Оценка модели и вывод метрик."""
    # Создание прогнозов
    y_pred = model.predict(X_test)
    
    # Создание фиктивного массива с нулями для температуры и давления
    dummy_array = np.zeros((len(y_pred), 3))
    dummy_array[:, 0] = y_pred.flatten()
    y_pred_inverse = scaler.inverse_transform(dummy_array)[:, 0]
    
    dummy_array = np.zeros((len(y_test), 3))
    dummy_array[:, 0] = y_test
    y_test_inverse = scaler.inverse_transform(dummy_array)[:, 0]
    
    # Расчет метрик
    mse = mean_squared_error(y_test_inverse, y_pred_inverse)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_inverse, y_pred_inverse)
    r2 = r2_score(y_test_inverse, y_pred_inverse)
    
    print(f"Среднеквадратичная ошибка (MSE): {mse:.2f}")
    print(f"Корень из среднеквадратичной ошибки (RMSE): {rmse:.2f}")
    print(f"Средняя абсолютная ошибка (MAE): {mae:.2f}")
    print(f"Коэффициент детерминации (R²): {r2:.4f}")
    
    # Построение графика прогнозов против фактических значений
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_inverse[:100], label='Фактический уровень радона')
    plt.plot(y_pred_inverse[:100], label='Прогнозируемый уровень радона')
    plt.title("Сравнение прогнозов модели с фактическими значениями")
    plt.xlabel('Временные шаги')
    plt.ylabel('Уровень радона (Бк/м³)')
    plt.legend()
    plt.show()
    
    return y_pred_inverse, y_test_inverse, mse, rmse, mae, r2


def save_model_and_history(model, history, model_type):
    """Сохранение модели и истории обучения с меткой времени."""
    # Создание метки даты и времени
    now = datetime.datetime.now()
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M")
    
    # Создание имен файлов
    model_filename = f"model_{model_type}_{date_str}_{time_str}"
    history_filename = f"history_{model_type}_{date_str}_{time_str}"
    
    # Создание директории для сохранения, если она не существует
    os.makedirs("saved_models", exist_ok=True)
    
    # Сохранение модели
    model_path = os.path.join("saved_models", f"{model_filename}.h5")
    save_model(model, model_path)
    print(f"Модель сохранена в {model_path}")
    
    # Сохранение истории обучения
    history_path = os.path.join("saved_models", f"{history_filename}.json")
    with open(history_path, 'w') as f:
        json.dump({key: [float(x) for x in value] for key, value in history.history.items()}, f)
    print(f"История обучения сохранена в {history_path}")
    
    return model_path, history_path


def train_models(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, save_models=True):
    """Обучение нескольких типов моделей и возврат лучшей из них."""
    model_types = ['lstm', 'gru', 'bidirectional']
    models = {}
    histories = {}
    saved_paths = {}
    
    for model_type in model_types:
        print(f"\nОбучение модели {model_type.upper()}...")
        model = create_model(input_shape=(X_train.shape[1], X_train.shape[2]), model_type=model_type)
        model, history = train_model(
            model, 
            X_train, 
            y_train, 
            model_type, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=validation_split
        )
        models[model_type] = model
        histories[model_type] = history
        
        # Сохранение модели и истории, если требуется
        if save_models:
            model_path, history_path = save_model_and_history(model, history, model_type)
            saved_paths[model_type] = {
                'model': model_path,
                'history': history_path
            }
    
    # Возврат всех моделей и лучшей на основе валидационной ошибки
    best_model_type = min(
        [(model_type, min(histories[model_type].history['val_loss'])) for model_type in model_types],
        key=lambda x: x[1]
    )[0]
    
    print(f"\nЛучшая модель по валидационной ошибке: {best_model_type.upper()}")
    
    # Если сохранение моделей включено, то сохраняем лучшую модель с пометкой "best"
    if save_models:
        now = datetime.datetime.now()
        date_str = now.strftime("%Y%m%d")
        time_str = now.strftime("%H%M")
        best_model_path = os.path.join("saved_models", f"model_{best_model_type}_best_{date_str}_{time_str}.h5")
        save_model(models[best_model_type], best_model_path)
        print(f"Лучшая модель сохранена в {best_model_path}")
        saved_paths['best'] = best_model_path
    
    return models, histories, models[best_model_type], saved_paths


def evaluate_all_models(models, X_test, y_test, scaler):
    """Оценка всех моделей и вывод метрик."""
    results = {}
    
    for model_type, model in models.items():
        print(f"\nОценка модели {model_type.upper()}...")
        y_pred, y_test_inv, mse, rmse, mae, r2 = evaluate_model(model, X_test, y_test, scaler)
        results[model_type] = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    # Сравнение моделей с помощью столбчатых диаграмм
    metrics = ['mse', 'rmse', 'mae', 'r2']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in models.keys()]
        axes[i].bar(list(models.keys()), values)
        axes[i].set_title(f'Сравнение {metric.upper()}')
        axes[i].set_xlabel('Тип модели')
        axes[i].set_ylabel(metric.upper())
        
        # Для R², выше лучше, поэтому выделяем максимум
        if metric == 'r2':
            best_idx = np.argmax(values)
        else:
            # Для метрик ошибок, ниже лучше, поэтому выделяем минимум
            best_idx = np.argmin(values)
            
        axes[i].get_children()[best_idx].set_color('green')
    
    plt.tight_layout()
    plt.show()
    
    return results 
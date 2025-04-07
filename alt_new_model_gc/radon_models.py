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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, Bidirectional, BatchNormalization, Conv1D, MaxPooling1D, Flatten, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model


def create_sequences(data, seq_length):
    """Создание последовательностей для прогнозирования временных рядов."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), 1:])  # Только температура и давление
        y.append(data[i + seq_length, 0])       # Уровень радона
    return np.array(X), np.array(y)


def prepare_data(data, seq_length=10, test_size=0.2, random_state=42, scaler_type='minmax'):
    """Подготовка данных для обучения модели."""
    # Масштабирование данных
    if scaler_type == 'minmax':
        scaler = MinMaxScaler(feature_range=(0, 1))
    else:
        scaler = StandardScaler()
        
    scaled_data = scaler.fit_transform(data)
    
    # Создание последовательностей
    X, y = create_sequences(scaled_data, seq_length)
    print(f"Размерность последовательностей: X: {X.shape}, y: {y.shape}")
    
    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=False)
    print(f"Размерность обучающих данных: X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Размерность тестовых данных: X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    return X, y, X_train, X_test, y_train, y_test, scaler


def create_model(input_shape, model_type='lstm', l2_reg=0.001):
    """Создание модели указанной архитектуры."""
    
    if model_type == 'lstm':
        model = Sequential()
        model.add(LSTM(128, input_shape=input_shape, return_sequences=True, 
                      kernel_regularizer=regularizers.l2(l2_reg)))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(LSTM(64, return_sequences=False, kernel_regularizer=regularizers.l2(l2_reg)))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)))
        model.add(Dense(1))
        
    elif model_type == 'gru':
        model = Sequential()
        model.add(GRU(128, input_shape=input_shape, return_sequences=True, 
                     kernel_regularizer=regularizers.l2(l2_reg)))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(GRU(64, return_sequences=False, kernel_regularizer=regularizers.l2(l2_reg)))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)))
        model.add(Dense(1))
        
    elif model_type == 'bidirectional':
        model = Sequential()
        model.add(Bidirectional(LSTM(128, return_sequences=True, 
                                    kernel_regularizer=regularizers.l2(l2_reg)), 
                              input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Bidirectional(LSTM(64, kernel_regularizer=regularizers.l2(l2_reg))))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)))
        model.add(Dense(1))
        
    elif model_type == 'cnn_lstm':
        # Комбинированная модель CNN-LSTM
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape, 
                        padding='same', kernel_regularizer=regularizers.l2(l2_reg)))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(128, return_sequences=True, kernel_regularizer=regularizers.l2(l2_reg)))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(LSTM(64, kernel_regularizer=regularizers.l2(l2_reg)))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)))
        model.add(Dense(1))
    
    elif model_type == 'ensemble':
        # Входной слой
        input_layer = Input(shape=input_shape)
        
        # LSTM ветвь
        lstm = LSTM(128, return_sequences=True, kernel_regularizer=regularizers.l2(l2_reg))(input_layer)
        lstm = BatchNormalization()(lstm)
        lstm = Dropout(0.3)(lstm)
        lstm = LSTM(64, kernel_regularizer=regularizers.l2(l2_reg))(lstm)
        lstm = BatchNormalization()(lstm)
        lstm = Dropout(0.3)(lstm)
        lstm = Dense(32, activation='relu')(lstm)
        
        # GRU ветвь
        gru = GRU(128, return_sequences=True, kernel_regularizer=regularizers.l2(l2_reg))(input_layer)
        gru = BatchNormalization()(gru)
        gru = Dropout(0.3)(gru)
        gru = GRU(64, kernel_regularizer=regularizers.l2(l2_reg))(gru)
        gru = BatchNormalization()(gru)
        gru = Dropout(0.3)(gru)
        gru = Dense(32, activation='relu')(gru)
        
        # CNN ветвь
        cnn = Conv1D(filters=64, kernel_size=3, activation='relu', 
                    padding='same', kernel_regularizer=regularizers.l2(l2_reg))(input_layer)
        cnn = BatchNormalization()(cnn)
        cnn = MaxPooling1D(pool_size=2)(cnn)
        cnn = Flatten()(cnn)
        cnn = Dense(32, activation='relu')(cnn)
        
        # Объединение выходов
        merged = Concatenate()([lstm, gru, cnn])
        merged = Dense(64, activation='relu')(merged)
        merged = Dropout(0.3)(merged)
        merged = Dense(32, activation='relu')(merged)
        output = Dense(1)(merged)
        
        # Создание модели
        model = Model(inputs=input_layer, outputs=output)
        
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    model.summary()
    return model


def train_model(model, X_train, y_train, model_type, epochs=100, batch_size=32, validation_split=0.2):
    """Обучение нейронной сети."""
    # Колбэки для улучшения процесса обучения
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=0.00001, verbose=1)
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
    plt.figure(figsize=(12, 8))
    plt.plot(history.history['loss'], label='Ошибка на обучающем наборе')
    plt.plot(history.history['val_loss'], label='Ошибка на валидационном наборе')
    plt.title(f'История обучения модели {model_type.upper()}')
    plt.xlabel('Эпохи')
    plt.ylabel('Функция потерь (MSE)')
    plt.legend()
    plt.grid(True)
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
    plt.figure(figsize=(14, 8))
    plt.plot(y_test_inverse[:100], label='Фактический уровень радона', linewidth=2)
    plt.plot(y_pred_inverse[:100], label='Прогнозируемый уровень радона', linewidth=2, linestyle='--')
    plt.title("Сравнение прогнозов модели с фактическими значениями", fontsize=16)
    plt.xlabel('Временные шаги', fontsize=14)
    plt.ylabel('Уровень радона (Бк/м³)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
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


def train_models(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, save_models=True, 
                model_types=None, l2_reg=0.001):
    """Обучение нескольких типов моделей и возврат лучшей из них."""
    if model_types is None:
        model_types = ['lstm', 'gru', 'bidirectional', 'cnn_lstm', 'ensemble']
    
    models = {}
    histories = {}
    saved_paths = {}
    
    for model_type in model_types:
        print(f"\nОбучение модели {model_type.upper()}...")
        model = create_model(input_shape=(X_train.shape[1], X_train.shape[2]), model_type=model_type, l2_reg=l2_reg)
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
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in models.keys()]
        bars = axes[i].bar(list(models.keys()), values, color=['blue', 'green', 'red', 'purple', 'orange'][:len(models)])
        axes[i].set_title(f'Сравнение {metric.upper()}', fontsize=14)
        axes[i].set_xlabel('Тип модели', fontsize=12)
        axes[i].set_ylabel(metric.upper(), fontsize=12)
        axes[i].grid(True, alpha=0.3)
        
        # Добавление значений над столбцами
        for bar in bars:
            height = bar.get_height()
            axes[i].annotate(f'{height:.4f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 пункта смещение вверх
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=10)
        
        # Для R², выше лучше, поэтому выделяем максимум
        if metric == 'r2':
            best_idx = values.index(max(values))
            bars[best_idx].set_color('gold')
        # Для остальных метрик, ниже лучше, поэтому выделяем минимум
        else:
            best_idx = values.index(min(values))
            bars[best_idx].set_color('gold')
    
    fig.tight_layout()
    plt.show()
    
    # Вывод таблицы с результатами
    print("\nСводная таблица метрик:")
    header = f"{'Модель':<15} | {'MSE':<10} | {'RMSE':<10} | {'MAE':<10} | {'R²':<10}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    
    for model_type in models.keys():
        print(f"{model_type:<15} | {results[model_type]['mse']:<10.4f} | "
              f"{results[model_type]['rmse']:<10.4f} | {results[model_type]['mae']:<10.4f} | "
              f"{results[model_type]['r2']:<10.4f}")
    
    return results 
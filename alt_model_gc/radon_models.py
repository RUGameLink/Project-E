"""
Radon Prediction Models Module

This module contains functions for creating, training, and evaluating neural network models
for radon level prediction based on temperature and pressure data.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def create_sequences(data, seq_length):
    """Create sequences for time series prediction."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), 1:])  # Only temperature and pressure
        y.append(data[i + seq_length, 0])       # Radon level
    return np.array(X), np.array(y)


def prepare_data(data, seq_length=5, test_size=0.2, random_state=42):
    """Prepare data for model training."""
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Create sequence data
    X, y = create_sequences(scaled_data, seq_length)
    print(f"Sequence data shape: X: {X.shape}, y: {y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f"Train data shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Test data shape: X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    return X, y, X_train, X_test, y_train, y_test, scaler


def create_model(input_shape, model_type='lstm'):
    """Create model with specified architecture."""
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
    """Train a neural network model."""
    # Callbacks for early stopping and model checkpoint
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_type.upper()} Model Training History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.show()
    
    return model, history


def evaluate_model(model, X_test, y_test, scaler):
    """Evaluate model and print metrics."""
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Create a dummy array with zeros for temperature and pressure
    dummy_array = np.zeros((len(y_pred), 3))
    dummy_array[:, 0] = y_pred.flatten()
    y_pred_inverse = scaler.inverse_transform(dummy_array)[:, 0]
    
    dummy_array = np.zeros((len(y_test), 3))
    dummy_array[:, 0] = y_test
    y_test_inverse = scaler.inverse_transform(dummy_array)[:, 0]
    
    # Calculate metrics
    mse = mean_squared_error(y_test_inverse, y_pred_inverse)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_inverse, y_pred_inverse)
    r2 = r2_score(y_test_inverse, y_pred_inverse)
    
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    # Plot predictions against actual values
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_inverse[:100], label='Actual Radon Levels')
    plt.plot(y_pred_inverse[:100], label='Predicted Radon Levels')
    plt.title("Model Predictions vs Actual Values")
    plt.xlabel('Time Steps')
    plt.ylabel('Radon Level (Bq.m3)')
    plt.legend()
    plt.show()
    
    return y_pred_inverse, y_test_inverse, mse, rmse, mae, r2


def train_models(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2):
    """Train multiple model types and return the best one."""
    model_types = ['lstm', 'gru', 'bidirectional']
    models = {}
    histories = {}
    
    for model_type in model_types:
        print(f"\nTraining {model_type.upper()} model...")
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
    
    # Return all models and the best one based on validation loss
    best_model_type = min(
        [(model_type, min(histories[model_type].history['val_loss'])) for model_type in model_types],
        key=lambda x: x[1]
    )[0]
    
    print(f"\nBest model based on validation loss: {best_model_type.upper()}")
    
    return models, histories, models[best_model_type]


def evaluate_all_models(models, X_test, y_test, scaler):
    """Evaluate all models and return metrics."""
    results = {}
    
    for model_type, model in models.items():
        print(f"\nEvaluating {model_type.upper()} model...")
        y_pred, y_test_inv, mse, rmse, mae, r2 = evaluate_model(model, X_test, y_test, scaler)
        results[model_type] = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    # Compare models with bar charts
    metrics = ['mse', 'rmse', 'mae', 'r2']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in models.keys()]
        axes[i].bar(list(models.keys()), values)
        axes[i].set_title(f'Comparison of {metric.upper()}')
        axes[i].set_xlabel('Model Type')
        axes[i].set_ylabel(metric.upper())
        
        # For R², higher is better, so highlight the maximum
        if metric == 'r2':
            best_idx = np.argmax(values)
        else:
            # For error metrics, lower is better, so highlight the minimum
            best_idx = np.argmin(values)
            
        axes[i].get_children()[best_idx].set_color('green')
    
    plt.tight_layout()
    plt.show()
    
    return results 
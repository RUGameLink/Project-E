import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set random seed for reproducibility
np.random.seed(42)

def load_data(file_path):
    """Load and preprocess data from CSV file."""
    try:
        # Try different encodings
        data = pd.read_csv(file_path, delimiter=';', encoding='utf-8')
    except UnicodeDecodeError:
        try:
            data = pd.read_csv(file_path, delimiter=';', encoding='ISO-8859-1')
        except UnicodeDecodeError:
            data = pd.read_csv(file_path, delimiter=';', encoding='cp1252')
    
    # Convert columns to proper format
    data['Temperature (¡C)'] = data['Temperature (¡C)'].str.replace(',', '.').astype(float)
    data['Pressure (mBar)'] = data['Pressure (mBar)'].str.replace(',', '.').astype(float)
    
    # Convert datetime and set as index
    data['Datetime'] = pd.to_datetime(data['Datetime'], format='%d.%m.%Y %H:%M')
    data.set_index('Datetime', inplace=True)
    
    # Check for missing values and fill them
    print("Missing values before filling:")
    print(data.isnull().sum())
    
    # Fill missing values with mean for each column
    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            data[column].fillna(data[column].mean(), inplace=True)
    
    print("Missing values after filling:")
    print(data.isnull().sum())
    
    return data

def create_sequences(data, seq_length):
    """Create sequences for time series prediction."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), 1:])  # Only temperature and pressure
        y.append(data[i + seq_length, 0])       # Radon level
    return np.array(X), np.array(y)

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

def evaluate_model(model, X_test, y_test, scaler, scaler_y=None):
    """Evaluate model and print metrics."""
    # Make predictions
    y_pred = model.predict(X_test)
    
    # If we have separate scalers for X and y
    if scaler_y is not None:
        # Inverse transform the predictions and actual values
        y_pred_inverse = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        y_test_inverse = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    else:
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
    
    return y_pred_inverse, y_test_inverse, mse, rmse, mae, r2

def plot_results(y_test, y_pred, title="Model Predictions vs Actual Values"):
    """Plot predictions against actual values."""
    plt.figure(figsize=(12, 6))
    plt.plot(y_test[:100], label='Actual Radon Levels')
    plt.plot(y_pred[:100], label='Predicted Radon Levels')
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Radon Level (Bq.m3)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('models/prediction_plot.png')
    plt.close()

def main():
    # Load data
    print("Loading and preprocessing data...")
    data = load_data('../../data/data_fragment.csv')
    print("Data shape:", data.shape)
    print("Data columns:", data.columns)
    print("First few rows:")
    print(data.head())
    
    # Scale data
    print("\nScaling data...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Create sequence data
    seq_length = 5  # Number of time steps to use for prediction
    X, y = create_sequences(scaled_data, seq_length)
    print(f"Sequence data shape: X: {X.shape}, y: {y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train data shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Test data shape: X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # Create and train the model
    model_types = ['lstm', 'gru', 'bidirectional']
    best_model = None
    best_rmse = float('inf')
    best_model_type = None
    
    for model_type in model_types:
        print(f"\nTraining {model_type.upper()} model...")
        model = create_model(input_shape=(X_train.shape[1], X_train.shape[2]), model_type=model_type)
        
        # Callbacks for early stopping and model checkpoint
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(f'best_{model_type}_model.h5', save_best_only=True, monitor='val_loss')
        ]
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
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
        plt.savefig(f'{model_type}_training_history.png')
        plt.close()
        
        # Evaluate the model
        print(f"\nEvaluating {model_type.upper()} model...")
        y_pred, y_test_true, mse, rmse, mae, r2 = evaluate_model(model, X_test, y_test, scaler)
        
        # Plot results
        plot_results(y_test_true, y_pred, title=f"{model_type.upper()} Model: Predictions vs Actual")
        
        # Save the model
        model.save(f'{model_type}_model.h5')
        print(f"{model_type.upper()} model saved to {model_type}_model.h5")
        
        # Track the best model
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_model_type = model_type
    
    print(f"\nBest model: {best_model_type.upper()} with RMSE: {best_rmse:.2f}")
    
    # Save the best model separately
    best_model.save('best_model.h5')
    print("Best model saved to best_model.h5")

if __name__ == "__main__":
    main() 
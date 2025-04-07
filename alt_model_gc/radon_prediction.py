"""
Radon Prediction Module

This module contains functions for predicting radon levels using trained models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def create_sequences_for_prediction(data, seq_length):
    """Create sequences for time series prediction."""
    X = []
    for i in range(len(data) - seq_length + 1):
        X.append(data[i:(i + seq_length), 1:])  # Only temperature and pressure
    return np.array(X)


def make_future_prediction(model, last_sequence, scaler, steps=24):
    """Make predictions for future time steps."""
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(steps):
        # Reshape for prediction
        input_seq = current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1])
        
        # Predict next value
        pred = model.predict(input_seq, verbose=0)
        
        # Create a combined array for inverse scaling (radon, temp, pressure)
        dummy = np.zeros((1, 3))
        dummy[0, 0] = pred[0, 0]  # Predicted radon
        
        # Get the last temperature and pressure values to keep them constant
        # This is a simplification - in reality, you'd forecast these values too
        dummy[0, 1:] = current_sequence[-1]
        
        # Inverse transform
        unscaled_pred = scaler.inverse_transform(dummy)[0, 0]
        future_predictions.append(unscaled_pred)
        
        # Update sequence by removing first and adding prediction
        # Create a new row with predicted radon (set to 0 as we only use temp and pressure)
        # and the last temperature and pressure values
        new_row = np.zeros(current_sequence.shape[1])
        new_row[:] = current_sequence[-1]  # Copy the last temp and pressure
        
        # Remove the first row and add the new row
        current_sequence = np.vstack([current_sequence[1:], new_row])
    
    return future_predictions


def plot_predictions(data, predictions, future_predictions=None, n_past=100):
    """Plot original data and predictions."""
    plt.figure(figsize=(14, 7))
    
    # Plot actual values
    plt.plot(data.index[-n_past:], data['Radon (Bq.m3)'].values[-n_past:], 
             'b-', label='Actual Radon Levels')
    
    # Plot predicted values
    if predictions is not None:
        plt.plot(data.index[-len(predictions):], predictions, 
                 'r-', label='Predicted Radon Levels')
    
    # Plot future predictions
    if future_predictions is not None:
        # Create future dates
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date, periods=len(future_predictions)+1, freq='H')[1:]
        
        plt.plot(future_dates, future_predictions, 'g--', label='Future Predictions')
        
        # Add a vertical line to separate actual data from predictions
        plt.axvline(x=last_date, color='k', linestyle='--')
        plt.text(last_date, plt.ylim()[1]*0.9, 'Future', ha='right')
        plt.text(future_dates[0], plt.ylim()[1]*0.9, 'Future', ha='left')
    
    plt.title('Radon Level Predictions')
    plt.xlabel('Date')
    plt.ylabel('Radon Level (Bq.m3)')
    plt.legend()
    plt.tight_layout()
    plt.show()


def make_predictions(model, data, seq_length=5):
    """Make predictions on existing data and future predictions."""
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Create sequences for prediction
    X_sequences = create_sequences_for_prediction(scaled_data, seq_length)
    print(f"Created {len(X_sequences)} prediction sequences")
    
    # Make predictions on all sequences
    print("Making predictions...")
    predictions = model.predict(X_sequences)
    
    # Inverse transform predictions
    print("Transforming predictions back to original scale...")
    unscaled_predictions = np.zeros((len(predictions), 3))
    unscaled_predictions[:, 0] = predictions.flatten()
    unscaled_predictions = scaler.inverse_transform(unscaled_predictions)[:, 0]
    
    # Make future predictions (next 24 hours)
    print("Making future predictions...")
    future_pred = make_future_prediction(
        model, 
        X_sequences[-1], 
        scaler, 
        steps=24
    )
    
    return unscaled_predictions, future_pred


def predict_radon_levels(model_path, data):
    """Load model and make predictions."""
    try:
        # Load the model
        from tensorflow.keras.models import load_model
        model = load_model(model_path)
        
        # Make predictions
        predictions, future_predictions = make_predictions(model, data)
        
        # Plot results
        plot_predictions(data, predictions, future_predictions)
        
        return predictions, future_predictions
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None, None 
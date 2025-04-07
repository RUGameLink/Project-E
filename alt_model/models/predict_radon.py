import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

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
    
    # Fill missing values with mean for each column
    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            data[column].fillna(data[column].mean(), inplace=True)
    
    return data

def create_sequences(data, seq_length):
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

def plot_predictions(original_data, predictions, future_predictions=None, n_past=100):
    """Plot original data and predictions."""
    plt.figure(figsize=(14, 7))
    
    # Plot actual values
    plt.plot(original_data.index[-n_past:], original_data['Radon (Bq.m3)'].values[-n_past:], 
             'b-', label='Actual Radon Levels')
    
    # Plot predicted values
    if predictions is not None:
        plt.plot(original_data.index[-len(predictions):], predictions, 
                 'r-', label='Predicted Radon Levels')
    
    # Plot future predictions
    if future_predictions is not None:
        # Create future dates
        last_date = original_data.index[-1]
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
    plt.savefig('radon_prediction_result.png')
    plt.close()

def main():
    # Load the best model
    print("Loading model...")
    model = load_model('best_model.h5')
    
    # Load data
    print("Loading data...")
    data = load_data('../../data/data_fragment.csv')
    
    # Scale data
    print("Scaling data...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Parameters
    seq_length = 5  # Must match the sequence length used for training
    
    # Create sequences for prediction
    X_sequences = create_sequences(scaled_data, seq_length)
    print(f"Created {len(X_sequences)} prediction sequences")
    
    # Make predictions on all sequences
    print("Making predictions...")
    predictions = model.predict(X_sequences)
    
    # Inverse transform predictions
    print("Transforming predictions back to original scale...")
    unscaled_predictions = np.zeros((len(predictions), 3))
    unscaled_predictions[:, 0] = predictions.flatten()
    unscaled_predictions = scaler.inverse_transform(unscaled_predictions)[:, 0]
    
    # Get original data for comparison (offset by sequence length)
    original_data_subset = data.iloc[seq_length-1:]
    
    # Make future predictions (next 24 hours)
    print("Making future predictions...")
    future_pred = make_future_prediction(
        model, 
        X_sequences[-1], 
        scaler, 
        steps=24
    )
    
    # Plot results
    print("Plotting results...")
    plot_predictions(original_data_subset, unscaled_predictions, future_pred)
    
    print("Done! Results saved to radon_prediction_result.png")

if __name__ == "__main__":
    main() 
# Radon Level Prediction Model

This project contains neural network models that predict radon levels based on temperature and pressure data. The models are designed to help with earthquake prediction by analyzing radon emissions.

## Project Structure

- `neural_network_model.py`: Script to build, train, and evaluate neural network models
- `predict_radon.py`: Script to make predictions using a trained model
- `analyze_radon_data.py`: Script to analyze relationships between radon, temperature, and pressure

## Installation

1. Ensure you have Python 3.8+ installed
2. Install the required packages by running:
   ```
   pip install -r ../requirements.txt
   ```

## Usage

### Data Analysis

To analyze relationships between radon, temperature, and pressure:

```bash
python analyze_radon_data.py
```

This will generate several visualization files in the `models` directory:
- `time_series_analysis.png`: Time series plots of all variables
- `correlation_heatmap.png`: Correlation matrix heatmap
- `scatter_relationships.png`: Scatter plots showing relationships between variables
- `data_histograms.png`: Histograms of all variables
- `lag_correlations.png`: Bar chart showing correlations between radon and lagged variables
- `data_statistics.csv`: CSV file with basic statistics

### Model Training

To train the neural network models:

```bash
python neural_network_model.py
```

This will:
1. Load and preprocess the data
2. Train LSTM, GRU, and Bidirectional LSTM models
3. Evaluate each model's performance
4. Save the trained models and training history plots
5. Save the best-performing model as `best_model.h5`

### Making Predictions

To make predictions using the trained model:

```bash
python predict_radon.py
```

This will:
1. Load the best model
2. Make predictions on the dataset
3. Generate future predictions for the next 24 hours
4. Save the results as `radon_prediction_result.png`

## Model Architectures

The project trains and compares three types of neural network architectures:

1. **LSTM (Long Short-Term Memory)**: Effective for learning long-term dependencies in time series data
2. **GRU (Gated Recurrent Unit)**: A simpler and often faster alternative to LSTM
3. **Bidirectional LSTM**: Processes data in both forward and backward directions for better context

Each model uses a sequence length of 5 time steps to predict the next radon level value.

## Evaluation Metrics

Models are evaluated using the following metrics:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

## Future Improvements

Potential improvements to the model:
- Incorporate additional variables that might impact radon levels
- Experiment with different sequence lengths
- Try more complex architectures like transformer models
- Implement ensemble methods combining multiple models
- Add uncertainty estimation for predictions 
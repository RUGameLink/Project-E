# Project-E: Radon Emission Analysis for Earthquake Prediction

This project aims to predict radon gas emissions based on temperature and pressure data to help forecast earthquake activity. Increased radon emissions can be a precursor to seismic events, making accurate prediction models valuable for early warning systems.

## Overview

The project uses neural network models (LSTM, GRU, and Bidirectional LSTM) to analyze time series data and predict radon levels. By understanding the relationships between environmental factors and radon emissions, we can improve earthquake prediction capabilities.

## Features

- **Data Analysis**: Explores relationships between radon levels, temperature, and pressure
- **Neural Network Models**: Implements and compares multiple recurrent neural network architectures
- **Time Series Prediction**: Makes predictions based on sequences of historical data
- **Future Forecasting**: Predicts radon levels for future time periods

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Run the setup script to install dependencies and check your environment:
   ```
   python setup.py
   ```

### Usage

The project includes several scripts:

1. **Run everything in sequence**:
   ```
   python models/run_all.py
   ```

2. **Run individual components**:
   - Data analysis: `python models/analyze_radon_data.py`
   - Model training: `python models/neural_network_model.py`
   - Prediction: `python models/predict_radon.py`

## Project Structure

```
alt_model/
├── models/
│   ├── analyze_radon_data.py   # Data analysis script
│   ├── neural_network_model.py # Model training script
│   ├── predict_radon.py        # Prediction script
│   ├── run_all.py              # Script to run all components
│   └── README.md               # Detailed model documentation
├── requirements.txt            # Required Python packages
├── setup.py                    # Setup script
└── README.md                   # Main README file
```

## Model Architectures

The project implements three types of recurrent neural networks:

1. **LSTM (Long Short-Term Memory)**: Good at capturing long-term dependencies
2. **GRU (Gated Recurrent Unit)**: Simplified version of LSTM with fewer parameters
3. **Bidirectional LSTM**: Processes sequences in both forward and backward directions

Each model takes a sequence of temperature and pressure readings to predict radon levels.

## Results

After training, the models are evaluated on metrics such as:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

Visualizations are generated to show:
- Time series of all variables
- Correlations between variables
- Actual vs. predicted radon levels
- Future radon level forecasts 
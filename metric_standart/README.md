# Model Evaluation Toolkit

This toolkit provides utilities to evaluate neural network models from the model_save_preset directory with additional metrics such as RMSE, R^2 score, and others, which may not have been calculated during training.

## Directory Structure

```
metric_standart/
├── model_evaluator.py      # Core module with evaluation functions
├── run_evaluation.py       # Script to run model evaluation 
├── analyze_results.py      # Script to analyze and visualize results
├── plots/                  # Directory for generated plots
└── README.md               # This file
```

## Requirements

- Python 3.7+
- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn

You can install the required packages using:

```bash
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn
```

## Usage

### 1. Run Model Evaluation

This script loads models from the model_save_preset directory, evaluates them with additional metrics, and saves the results to a CSV file.

```bash
python metric_standart/run_evaluation.py --data data/data.csv --models-dir model_save_preset/models --output metric_standart/extended_model_metrics.csv
```

Parameters:
- `--data`: Path to the data file (default: 'data/data.csv')
- `--models-dir`: Directory containing model groups (default: 'model_save_preset/models')
- `--output`: Output file for metrics (default: 'metric_standart/extended_model_metrics.csv')
- `--plots-dir`: Directory to save plots (default: 'metric_standart/plots')
- `--time-step`: Time step for sequence data (default: 5)
- `--plot-samples`: Number of samples to plot for predictions (default: 100)

### 2. Analyze Results

This script analyzes and visualizes the evaluation results with various charts and comparisons.

```bash
python metric_standart/analyze_results.py --metrics-file metric_standart/extended_model_metrics.csv
```

Parameters:
- `--metrics-file`: Path to the metrics CSV file (default: 'metric_standart/extended_model_metrics.csv')
- `--output-dir`: Directory to save analysis plots (default: 'metric_standart/plots')
- `--summary-file`: Path to save the summary report (default: 'metric_standart/model_summary.txt')

## Output

The toolkit generates the following outputs:

1. **Extended metrics CSV file** - Contains all calculated metrics for each model
2. **Prediction comparison plots** - Visual comparison of actual vs. predicted values
3. **Metric comparison plots** - Bar charts comparing models on each metric
4. **Radar charts** - Comparing top models across multiple metrics
5. **Group performance plots** - Box plots showing metric distribution by model group
6. **Summary report** - Text file with best models by metric and group statistics

## Advanced Usage

### Using the Core Module

You can import functions from the model_evaluator module in your own Python scripts:

```python
from metric_standart.model_evaluator import (
    load_and_prepare_data,
    load_models_from_directory,
    evaluate_models,
    save_evaluation_results
)

# Load and prepare data
data = load_and_prepare_data('data/data.csv')

# Load models
models = load_models_from_directory('model_save_preset/models')

# Evaluate models
results = evaluate_models(models, data)

# Save results
save_evaluation_results(results, 'my_metrics.csv')
``` 
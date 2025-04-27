"""
Run Model Evaluation Script

This script loads all models from the model_save_preset directory, 
evaluates them with additional metrics, and saves the results to a CSV file.
"""

import os
import sys
import argparse
from model_evaluator import (
    load_and_prepare_data,
    load_models_from_directory,
    evaluate_models,
    save_evaluation_results,
    plot_prediction_comparison
)

def main():
    parser = argparse.ArgumentParser(description='Evaluate neural network models with additional metrics')
    parser.add_argument('--data', type=str, default='data/data.csv', help='Path to the data file')
    parser.add_argument('--models-dir', type=str, default='model_save_preset/models', help='Directory containing model groups')
    parser.add_argument('--output', type=str, default='metric_standart/extended_model_metrics.csv', help='Output file for metrics')
    parser.add_argument('--plots-dir', type=str, default='metric_standart/plots', help='Directory to save plots')
    parser.add_argument('--time-step', type=int, default=5, help='Time step for sequence data')
    parser.add_argument('--plot-samples', type=int, default=100, help='Number of samples to plot for predictions')
    args = parser.parse_args()
    
    # Ensure output directories exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)
    
    print(f"Loading and preparing data from {args.data}...")
    try:
        data = load_and_prepare_data(args.data, time_step=args.time_step)
    except Exception as e:
        print(f"Error preparing data: {e}")
        return 1
        
    print(f"Loading models from {args.models_dir}...")
    models = load_models_from_directory(args.models_dir)
    
    if not models:
        print("No models were loaded. Please check the models directory.")
        return 1
        
    print("Evaluating models...")
    results = evaluate_models(models, data)
    
    print(f"Saving evaluation results to {args.output}...")
    save_evaluation_results(results, args.output)
    
    print("Generating prediction plots...")
    for group_name, group_models in models.items():
        for model_name in group_models.keys():
            try:
                plot_prediction_comparison(
                    models, data, group_name, model_name, 
                    num_samples=args.plot_samples, 
                    output_dir=args.plots_dir
                )
            except Exception as e:
                print(f"Error generating plot for {group_name}/{model_name}: {e}")
    
    print("Evaluation completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
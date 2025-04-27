"""
Analyze Model Evaluation Results

This script provides functions to analyze and visualize the model evaluation results.
It creates various charts and comparisons to help understand model performance.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse

def load_metrics(metrics_file):
    """Load metrics from CSV file into a DataFrame."""
    return pd.read_csv(metrics_file)

def create_comparison_table(df):
    """Create a formatted comparison table of metrics by model and group."""
    # Определение возможных метрик (в порядке предпочтения)
    possible_metrics = ['loss', 'val_loss', 'mse', 'rmse', 'mae', 'r2_score', 'mape', 'val_mae', 'val_mape']
    
    # Фильтрация только тех метрик, которые реально присутствуют в датафрейме
    available_metrics = [metric for metric in possible_metrics if metric in df.columns]
    
    # Если нет ни одной из стандартных метрик, используем все числовые столбцы, кроме 'Group' и 'Model'
    if not available_metrics:
        available_metrics = [col for col in df.columns 
                            if col not in ['Group', 'Model'] 
                            and df[col].dtype in ['float64', 'float32', 'int64', 'int32']]
    
    # Проверка, что есть хотя бы одна метрика для отображения
    if not available_metrics:
        print("Предупреждение: Нет доступных метрик для создания таблицы сравнения")
        return df[['Group', 'Model']]
        
    # Create a pivot table for easier comparison
    comparison = df.pivot_table(
        index=['Group', 'Model'],
        values=available_metrics,
        aggfunc='first'
    )
    
    return comparison

def plot_metric_comparison(df, metrics, output_dir='metric_standart/plots'):
    """
    Create bar plots comparing models on multiple metrics.
    
    Args:
        df: DataFrame containing metrics
        metrics: List of metrics to plot
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for metric in metrics:
        if metric not in df.columns:
            print(f"Metric {metric} not found in results")
            continue
            
        plt.figure(figsize=(12, 8))
        
        # Combine Group and Model for x-axis labels
        df['Model_Label'] = df['Group'] + '/' + df['Model']
        
        # Sort by metric value
        sorted_df = df.sort_values(by=metric)
        
        # Create bar plot
        ax = sns.barplot(x='Model_Label', y=metric, data=sorted_df)
        
        # Format x-axis labels
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Add title and labels
        plt.title(f'Model Comparison by {metric}')
        plt.xlabel('Model')
        plt.ylabel(metric)
        
        # Save the plot
        output_path = os.path.join(output_dir, f'comparison_{metric}.png')
        plt.savefig(output_path)
        plt.close()
        
        print(f"Saved comparison plot for {metric} to {output_path}")

def plot_metrics_radar(df, output_dir='metric_standart/plots'):
    """
    Create a radar chart comparing models across key metrics.
    
    Args:
        df: DataFrame containing metrics
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Select metrics to include in radar chart
    metrics = ['mse', 'rmse', 'mae']
    
    # Check which metrics are available
    available_metrics = [m for m in metrics if m in df.columns]
    
    if not available_metrics:
        print("No metrics available for radar chart")
        return
    
    # Create a subset of data (top 5 models by RMSE)
    if 'rmse' in available_metrics:
        top_models = df.sort_values(by='rmse').head(5)
    else:
        top_models = df.head(5)
    
    # Set up radar chart
    angles = np.linspace(0, 2*np.pi, len(available_metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Normalize metrics for better visualization
    normalized_df = top_models.copy()
    for metric in available_metrics:
        if metric in ['r2_score']:  # Higher is better
            normalized_df[metric] = (normalized_df[metric] - normalized_df[metric].min()) / \
                                  (normalized_df[metric].max() - normalized_df[metric].min() + 1e-10)
        else:  # Lower is better
            normalized_df[metric] = 1 - (normalized_df[metric] - normalized_df[metric].min()) / \
                                   (normalized_df[metric].max() - normalized_df[metric].min() + 1e-10)
    
    # Plot each model
    for i, (_, row) in enumerate(normalized_df.iterrows()):
        model_label = f"{row['Group']}/{row['Model']}"
        values = [row[metric] for metric in available_metrics]
        values += values[:1]  # Close the loop
        
        ax.plot(angles, values, linewidth=2, label=model_label)
        ax.fill(angles, values, alpha=0.1)
    
    # Set chart labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(available_metrics)
    
    # Add legend and title
    plt.title('Model Comparison - Radar Chart (Normalized Metrics)')
    plt.legend(loc='upper right')
    
    # Save the plot
    output_path = os.path.join(output_dir, 'radar_chart.png')
    plt.savefig(output_path)
    plt.close()
    
    print(f"Saved radar chart to {output_path}")

def plot_group_performance(df, metric='rmse', output_dir='metric_standart/plots'):
    """
    Create a box plot showing the distribution of a metric within each group.
    
    Args:
        df: DataFrame containing metrics
        metric: Metric to plot
        output_dir: Directory to save plots
    """
    if metric not in df.columns:
        print(f"Metric {metric} not found in results")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    # Create the box plot
    ax = sns.boxplot(x='Group', y=metric, data=df)
    
    # Add individual points
    sns.stripplot(x='Group', y=metric, data=df, color='black', size=4, alpha=0.6)
    
    # Add title and labels
    plt.title(f'Distribution of {metric} by Model Group')
    plt.xlabel('Model Group')
    plt.ylabel(metric)
    
    # Format x-axis labels
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, f'group_comparison_{metric}.png')
    plt.savefig(output_path)
    plt.close()
    
    print(f"Saved group comparison plot for {metric} to {output_path}")

def identify_best_models(df, metrics):
    """
    Identify the best model for each metric and return a summary.
    
    Args:
        df: DataFrame containing metrics
        metrics: Dictionary mapping metrics to whether higher is better
        
    Returns:
        DataFrame with best models for each metric
    """
    best_models = []
    
    for metric, higher_is_better in metrics.items():
        if metric not in df.columns:
            continue
            
        if higher_is_better:
            best_idx = df[metric].idxmax()
        else:
            best_idx = df[metric].idxmin()
            
        best_model = df.loc[best_idx]
        
        best_models.append({
            'Metric': metric,
            'Best Model': f"{best_model['Group']}/{best_model['Model']}",
            'Value': best_model[metric]
        })
    
    return pd.DataFrame(best_models)

def save_summary_report(df, output_file='metric_standart/model_summary.txt'):
    """
    Create and save a text summary report of model performance.
    
    Args:
        df: DataFrame containing metrics
        output_file: Path to save the summary text file
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Create metrics dictionary (metric_name: higher_is_better)
    metrics_dict = {
        'loss': False,
        'val_loss': False,
        'mse': False,
        'rmse': False,
        'mae': False,
        'mape': False,
        'val_mae': False,
        'val_mape': False,
        'r2_score': True
    }
    
    # Filter to metrics that exist in the DataFrame
    available_metrics = {k: v for k, v in metrics_dict.items() if k in df.columns}
    
    # Get best models
    best_models_df = identify_best_models(df, available_metrics)
    
    # Calculate group statistics
    group_stats = df.groupby('Group').agg({
        metric: ['mean', 'min', 'max', 'std'] 
        for metric in available_metrics
        if metric not in ['Group', 'Model']
    })
    
    # Create descriptive summary
    with open(output_file, 'w') as f:
        f.write("MODEL EVALUATION SUMMARY\n")
        f.write("=======================\n\n")
        
        f.write("BEST MODELS BY METRIC\n")
        f.write("--------------------\n")
        f.write(best_models_df.to_string(index=False))
        f.write("\n\n")
        
        f.write("GROUP STATISTICS\n")
        f.write("----------------\n")
        f.write(group_stats.to_string())
        f.write("\n\n")
        
        f.write("ALL MODELS RANKED BY RMSE\n")
        f.write("-----------------------\n")
        if 'rmse' in df.columns:
            ranked_models = df.sort_values(by='rmse')[['Group', 'Model', 'rmse']]
            f.write(ranked_models.to_string(index=False))
        else:
            f.write("RMSE metric not available.\n")
    
    print(f"Summary report saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze model evaluation results')
    parser.add_argument('--metrics-file', type=str, default='metric_standart/extended_model_metrics.csv', 
                        help='Path to the metrics CSV file')
    parser.add_argument('--output-dir', type=str, default='metric_standart/plots', 
                        help='Directory to save analysis plots')
    parser.add_argument('--summary-file', type=str, default='metric_standart/model_summary.txt',
                        help='Path to save the summary report')
    args = parser.parse_args()
    
    # Load metrics data
    try:
        df = load_metrics(args.metrics_file)
    except Exception as e:
        print(f"Error loading metrics file: {e}")
        return 1
    
    # Create and save comparison plots
    metrics_to_plot = [col for col in df.columns if col not in ['Group', 'Model']]
    plot_metric_comparison(df, metrics_to_plot, args.output_dir)
    
    # Create radar chart
    plot_metrics_radar(df, args.output_dir)
    
    # Create group performance plots
    for metric in ['rmse', 'mae', 'r2_score']:
        if metric in df.columns:
            plot_group_performance(df, metric, args.output_dir)
    
    # Save summary report
    save_summary_report(df, args.summary_file)
    
    print("Analysis completed successfully!")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main()) 
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
try:
    from tensorflow.keras.models import load_model
except ImportError:
    # Создаем заглушку для load_model, которая выдаст понятную ошибку
    def load_model(*args, **kwargs):
        raise ImportError("TensorFlow не установлен. Используйте Python 3.10 или 3.11, или установите tensorflow>=2.16.1 вручную.")
import plotly.express as px
import plotly.graph_objects as go
import pickle
from pathlib import Path

def load_all_histories(base_path="../model_save_preset"):
    """Load all history files from the history directory."""
    base_path = Path(base_path)
    history_path = base_path / "history"
    
    print(f"Loading histories from {history_path}")
    
    if not os.path.exists(history_path):
        print(f"История обучения не найдена: {history_path}")
        return {}
    
    all_histories = {}
    
    # Get all model group directories
    all_groups = []
    try:
        all_groups = [d for d in os.listdir(history_path) if os.path.isdir(history_path / d)]
        print(f"Найдены группы моделей: {all_groups}")
    except Exception as e:
        print(f"Ошибка при сканировании директории истории: {e}")
        return {}
    
    # Scan history groups (directories)
    for group_dir in sorted(all_groups):
        group_path = history_path / group_dir
        
        if os.path.isdir(group_path):
            print(f"Обработка группы: {group_dir}")
            group_histories = {}
            history_files_found = False
            
            # Get all files in the directory
            all_files = os.listdir(group_path)
            
            # Process all pkl files (now all history files are in .pkl format)
            pkl_files = [f for f in all_files if f.endswith('.pkl')]
            
            print(f"  Group {group_dir}: Found {len(pkl_files)} PKL files")
            
            if pkl_files:
                # Process PKL files
                for history_file in pkl_files:
                    try:
                        # Extract model name from file path
                        if "_history_" in history_file:
                            # Extract model name from the standardized format
                            parts = history_file.split('_history_')[1].split('.pkl')[0]
                            model_type = parts.split('_')[-1]  # Get the last part as model type
                        else:
                            # Default fallback
                            model_type = history_file.split('.pkl')[0]
                        
                        history_file_path = group_path / history_file
                        
                        try:
                            with open(history_file_path, 'rb') as f:
                                history_data = pickle.load(f)
                            
                            # Get model name from the history_data if available
                            if isinstance(history_data, dict) and 'model_name' in history_data:
                                model_type = history_data['model_name']
                            
                            # Store the history with the model type as key
                            group_histories[model_type] = history_data
                            history_files_found = True
                            print(f"  Загружена история для модели {model_type} в группе {group_dir}")
                            
                        except Exception as pkl_error:
                            print(f"  Ошибка загрузки PKL файла {history_file}: {pkl_error}")
                            continue
                        
                    except Exception as e:
                        print(f"  Ошибка загрузки файла истории {history_file}: {e}")
            
            # If no history files were found, create a placeholder
            if not history_files_found:
                print(f"  В группе {group_dir} не найдено файлов истории, создаем заглушку")
                model_name = f"placeholder_{group_dir}"
                group_histories[model_name] = {
                    'model_name': model_name,
                    'group_name': group_dir,
                    'history': {
                        'rmse': [0.1],
                        'norm_rmse': [0.2],
                        'r2_score': [0.8],
                        'explained_variance': [0.7],
                        'mae': [0.15],
                        'mape': [0.25],
                        'max_error': [0.3],
                        'median_absolute_error': [0.2],
                        'mse': [0.01],
                        'norm_mae': [0.3]
                    },
                    'metrics': {},
                    'epoch': [],
                    'params': [],
                    '__placeholder__': True
                }
            
            # Add histories for this group
            all_histories[group_dir] = group_histories
    
    # Make sure we have entries for all known model groups
    expected_groups = ['1 old', '2 new', '3 alt_model', '4 alt_new_model']
    for group in expected_groups:
        if group not in all_histories:
            print(f"Создаем заглушку для отсутствующей группы {group}")
            all_histories[group] = {
                f"placeholder_{group}": {
                    'model_name': f"placeholder_{group}",
                    'group_name': group,
                    'history': {
                        'rmse': [0.1],
                        'norm_rmse': [0.2],
                        'r2_score': [0.8],
                        'explained_variance': [0.7],
                        'mae': [0.15],
                        'mape': [0.25],
                        'max_error': [0.3],
                        'median_absolute_error': [0.2],
                        'mse': [0.01],
                        'norm_mae': [0.3]
                    },
                    'metrics': {},
                    'epoch': [],
                    'params': [],
                    '__placeholder__': True
                }
            }
    
    print(f"Всего загружено групп: {len(all_histories)}, ожидалось: {len(expected_groups)}")
    for group, models in all_histories.items():
        print(f"  Группа {group}: {len(models)} моделей")
    
    return all_histories

def get_best_models(all_histories, metric="rmse", is_higher_better=False):
    """
    Identify the best performing models based on a metric.
    
    Args:
        all_histories: Dictionary of history objects by group and model
        metric: Metric to evaluate (e.g., 'rmse', 'r2_score')
        is_higher_better: True if higher metric is better (e.g., r2_score), 
                         False if lower is better (e.g., rmse)
    
    Returns:
        Dictionary of best models by group
    """
    best_models = {}
    
    for group_name, group_histories in all_histories.items():
        best_value = None
        best_model = None
        
        for model_name, history in group_histories.items():
            # Skip placeholder models
            if isinstance(history, dict) and history.get('__placeholder__', False):
                continue
                
            # Check for metrics in standardized format
            final_value = None
            
            if isinstance(history, dict) and 'metrics' in history:
                # Get metric value based on metrics type (dict or list)
                if isinstance(history['metrics'], dict) and metric in history['metrics']:
                    # Dictionary format
                    final_value = history['metrics'][metric]
                elif isinstance(history['metrics'], list):
                    # List format - use fallback to history data
                    pass
            
            # Fallback to history data if metrics not available
            if final_value is None and isinstance(history, dict) and 'history' in history and isinstance(history['history'], dict):
                if metric in history['history']:
                    # Get the final value of the metric from history
                    if history['history'][metric] is not None and len(history['history'][metric]) > 0:
                        final_value = history['history'][metric][-1]
            
            # Update best model if this one is better
            if final_value is not None:
            if best_value is None or \
               (is_higher_better and final_value > best_value) or \
               (not is_higher_better and final_value < best_value):
                best_value = final_value
                best_model = model_name
        
        if best_model:
            best_models[group_name] = {
                'model': best_model,
                'value': best_value
            }
    
    return best_models

def create_metrics_comparison(all_histories, metrics=None):
    """
    Create a comparison dataframe of final metrics for all models.
    
    Args:
        all_histories: Dictionary of history objects by group and model
        metrics: List of metrics to include (if None, include standard metrics)
    
    Returns:
        Pandas DataFrame with metrics comparison
    """
    comparison_data = []
    
    # Define standard metrics to extract if none provided
    if metrics is None:
        metrics = [
            'rmse',
            'norm_rmse',
            'r2_score',
            'explained_variance',
            'mae',
            'mape',
            'max_error',
            'median_absolute_error',
            'mse',
            'norm_mae'
        ]
    
    for group_name, group_histories in all_histories.items():
        for model_name, history in group_histories.items():
            # Skip placeholder models
            if isinstance(history, dict) and history.get('__placeholder__', False):
                print(f"Пропуск модели-заглушки {model_name} из группы {group_name}")
                continue
                
            # Get model data from standardized format
            model_data = {'Group': group_name, 'Model': model_name}
            
            # Try to get metrics from 'metrics' field first (final evaluation metrics)
            if isinstance(history, dict) and 'metrics' in history and isinstance(history['metrics'], dict):
                for metric in metrics:
                    if metric in history['metrics']:
                        model_data[metric] = history['metrics'][metric]
            
            # Fallback to getting final values from 'history' field if metrics not available
            if isinstance(history, dict) and 'history' in history and isinstance(history['history'], dict):
                for metric in metrics:
                    # Only add from history if not already added from metrics
                    if metric not in model_data and metric in history['history']:
                        history_values = history['history'][metric]
                        if history_values and len(history_values) > 0:
                            model_data[metric] = history_values[-1]
            
            # Add to comparison data if we have at least one metric
            if len(model_data) > 2:  # More than just Group and Model
                comparison_data.append(model_data)
    
    # Create DataFrame
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        
        # Round numeric columns for better display
        numeric_cols = comparison_df.select_dtypes(include=['float64', 'float32', 'int64']).columns
        for col in numeric_cols:
            comparison_df[col] = comparison_df[col].round(4)
        
        return comparison_df
    else:
        return pd.DataFrame()

def plot_metric_comparison(comparison_df, metric, title=None):
    """
    Create a bar chart comparing the given metric across all models.
    
    Args:
        comparison_df: DataFrame with model metrics
        metric: Metric to plot
        title: Optional title for the plot
    
    Returns:
        Plotly figure object
    """
    if metric not in comparison_df.columns:
        return None
    
    # Sort by metric value
    is_higher_better = metric in ['r2_score', 'explained_variance']
    df_sorted = comparison_df.sort_values(by=metric, ascending=not is_higher_better)
    
    # Create model labels with group information
    df_sorted['model_label'] = df_sorted['Model'] + ' (' + df_sorted['Group'] + ')'
    
    # Create color mapping by group
    groups = df_sorted['Group'].unique()
    color_map = {group: f'rgba({hash(group) % 255}, {(hash(group) // 255) % 255}, {(hash(group) // (255*255)) % 255}, 0.7)' 
                 for group in groups}
    
    # Create bar colors based on group
    bar_colors = [color_map[group] for group in df_sorted['Group']]
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=df_sorted['model_label'],
            y=df_sorted[metric],
            marker_color=bar_colors
        )
    ])
    
    # Update layout
    fig.update_layout(
        title=title or f'Comparison of {metric} across models',
        xaxis_title='Model',
        yaxis_title=metric,
        height=500,
        xaxis={'categoryorder': 'array', 'categoryarray': df_sorted['model_label']}
    )
    
    return fig

def create_radar_chart(comparison_df, metrics=None, normalize=True):
    """
    Create a radar chart comparing models across multiple metrics.
    
    Args:
        comparison_df: DataFrame with model metrics
        metrics: List of metrics to include (if None, use all numeric columns)
        normalize: Whether to normalize metrics to 0-1 scale
    
    Returns:
        Plotly figure object
    """
    if comparison_df.empty:
        return None
    
    # Select metrics to use
    if metrics is None:
        metrics = [col for col in comparison_df.columns 
                   if col not in ['Group', 'Model'] 
                   and pd.api.types.is_numeric_dtype(comparison_df[col])]
    else:
        # Filter to only include metrics that exist in the DataFrame
        metrics = [m for m in metrics if m in comparison_df.columns]
    
    if not metrics:
        return None
    
    # Determine if higher is better for each metric
    higher_is_better = {
        'r2_score': True,
        'explained_variance': True,
        'rmse': False,
        'mse': False,
        'mae': False,
        'mape': False,
        'max_error': False,
        'median_absolute_error': False,
        'norm_rmse': False,
        'norm_mae': False
    }
    
    # Create a copy for normalization
    df = comparison_df.copy()
    
    # Normalize metrics to 0-1 scale where 1 is always better
    if normalize:
        for metric in metrics:
            if not pd.api.types.is_numeric_dtype(df[metric]):
                continue
            
            # Determine whether higher or lower is better for this metric
            is_higher_better = higher_is_better.get(metric, True)
            
            # Normalize based on whether higher or lower is better
            if is_higher_better:
                min_val = df[metric].min()
                max_val = df[metric].max()
                if max_val > min_val:
                    df[metric] = (df[metric] - min_val) / (max_val - min_val)
                    else:
                min_val = df[metric].min()
                max_val = df[metric].max()
                if max_val > min_val:
                    df[metric] = 1 - (df[metric] - min_val) / (max_val - min_val)
    
    # Create radar chart
    fig = go.Figure()
    
    # Add traces for each model
    for _, row in df.iterrows():
        model_name = row['Model']
        group_name = row['Group']
        
        # Get values for this model
        values = [row[m] for m in metrics]
        # Add the first value again to close the polygon
        values.append(values[0])
        
        # Create a label that includes both model and group
        label = f"{model_name} ({group_name})"
        
        # Add trace
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics + [metrics[0]],  # Complete the loop
            fill='toself',
            name=label
        ))
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1] if normalize else None
            )
        ),
        title="Model Comparison Across Metrics",
        height=600,
        showlegend=True
    )
    
    return fig

def plot_training_progress(all_histories, selected_models, metric='rmse'):
    """
    Plot training progress for selected models.
    
    Args:
        all_histories: Dictionary of history objects by group and model
        selected_models: List of tuples (group_name, model_name) to plot
        metric: Metric to plot
    
    Returns:
        Plotly figure object
    """
    if not selected_models:
        return None
    
    fig = go.Figure()
    has_data = False
    
    for group_name, model_name in selected_models:
        if group_name in all_histories and model_name in all_histories[group_name]:
            history = all_histories[group_name][model_name]
            
            # Check for history in standardized format
            if isinstance(history, dict) and 'history' in history and isinstance(history['history'], dict):
                history_data = history['history']
                
                if metric in history_data and history_data[metric] is not None and len(history_data[metric]) > 0:
                    y_values = history_data[metric]
                    
                    # Add trace
            fig.add_trace(go.Scatter(
                        y=y_values,
                mode='lines',
                        name=f"{model_name} ({group_name})"
                    ))
                    has_data = True
    
    # Only proceed with layout if we have data
    if has_data:
    # Update layout
    fig.update_layout(
            title=f"Training Progress - {metric}",
        xaxis_title="Epoch",
            yaxis_title=metric,
            height=500
        )
    return fig 
    else:
        return None 
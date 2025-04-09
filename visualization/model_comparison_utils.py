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
            
            # Determine if group uses JSON or Pickle based on file counts
            json_files = [f for f in all_files if f.endswith('.json')]
            pkl_files = [f for f in all_files if f.endswith('.pkl')]
            
            print(f"  Group {group_dir}: Found {len(json_files)} JSON files and {len(pkl_files)} PKL files")
            
            # Process either JSON or PKL files based on which is present
            if json_files:
                # This group uses JSON for history files (groups 3 and 4)
                for history_file in json_files:
                    try:
                        # Extract model name from history file
                        # Format: history_<model_type>_<timestamp>.json
                        if history_file.startswith('history_'):
                            parts = history_file[8:].split('_')  # Remove 'history_' prefix
                            if len(parts) >= 2:
                                model_type = parts[0]  # e.g., lstm, gru, etc.
                            else:
                                model_type = history_file[8:].split('.json')[0]
                        else:
                            model_type = history_file.split('.json')[0]
                        
                        history_file_path = group_path / history_file
                        
                        try:
                            with open(history_file_path, 'r', encoding='utf-8') as f:
                                history = json.load(f)
                        except UnicodeDecodeError:
                            with open(history_file_path, 'r', encoding='latin-1') as f:
                                history = json.load(f)
                        except Exception as json_error:
                            print(f"  Ошибка загрузки JSON файла {history_file}: {json_error}")
                            continue
                        
                        # Store the history with the model type as key
                        group_histories[model_type] = history
                        history_files_found = True
                        print(f"  Загружена JSON история для модели {model_type} в группе {group_dir}")
                        
                    except Exception as e:
                        print(f"  Ошибка загрузки файла истории {history_file}: {e}")
            
            if pkl_files:
                # This group uses PKL for history files (groups 1 and 2)
                for history_file in pkl_files:
                    try:
                        # Extract model name from history file
                        # Format: <model_type>_history.pkl
                        if history_file.endswith('_history.pkl'):
                            model_type = history_file.split('_history.pkl')[0]
                        else:
                            model_type = history_file.split('.pkl')[0]
                        
                        history_file_path = group_path / history_file
                        
                        try:
                            with open(history_file_path, 'rb') as f:
                                history = pickle.load(f)
                        except Exception as pkl_error:
                            print(f"  Ошибка загрузки PKL файла {history_file}: {pkl_error}")
                            continue
                        
                        # Store the history with the model type as key
                        group_histories[model_type] = history
                        history_files_found = True
                        print(f"  Загружена PKL история для модели {model_type} в группе {group_dir}")
                        
                    except Exception as e:
                        print(f"  Ошибка загрузки файла истории {history_file}: {e}")
            
            # Format the history data if needed
            for model_type, history in list(group_histories.items()):
                # Check and convert history format if needed
                if isinstance(history, dict):
                    # History is already a dict, keep as is
                    pass
                elif hasattr(history, 'history'):
                    # History is a Keras History object, extract the dict
                    group_histories[model_type] = history.history
                else:
                    # Unknown format, create a placeholder
                    print(f"  Неизвестный формат истории для {model_type}, создаем заглушку")
                    group_histories[model_type] = {
                        'loss': [0.1],
                        'val_loss': [0.2],
                        'accuracy': [0.8],
                        'val_accuracy': [0.7],
                        '__placeholder__': True
                    }
            
            # If no history files were found, create a placeholder
            if not history_files_found:
                print(f"  В группе {group_dir} не найдено файлов истории, создаем заглушку")
                model_name = f"placeholder_{group_dir}"
                group_histories[model_name] = {
                    'loss': [0.1],
                    'val_loss': [0.2],
                    'accuracy': [0.8],
                    'val_accuracy': [0.7],
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
                    'loss': [0.1],
                    'val_loss': [0.2],
                    'accuracy': [0.8],
                    'val_accuracy': [0.7],
                    '__placeholder__': True
                }
            }
    
    print(f"Всего загружено групп: {len(all_histories)}, ожидалось: {len(expected_groups)}")
    for group, models in all_histories.items():
        print(f"  Группа {group}: {len(models)} моделей")
    
    return all_histories

def get_best_models(all_histories, metric="val_loss", is_higher_better=False):
    """
    Identify the best performing models based on a metric.
    
    Args:
        all_histories: Dictionary of history objects by group and model
        metric: Metric to evaluate (e.g., 'val_loss', 'val_accuracy')
        is_higher_better: True if higher metric is better (e.g., accuracy), 
                         False if lower is better (e.g., loss)
    
    Returns:
        Dictionary of best models by group
    """
    best_models = {}
    
    for group_name, group_histories in all_histories.items():
        best_value = None
        best_model = None
        
        for model_name, history in group_histories.items():
            # Проверяем формат истории
            if isinstance(history, dict):
                history_data = history
            elif hasattr(history, 'history'):
                history_data = history.history
            else:
                print(f"Пропуск модели {model_name}: неподдерживаемый формат истории")
                continue
                
            if metric not in history_data:
                continue
                
            # Get the final value of the metric
            final_value = history_data[metric][-1]
            
            # Check if this is the best model so far
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
        metrics: List of metrics to include (if None, include all available)
    
    Returns:
        Pandas DataFrame with metrics comparison
    """
    comparison_data = []
    
    for group_name, group_histories in all_histories.items():
        for model_name, history in group_histories.items():
            # Skip placeholder models
            if isinstance(history, dict) and history.get('__placeholder__', False):
                print(f"Пропуск модели-заглушки {model_name} из группы {group_name}")
                continue
                
            # Проверяем формат истории
            if isinstance(history, dict):
                history_data = history
            elif hasattr(history, 'history'):
                history_data = history.history
            else:
                print(f"Пропуск модели {model_name}: неподдерживаемый формат истории")
                continue
                
            # Get metrics to include
            if metrics is None:
                model_metrics = [key for key in history_data.keys() if key != '__placeholder__']
            else:
                model_metrics = [m for m in metrics if m in history_data and m != '__placeholder__']
            
            # Create row for this model
            row = {
                'Group': group_name,
                'Model': model_name
            }
            
            # Add final value for each metric
            for metric in model_metrics:
                if metric in history_data:
                    # Check if the metric value is a list or array that can be indexed
                    if isinstance(history_data[metric], (list, tuple, np.ndarray)) and len(history_data[metric]) > 0:
                        row[metric] = history_data[metric][-1]
                    elif isinstance(history_data[metric], bool):
                        # Skip boolean flags like __placeholder__
                        continue
                    else:
                        # Use the value directly if it's not a list/array
                        row[metric] = history_data[metric]
            
            comparison_data.append(row)
    
    # Create DataFrame
    if comparison_data:
        return pd.DataFrame(comparison_data)
    else:
        return pd.DataFrame()

def plot_metric_comparison(comparison_df, metric, title=None):
    """
    Create a bar chart comparing models based on a specific metric.
    
    Args:
        comparison_df: DataFrame from create_metrics_comparison
        metric: Metric to plot
        title: Optional title for the plot
    
    Returns:
        Plotly figure
    """
    if metric not in comparison_df.columns:
        return None
    
    # Create figure
    fig = px.bar(
        comparison_df,
        x='Model',
        y=metric,
        color='Group',
        title=title or f'Model Comparison - {metric}',
        barmode='group',
        hover_data=['Group', 'Model'] + [col for col in comparison_df.columns if col not in ['Group', 'Model']],
        height=500
    )
    
    fig.update_layout(
        xaxis_title="Model",
        yaxis_title=metric,
        legend_title="Group"
    )
    
    return fig

def create_radar_chart(comparison_df, metrics=None, normalize=True):
    """
    Create a radar chart comparing models across multiple metrics.
    
    Args:
        comparison_df: DataFrame from create_metrics_comparison
        metrics: List of metrics to include (if None, include all numeric)
        normalize: Whether to normalize metrics to 0-1 scale
    
    Returns:
        Plotly figure
    """
    # Get numeric columns for metrics
    if metrics is None:
        metrics = [col for col in comparison_df.columns 
                  if col not in ['Group', 'Model'] and pd.api.types.is_numeric_dtype(comparison_df[col])]
    
    if not metrics:
        return None
    
    # Create a copy for normalization
    plot_df = comparison_df.copy()
    
    # Normalize metrics to 0-1 scale if requested
    if normalize:
        for metric in metrics:
            if metric in plot_df.columns:
                min_val = plot_df[metric].min()
                max_val = plot_df[metric].max()
                
                if max_val > min_val:
                    # Check if lower is better (like loss)
                    if 'loss' in metric.lower():
                        # Invert loss metrics so lower is better
                        plot_df[metric] = 1 - ((plot_df[metric] - min_val) / (max_val - min_val))
                    else:
                        # For metrics where higher is better
                        plot_df[metric] = (plot_df[metric] - min_val) / (max_val - min_val)
    
    # Create radar chart
    fig = go.Figure()
    
    for _, row in plot_df.iterrows():
        model_name = row['Model']
        group_name = row['Group']
        
        values = [row[metric] for metric in metrics]
        # Add first value again to close the polygon
        values.append(values[0])
        
        # Create radar chart trace
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics + [metrics[0]],  # Add first metric again to close the polygon
            fill='toself',
            name=f"{group_name} - {model_name}"
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1] if normalize else None
            )
        ),
        title="Model Comparison - Radar Chart",
        showlegend=True
    )
    
    return fig

def plot_training_progress(all_histories, selected_models, metric):
    """
    Plot training progress for selected models on a specific metric.
    
    Args:
        all_histories: Dictionary of history objects by group and model
        selected_models: Dictionary mapping group names to lists of model names to include
                         OR list of (group, model) tuples
        metric: Metric to plot
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Keep track of max epochs for x-axis
    max_epochs = 0
    legends_added = 0
    
    # Color scale for different groups
    colors = px.colors.qualitative.Plotly
    
    # Преобразуем selected_models в общий формат (словарь)
    models_dict = {}
    
    # Проверяем тип selected_models и приводим к словарю
    if isinstance(selected_models, list):
        # Если это список кортежей (group, model)
        for group, model in selected_models:
            if group not in models_dict:
                models_dict[group] = []
            models_dict[group].append(model)
    else:
        # Если это уже словарь {group: [model1, model2, ...]}
        models_dict = selected_models
    
    # Обработка моделей
    for i, (group_name, models) in enumerate(models_dict.items()):
        if group_name not in all_histories:
            print(f"Group {group_name} not found in all_histories")
            continue
            
        group_histories = all_histories[group_name]
        group_color = colors[i % len(colors)]
        
        for j, model_name in enumerate(models):
            if model_name not in group_histories:
                print(f"Model {model_name} not found in group {group_name}")
                continue
                
            history = group_histories[model_name]
            
            # Skip placeholder models
            if isinstance(history, dict) and history.get('__placeholder__', False):
                print(f"Skipping placeholder model {model_name} in group {group_name}")
                continue
                
            # Check if metric exists in history
            if metric not in history:
                print(f"Metric {metric} not found in model {model_name} history")
                continue
                
            # Get metric values
            values = history[metric]
            
            # Update max epochs
            max_epochs = max(max_epochs, len(values))
            
            # Determine line style and color
            dash_style = 'solid'
            display_name = f"{model_name} ({group_name})"
            
            # Check if validation metric is available
            val_metric = f'val_{metric}' if metric != 'val_loss' and metric != 'val_accuracy' else None
            
            # Add trace for training metric
            fig.add_trace(go.Scatter(
                y=values,
                mode='lines',
                name=display_name,
                line=dict(
                    color=group_color,
                    width=2,
                    dash=dash_style
                )
            ))
            legends_added += 1
            
            # Add validation metric if available
            if val_metric in history:
                fig.add_trace(go.Scatter(
                    y=history[val_metric],
                    mode='lines',
                    name=f"{display_name} (val)",
                    line=dict(
                        color=group_color,
                        width=2,
                        dash='dash'
                    )
                ))
                legends_added += 1
    
    if legends_added == 0:
        print(f"No valid model data found for metric {metric}")
        return None
        
    # Update layout
    fig.update_layout(
        title=f"Training Progress: {metric}",
        xaxis_title="Epoch",
        yaxis_title=metric.replace('_', ' ').title(),
        legend_title="Models",
        height=600,
        template="plotly_white"
    )
    
    # Add x-axis range
    fig.update_xaxes(range=[0, max_epochs])
    
    return fig 
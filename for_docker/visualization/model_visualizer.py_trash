import os
import pickle
import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except ImportError:
    # Создаем заглушки для случая, когда TensorFlow не установлен
    tf = None
    def load_model(*args, **kwargs):
        st.error("""
        ### Ошибка импорта TensorFlow
        
        TensorFlow не установлен или не совместим с текущей версией Python.
        
        Возможные решения:
        1. Установите TensorFlow вручную: `pip install tensorflow>=2.16.1`
        2. Используйте Python 3.10 или 3.11 вместо Python 3.12
        3. Переустановите визуализатор с помощью скрипта install_visualization.bat
        """)
        st.stop()
import seaborn as sns
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Neural Network Models Visualizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set paths
BASE_PATH = Path("../model_save_preset")
MODELS_PATH = BASE_PATH / "models"
HISTORY_PATH = BASE_PATH / "history"

def load_history(history_file):
    """Load training history from pickle or JSON file."""
    if not history_file:
        return None
        
    try:
        if history_file.endswith('.pkl'):
            with open(history_file, 'rb') as f:
                history = pickle.load(f)
        elif history_file.endswith('.json'):
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except UnicodeDecodeError:
                # Try different encoding if UTF-8 fails
                with open(history_file, 'r', encoding='latin-1') as f:
                    history = json.load(f)
        else:
            st.error(f"Unsupported history file format: {history_file}")
            return None
            
        return history
    except Exception as e:
        st.error(f"Error loading history file {history_file}: {e}")
        return None

def load_model_summary(model_file):
    """Load model architecture and return summary."""
    try:
        model = load_model(model_file)
        # Capture summary as string
        summary_list = []
        model.summary(print_fn=lambda x: summary_list.append(x))
        summary_str = '\n'.join(summary_list)
        
        # Get model info
        model_info = {
            'layers': len(model.layers),
            'parameters': model.count_params(),
            'input_shape': model.input_shape,
            'output_shape': model.output_shape
        }
        
        return model_info, summary_str
    except Exception as e:
        st.error(f"Error loading model file {model_file}: {e}")
        return None, None

def plot_history(history):
    """Plot training history."""
    # Проверяем формат истории
    if isinstance(history, dict):
        history_data = history
    elif hasattr(history, 'history'):
        history_data = history.history
    else:
        print("Неподдерживаемый формат истории")
        return None
        
    # Создание графика с метриками
    fig = go.Figure()
    
    # Получаем все метрики, кроме валидационных
    metrics = [key for key in history_data.keys() if not key.startswith('val_')]
    
    # Добавляем графики для каждой метрики
    for metric in metrics:
        fig.add_trace(go.Scatter(
            y=history_data[metric],
            mode='lines',
            name=f'Training {metric}'
        ))
        
        # Если есть валидационная метрика, добавляем ее
        val_metric = f'val_{metric}'
        if val_metric in history_data:
            fig.add_trace(go.Scatter(
                y=history_data[val_metric],
                mode='lines',
                name=f'Validation {metric}'
            ))
    
    # Настройка графика
    fig.update_layout(
        title="Training Metrics",
        xaxis_title="Epoch",
        yaxis_title="Value",
        legend_title="Metrics",
        height=400
    )
    
    return fig

def compare_models(histories, metric):
    """Compare models based on a specific metric."""
    fig = go.Figure()
    
    for model_name, history in histories.items():
        # Проверяем формат истории
        if isinstance(history, dict):
            history_data = history
        elif hasattr(history, 'history'):
            history_data = history.history
        else:
            print(f"Пропуск модели {model_name}: неподдерживаемый формат истории")
            continue
            
        if metric in history_data:
            fig.add_trace(go.Scatter(
                y=history_data[metric],
                mode='lines',
                name=f'{model_name}'
            ))
    
    fig.update_layout(
        title=f"Model Comparison - {metric}",
        xaxis_title="Epoch",
        yaxis_title=metric,
        legend_title="Models",
        height=500
    )
    
    return fig

def scan_model_directories():
    """Scan model directories and return available models."""
    available_models = {}
    
    # Check if directories exist
    if not os.path.exists(MODELS_PATH):
        st.error(f"Models directory not found: {MODELS_PATH}")
        return available_models
    
    if not os.path.exists(HISTORY_PATH):
        st.error(f"History directory not found: {HISTORY_PATH}")
        return available_models
    
    # Scan model groups (directories)
    for group_dir in sorted(os.listdir(MODELS_PATH)):
        group_path = MODELS_PATH / group_dir
        history_group_path = HISTORY_PATH / group_dir
        
        if os.path.isdir(group_path) and os.path.isdir(history_group_path):
            model_files = {}
            # Scan for model files (.h5)
            for name in os.listdir(group_path):
                if name.endswith('.h5'):
                    # Extract model name without extension
                    if name.startswith('model_'):
                        # Format for groups 3 and 4: model_type_timestamp.h5
                        model_name = name.split('.')[0]
                    else:
                        # Format for groups 1 and 2: type.h5
                        model_name = name.split('.')[0]
                    
                    model_files[model_name] = str(group_path / name)
            
            history_files = {}
            # Scan for both .pkl and .json history files
            for name in os.listdir(history_group_path):
                if name.endswith('.pkl') or name.endswith('.json'):
                    history_name = None
                    
                    # Handle different naming formats
                    if name.endswith('_history.pkl'):
                        # Format for groups 1 and 2: type_history.pkl
                        history_name = name.split('_history.pkl')[0]
                    elif name.startswith('history_'):
                        # Format for groups 3 and 4: history_type_timestamp.json
                        history_name = 'model_' + name[8:].split('.json')[0]
                    
                    if history_name:
                        history_files[history_name] = str(history_group_path / name)
            
            # Map models to histories
            group_models = {}
            
            # For groups 1 and 2 with standard naming
            for model_name, model_path in model_files.items():
                if model_name in history_files:
                    # Direct match between model name and history name
                    group_models[model_name] = {
                        'model_path': model_path,
                        'history_path': history_files[model_name]
                    }
                else:
                    # Try matching based on prefixes (for groups 3 and 4)
                    matched = False
                    for history_name, history_path in history_files.items():
                        # Extract the model type from history_name
                        if '_' in history_name:
                            model_type = history_name.split('_')[1]  # Extract type from model_type_timestamp
                            
                            if model_name.startswith(model_type) or model_name.endswith(model_type):
                                group_models[model_name] = {
                                    'model_path': model_path,
                                    'history_path': history_path
                                }
                                matched = True
                                break
                    
                    if not matched and '_' in model_name:
                        # Try another matching approach for models with timestamps
                        model_type = model_name.split('_')[1]  # Extract type from model_type_timestamp
                        
                        for history_name, history_path in history_files.items():
                            if model_type in history_name:
                                group_models[model_name] = {
                                    'model_path': model_path,
                                    'history_path': history_path
                                }
                                break
            
            # Add to available models
            if group_models:
                available_models[group_dir] = group_models
    
    return available_models

def main():
    st.title("Neural Network Models Visualizer")
    
    # Scan directories for models
    available_models = scan_model_directories()
    
    if not available_models:
        st.warning("No models found in the specified directories.")
        return
    
    # Sidebar - Model selection
    st.sidebar.header("Model Selection")
    
    # Select model group
    selected_group = st.sidebar.selectbox(
        "Select Model Group:", 
        options=list(available_models.keys()),
        format_func=lambda x: x
    )
    
    if selected_group:
        # Select model from group
        selected_model_name = st.sidebar.selectbox(
            "Select Model:",
            options=list(available_models[selected_group].keys()),
            format_func=lambda x: x
        )
        
        if selected_model_name:
            selected_model = available_models[selected_group][selected_model_name]
            
            # Load model and history
            model_info, model_summary = load_model_summary(selected_model['model_path'])
            history = load_history(selected_model['history_path'])
            
            # Display model information
            st.header(f"Model: {selected_model_name} (Group: {selected_group})")
            
            # Model summary and training history in columns
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Model Information")
                if model_info:
                    st.write(f"**Layers:** {model_info['layers']}")
                    st.write(f"**Parameters:** {model_info['parameters']:,}")
                    st.write(f"**Input Shape:** {model_info['input_shape']}")
                    st.write(f"**Output Shape:** {model_info['output_shape']}")
                    
                    with st.expander("Model Architecture"):
                        st.text(model_summary)
            
            with col2:
                st.subheader("Training History")
                if history:
                    # Проверяем формат истории
                    if isinstance(history, dict):
                        history_data = history
                    elif hasattr(history, 'history'):
                        history_data = history.history
                    else:
                        st.error("Неподдерживаемый формат истории обучения")
                        return
                        
                    # Get available metrics
                    metrics = [key for key in history_data.keys() if not key.startswith('val_')]
                    
                    # Plot training history
                    fig = plot_history(history)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Display final metrics
                    st.subheader("Final Metrics")
                    metrics_df = pd.DataFrame({
                        'Metric': list(history_data.keys()),
                        'Final Value': [history_data[key][-1] for key in history_data.keys()]
                    })
                    st.table(metrics_df)
    
    # Model comparison section
    st.header("Model Comparison")
    
    # Select models to compare
    st.subheader("Select Models to Compare")
    
    # Create multiselect for model groups
    selected_groups = st.multiselect(
        "Select Model Groups:",
        options=list(available_models.keys()),
        default=[next(iter(available_models.keys()))] if available_models else []
    )
    
    if selected_groups:
        # Collect all models from selected groups
        all_models = {}
        for group in selected_groups:
            for model_name, model_info in available_models[group].items():
                all_models[f"{group} - {model_name}"] = model_info
        
        # Select specific models to compare
        selected_models_to_compare = st.multiselect(
            "Select Models:",
            options=list(all_models.keys()),
            default=list(all_models.keys())[:min(5, len(all_models))]
        )
        
        if selected_models_to_compare:
            # Load histories for selected models
            histories = {}
            for model_key in selected_models_to_compare:
                model_info = all_models[model_key]
                history = load_history(model_info['history_path'])
                if history:
                    histories[model_key] = history
            
            # Get common metrics across all selected models
            common_metrics = set()
            for history in histories.values():
                # Проверяем формат истории
                if isinstance(history, dict):
                    metrics = [key for key in history.keys()]
                elif hasattr(history, 'history'):
                    metrics = [key for key in history.history.keys()]
                else:
                    continue
                    
                if not common_metrics:
                    common_metrics = set(metrics)
                else:
                    common_metrics &= set(metrics)
            
            if common_metrics:
                # Select metric for comparison
                selected_metric = st.selectbox(
                    "Select Metric for Comparison:",
                    options=sorted(list(common_metrics))
                )
                
                if selected_metric:
                    # Create comparison plot
                    fig = compare_models(histories, selected_metric)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No common metrics found across selected models.")

if __name__ == "__main__":
    main() 
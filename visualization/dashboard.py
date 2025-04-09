import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import traceback

# Настройка для отображения кириллицы
plt.rcParams['font.family'] = 'DejaVu Sans'

# Add visualization directory to path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# Import functions from modules - определим для линтера
from model_comparison_utils import load_all_histories
from model_comparison_utils import get_best_models
from model_comparison_utils import create_metrics_comparison
from model_comparison_utils import plot_metric_comparison
from model_comparison_utils import create_radar_chart
from model_comparison_utils import plot_training_progress
from model_architecture import visualize_keras_model

# Импортирование функций
try:
    # Импортируем существующие функции, исправляем неверные импорты
    from model_comparison_utils import (
        load_all_histories,
        get_best_models,
        create_metrics_comparison,
        plot_metric_comparison,
        create_radar_chart,
        plot_training_progress
    )
    from model_architecture import visualize_keras_model
except ModuleNotFoundError as e:
    if "tensorflow" in str(e):
        st.error("""
        ### Ошибка импорта TensorFlow
        
        TensorFlow не установлен или не совместим с текущей версией Python.
        
        Возможные решения:
        1. Установите TensorFlow вручную: `pip install tensorflow>=2.16.1`
        2. Используйте Python 3.10 или 3.11 вместо Python 3.12
        3. Переустановите визуализатор с помощью скрипта install_visualization.bat
        """)
        st.stop()
    else:
        st.error(f"Ошибка импорта: {e}")
        st.stop()

# Set page configuration
st.set_page_config(
    page_title="Neural Network Models Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set paths
BASE_PATH = Path("../model_save_preset")
MODELS_PATH = BASE_PATH / "models"
HISTORY_PATH = BASE_PATH / "history"

def main():
    # Add custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4F8BF9;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1F618D;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .section-divider {
        margin-top: 2rem;
        margin-bottom: 2rem;
        border-top: 1px solid #e0e0e0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Application title
    st.markdown('<div class="main-header">Neural Network Models Dashboard</div>', unsafe_allow_html=True)
    
    # Load all model histories
    all_histories = load_all_histories(str(BASE_PATH))
    
    if not all_histories:
        st.error("No model histories found in the specified directory.")
        return
    
    # Sidebar
    st.sidebar.title("Navigation")
    
    # Navigation
    pages = [
        "Model Overview",
        "Training History",
        "Model Comparison",
        "Best Models"
    ]
    
    selected_page = st.sidebar.radio("Go to", pages)
    
    # Create metrics comparison dataframe
    comparison_df = create_metrics_comparison(all_histories)
    
    # Display selected page
    if selected_page == "Model Overview":
        display_model_overview(all_histories, comparison_df)
    
    elif selected_page == "Training History":
        display_training_history(all_histories)
    
    elif selected_page == "Model Comparison":
        display_model_comparison(all_histories, comparison_df)
    
    elif selected_page == "Best Models":
        display_best_models(all_histories)

def display_model_overview(all_histories, comparison_df):
    """Display overview of all models."""
    st.markdown('<div class="sub-header">Model Overview</div>', unsafe_allow_html=True)
    
    # Make sure all expected groups are in the history data
    expected_groups = ['1 old', '2 new', '3 alt_model', '4 alt_new_model']
    
    # Count models by group
    model_counts = {}
    for group in expected_groups:
        if group in all_histories:
            # Count non-placeholder models
            real_models = [model for model, history in all_histories[group].items() 
                        if not (isinstance(history, dict) and history.get('__placeholder__', False))]
            model_counts[group] = len(real_models)
        else:
            model_counts[group] = 0
    
    # Create metrics for display
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Model Groups", len(expected_groups))
        
        # Display models by group
        st.subheader("Models by Group")
        for group in expected_groups:
            count = model_counts.get(group, 0)
            st.write(f"**{group}:** {count} models")
    
    with col2:
        total_models = sum(model_counts.values())
        st.metric("Total Models", total_models)
        
        # Pie chart of models by group
        if model_counts:
            # Filter out groups with zero models for the chart
            non_empty_groups = {k: v for k, v in model_counts.items() if v > 0}
            
            # Create pie chart
            if non_empty_groups:
                fig = px.pie(
                    values=list(non_empty_groups.values()),
                    names=list(non_empty_groups.keys()),
                    title="Model Distribution by Group"
                )
                st.plotly_chart(fig)
            else:
                st.warning("No models found in any group.")
    
    # Display model metrics table
    st.markdown('<div class="sub-header">Model Metrics Overview</div>', unsafe_allow_html=True)
    
    if not comparison_df.empty:
        # Make sure comparison dataframe has entries for all expected groups
        if 'Group' in comparison_df.columns:
            missing_groups = [g for g in expected_groups if g not in comparison_df['Group'].unique()]
            
            if missing_groups:
                st.warning(f"Note: The following model groups have no metrics available: {', '.join(missing_groups)}")
        
        # Allow sorting by different columns
        sort_by = st.selectbox(
            "Sort by metric:", 
            ["Group", "Model"] + [col for col in comparison_df.columns if col not in ["Group", "Model"]]
        )
        
        ascending = st.checkbox("Sort ascending", value=True)
        
        # Sort and display
        sorted_df = comparison_df.sort_values(by=sort_by, ascending=ascending)
        st.dataframe(sorted_df)
        
        # Download button for the dataframe
        csv = sorted_df.to_csv(index=False)
        st.download_button(
            label="Download metrics as CSV",
            data=csv,
            file_name="model_metrics.csv",
            mime="text/csv"
        )
    else:
        st.warning("No model metrics available for comparison.")

def display_model_architecture(all_histories):
    """Display architecture visualization for selected model."""
    st.markdown('<div class="sub-header">Model Architecture Visualization</div>', unsafe_allow_html=True)
    
    # Select group and model
    col1, col2 = st.columns(2)
    
    with col1:
        selected_group = st.selectbox(
            "Select Model Group:", 
            options=list(all_histories.keys())
        )
    
    with col2:
        if selected_group:
            selected_model = st.selectbox(
                "Select Model:",
                options=list(all_histories[selected_group].keys())
            )
        else:
            selected_model = None
    
    if selected_group and selected_model:
        # Try to find the matching model file in the models directory
        model_found = False
        model_files = []
        model_path = None
        
        # Different model naming patterns for different groups
        group_path = MODELS_PATH / selected_group
        
        if os.path.exists(group_path):
            model_files = [f for f in os.listdir(group_path) if f.endswith('.h5')]
            
            # 1. Для групп 1 и 2: model_name.h5 (например, lstm.h5)
            exact_match = f"{selected_model}.h5"
            if exact_match in model_files:
                model_path = group_path / exact_match
                model_found = True
            
            # 2. Для групп 3 и 4: model_<model_type>_<timestamp>.h5 (например, model_lstm_20250407_0741.h5)
            if not model_found:
                for model_file in model_files:
                    if model_file.startswith('model_') and selected_model in model_file:
                        model_path = group_path / model_file
                        model_found = True
                        break
            
            # 3. Для групп 3 и 4 с best в имени: model_<model_type>_best_<timestamp>.h5
            if not model_found:
                for model_file in model_files:
                    if model_file.startswith('model_') and 'best' in model_file and selected_model in model_file:
                        model_path = group_path / model_file
                        model_found = True
                        break
        
        if model_found and model_path:
            # Visualize model
            with st.spinner(f"Loading model architecture for {selected_model}..."):
                model, architecture_fig, summary_fig = visualize_keras_model(str(model_path))
                
                if model:
                    # Display visualizations
                    if architecture_fig:
                        st.plotly_chart(architecture_fig, use_container_width=True)
                    else:
                        st.warning("Could not generate architecture visualization for this model.")
                    
                    if summary_fig:
                        st.plotly_chart(summary_fig, use_container_width=True)
                    else:
                        # If we couldn't create a summary figure, try to display a text summary
                        st.subheader("Model Summary")
                        summary_list = []
                        try:
                            model.summary(print_fn=lambda x: summary_list.append(x))
                            st.text('\n'.join(summary_list))
                        except Exception as e:
                            st.error(f"Could not generate model summary: {e}")
                else:
                    st.error(f"Failed to load model. This could be due to compatibility issues with TensorFlow or corrupted model files.")
        else:
            st.error(f"No matching model file found for {selected_model} in group {selected_group}.")
            
            # Show available model files to help diagnose
            if model_files:
                st.info(f"Available model files in {selected_group}:")
                for file in model_files:
                    st.write(f"- {file}")
            else:
                st.warning(f"No model files found in group {selected_group}.")

def display_training_history(all_histories):
    """Display training history for selected models."""
    st.markdown('<div class="sub-header">Training History</div>', unsafe_allow_html=True)
    
    # Available metrics
    all_metrics = set()
    for group_histories in all_histories.values():
        for history in group_histories.values():
            if isinstance(history, dict):
                all_metrics.update(history.keys())
            elif hasattr(history, 'history'):
                all_metrics.update(history.history.keys())
    
    # Filter metrics (убираем служебные)
    all_metrics = [m for m in all_metrics if not m.startswith('__')]
    
    if not all_metrics:
        st.warning("No metrics found in history data.")
        return
    
    # Select models to display
    st.subheader("Select Models")
    
    # Multiselect for groups
    selected_groups = st.multiselect(
        "Select Model Groups:",
        options=list(all_histories.keys()),
        default=[next(iter(all_histories.keys()))] if all_histories else []
    )
    
    if not selected_groups:
        st.warning("Please select at least one model group.")
        return
    
    # Create options for model selection
    model_options = []
    for group in selected_groups:
        for model in all_histories[group]:
            # Отфильтруем placeholder-модели
            if model.startswith('placeholder_'):
                continue
            model_options.append((group, model))
    
    # Select models
    selected_models = []
    for group, model in model_options:
        if st.checkbox(f"{group} - {model}", value=group == selected_groups[0]):
            selected_models.append((group, model))
    
    if not selected_models:
        st.warning("Please select at least one model.")
        return
    
    # Select metric
    selected_metric = st.selectbox(
        "Select Metric:",
        options=sorted(list(all_metrics))
    )
    
    # Конвертируем список кортежей в словарь для plot_training_progress
    models_dict = {}
    for group, model in selected_models:
        if group not in models_dict:
            models_dict[group] = []
        models_dict[group].append(model)
    
    # Create plot
    fig = plot_training_progress(all_histories, models_dict, selected_metric)
    
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"No data available for selected metric: {selected_metric}")

def display_model_comparison(all_histories, comparison_df):
    """Display comparison of models on various metrics."""
    st.markdown('<div class="sub-header">Model Comparison</div>', unsafe_allow_html=True)
    
    # Get all available metrics from the history data
    all_metrics = set()
    for group, models in all_histories.items():
        for model, history in models.items():
            # Skip placeholder models
            if isinstance(history, dict) and history.get('__placeholder__', False):
                continue
                
            all_metrics.update(history.keys())
    
    # Remove placeholder flag from metrics
    if '__placeholder__' in all_metrics:
        all_metrics.remove('__placeholder__')
    
    # Filter to show only common metrics and sort alphabetically
    valid_metrics = sorted([m for m in all_metrics if m != '__placeholder__' and not isinstance(m, bool)])
    
    # Metric selection
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_metric = st.selectbox(
            "Select metric for comparison:",
            options=valid_metrics,
            index=min(1, len(valid_metrics)-1) if len(valid_metrics) > 0 else 0
        )
        
        # Select display mode
        display_mode = st.radio(
            "Display mode:",
            ["Line chart", "Bar chart", "Radar chart"]
        )
    
    with col2:
        # Model selection
        model_options = {}
        
        # Organize models by group for the selection UI
        for group in all_histories:
            for model in all_histories[group]:
                # Skip placeholders
                if model.startswith('placeholder_') or (
                   isinstance(all_histories[group][model], dict) and 
                   all_histories[group][model].get('__placeholder__', False)):
                    continue
                    
                # Add to selection options
                model_id = f"{group}|{model}"
                model_options[model_id] = f"{model} ({group})"
        
        # Allow selecting multiple models for comparison
        if model_options:
            default_models = list(model_options.keys())[:min(5, len(model_options))]
            selected_models = st.multiselect(
                "Select models to compare:",
                options=list(model_options.keys()),
                format_func=lambda x: model_options[x],
                default=default_models
            )
        else:
            selected_models = []
            st.warning("No models found for comparison")
    
    if not valid_metrics:
        st.warning("No metrics found in the history data.")
        return
    
    if not selected_models:
        st.warning("Please select at least one model for comparison.")
        return
    
    # Display the selected visualization
    if display_mode == "Line chart" and selected_metric and selected_models:
        # Convert selected_models to dictionary for plot_training_progress
        models_dict = {}
        for model_id in selected_models:
            group, model = model_id.split('|')
            if group not in models_dict:
                models_dict[group] = []
            models_dict[group].append(model)
        
        # Plot line chart
        st.subheader(f"Training Progress: {selected_metric}")
        progress_fig = plot_training_progress(all_histories, models_dict, selected_metric)
        if progress_fig:
            st.plotly_chart(progress_fig, use_container_width=True)
        else:
            st.warning(f"Could not generate training progress chart for {selected_metric}.")
    
    elif display_mode == "Bar chart" and selected_metric and selected_models:
        # Create bar chart using plotly
        st.subheader(f"Final {selected_metric} Comparison")
        
        # Collect data for bar chart
        bar_data = []
        
        for model_id in selected_models:
            group, model = model_id.split('|')
            
            # Skip if model not in histories
            if group not in all_histories or model not in all_histories[group]:
                continue
                
            history = all_histories[group][model]
            
            # Skip placeholder models
            if isinstance(history, dict) and history.get('__placeholder__', False):
                continue
                
            # Check if metric exists
            if selected_metric not in history:
                continue
                
            # Get final value
            final_value = history[selected_metric][-1]
            
            bar_data.append({
                'Model': f"{model} ({group})",
                'Value': final_value
            })
        
        if bar_data:
            # Create bar chart
            bar_df = pd.DataFrame(bar_data)
            fig = px.bar(
                bar_df, 
                x='Model', 
                y='Value',
                title=f"Final {selected_metric} Comparison",
                color='Model'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"No data available for {selected_metric} in the selected models.")
    
    elif display_mode == "Radar chart" and selected_models:
        st.subheader("Multi-Metric Comparison (Radar Chart)")
        
        # Allow selecting metrics for radar chart
        selected_radar_metrics = st.multiselect(
            "Select metrics for radar chart:",
            options=valid_metrics,
            default=valid_metrics[:min(5, len(valid_metrics))]
        )
        
        if selected_radar_metrics:
            # Filter comparison df to only selected models and metrics
            filtered_models = []
            for model_id in selected_models:
                group, model = model_id.split('|')
                filtered_models.append((group, model))
            
            # Create radar chart
            if comparison_df is not None and not comparison_df.empty:
                filtered_df = comparison_df[
                    comparison_df.apply(
                        lambda row: (row['Group'], row['Model']) in filtered_models, 
                        axis=1
                    )
                ]
                
                if not filtered_df.empty:
                    radar_fig = create_radar_chart(filtered_df, selected_radar_metrics)
                    if radar_fig:
                        st.plotly_chart(radar_fig, use_container_width=True)
                    else:
                        st.warning("Could not create radar chart. Check if metrics data is available.")
                else:
                    st.warning("No matching data found for selected models in the comparison dataframe.")
            else:
                st.warning("No comparison data available for creating radar chart.")
        else:
            st.info("Please select at least one metric for the radar chart.")

def display_best_models(all_histories):
    """Display the best models by group."""
    st.markdown('<div class="sub-header">Best Models</div>', unsafe_allow_html=True)
    
    # Select metric for evaluation
    metrics = set()
    for group, models in all_histories.items():
        for model, history in models.items():
            if hasattr(history, 'history'):
                metrics.update(history.history.keys())
            elif isinstance(history, dict):
                metrics.update(history.keys())
    
    metrics = sorted(list(metrics))
    
    if not metrics:
        st.warning("No metrics found in model histories.")
        return
        
    selected_metric = st.selectbox(
        "Select metric to identify best models:",
        options=metrics
    )
    
    if not selected_metric:
        st.warning("Please select a metric to continue.")
        return
    
    # Determine if higher is better based on metric name
    is_higher_better = not ('loss' in selected_metric.lower())
    
    if st.button(f"Find Best Models by {selected_metric}"):
        # Get best models for each group
        best_models = get_best_models(all_histories, metric=selected_metric, is_higher_better=is_higher_better)
        
        if best_models:
            # Create comparison table
            best_data = []
            for group, info in best_models.items():
                model_name = info['model']
                value = info['value']
                
                best_data.append({
                    "Group": group,
                    "Best Model": model_name,
                    selected_metric: value
                })
            
            best_df = pd.DataFrame(best_data)
            
            # Display table with best models
            st.subheader("Best Models by Group")
            st.dataframe(best_df)
            
            # Create visual comparison
            fig = px.bar(
                best_df,
                x="Group",
                y=selected_metric,
                color="Best Model",
                title=f"Best Models by {selected_metric}",
                hover_data=["Group", "Best Model"]
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"No comparable models found using metric: {selected_metric}")

if __name__ == "__main__":
    main() 
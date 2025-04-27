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

# Define standard metrics used in the updated history format
STANDARD_METRICS = [
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

# Define metrics where higher values are better
HIGHER_IS_BETTER = {
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

# Friendly display names for metrics
METRIC_DISPLAY_NAMES = {
    'rmse': 'Root Mean Squared Error (RMSE)',
    'norm_rmse': 'Normalized RMSE',
    'r2_score': 'R² Score',
    'explained_variance': 'Explained Variance',
    'mae': 'Mean Absolute Error (MAE)',
    'mape': 'Mean Absolute Percentage Error (MAPE)',
    'max_error': 'Maximum Error',
    'median_absolute_error': 'Median Absolute Error',
    'mse': 'Mean Squared Error (MSE)',
    'norm_mae': 'Normalized MAE'
}

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
    
    # Display warning about incompatible pickle files
    st.warning("""
    **Note:** Due to NumPy version incompatibility, we are unable to load history files from groups '1 old' and '2 new'.
    Only models from groups '3 alt_model' and '4 alt_new_model' will be displayed.
    """)
    
    # Filter out groups 1 and 2 as they are incompatible
    compatible_histories = {
        group: models 
        for group, models in all_histories.items() 
        if group not in ['1 old', '2 new'] or (len(models) > 0 and not all(
            isinstance(history, dict) and history.get('__placeholder__', False) 
            for history in models.values()
        ))
    }
    
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
    comparison_df = create_metrics_comparison(compatible_histories, metrics=STANDARD_METRICS)
    
    # Display selected page
    if selected_page == "Model Overview":
        display_model_overview(compatible_histories, comparison_df)
    
    elif selected_page == "Training History":
        display_training_history(compatible_histories)
    
    elif selected_page == "Model Comparison":
        display_model_comparison(compatible_histories, comparison_df)
    
    elif selected_page == "Best Models":
        display_best_models(compatible_histories)

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
        
        try:
        if os.path.exists(group_path):
                # Get all model files in the group directory
            model_files = [f for f in os.listdir(group_path) if f.endswith('.h5')]
            
                # First attempt: Direct match with model name
                if f"{selected_model}.h5" in model_files:
                    model_path = group_path / f"{selected_model}.h5"
                model_found = True
                elif f"model_{selected_model}.h5" in model_files:
                    model_path = group_path / f"model_{selected_model}.h5"
                        model_found = True
                else:
                    # Partial match based on model name components
                    model_name_parts = selected_model.split('_')
                for model_file in model_files:
                        if any(part in model_file for part in model_name_parts if len(part) > 2):
                        model_path = group_path / model_file
                        model_found = True
                        break
        except Exception as e:
            st.error(f"Error searching for model files: {e}")
            return
        
        # Display model info
        if model_found and model_path:
            try:
                # Try to load and visualize the model
                visualize_keras_model(str(model_path))
                
                # Display model file location
                st.info(f"Model file: {model_path}")
                        except Exception as e:
                st.error(f"Error visualizing model: {e}")
                st.code(traceback.format_exc())
        else:
            st.warning(f"No matching model file found for {selected_model} in {selected_group}.")
            
            if model_files:
                st.info(f"Available model files in {selected_group}: {', '.join(model_files)}")
            else:
                st.info(f"No model files found in {selected_group}.")
    else:
        st.info("Select a model group and model to visualize its architecture.")

def display_training_history(all_histories):
    """Display training history for selected models."""
    st.markdown('<div class="sub-header">Training History</div>', unsafe_allow_html=True)
    
    # Allow selection of multiple models for comparison
    col1, col2 = st.columns(2)
    
    with col1:
        # Select group
        selected_group = st.selectbox(
            "Select Group:",
            options=list(all_histories.keys())
        )
    
    with col2:
    # Select metric
    selected_metric = st.selectbox(
        "Select Metric:",
            options=STANDARD_METRICS,
            format_func=lambda x: METRIC_DISPLAY_NAMES.get(x, x)
        )
    
    if selected_group:
        # Get models in the selected group
        models = list(all_histories[selected_group].keys())
        
        # Filter out placeholder models
        models = [model for model in models 
                 if not (isinstance(all_histories[selected_group][model], dict) 
                         and all_histories[selected_group][model].get('__placeholder__', False))]
        
        if models:
            # Select models for comparison
            selected_models = st.multiselect(
                "Select Models to Compare:",
                options=models,
                default=models[:min(3, len(models))]  # Default to first 3 models
            )
            
            if selected_models:
                # Create list of (group, model) tuples
                model_tuples = [(selected_group, model) for model in selected_models]
                
                # Plot training progress
                fig = plot_training_progress(all_histories, model_tuples, selected_metric)
    
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
                    st.warning(f"No training history data available for the selected models and metric: {selected_metric}")
                
                # Show detailed metrics for each selected model
                st.markdown('<div class="sub-header">Model Metrics Details</div>', unsafe_allow_html=True)
                
                for model in selected_models:
                    history = all_histories[selected_group][model]
                    
                    # Create expander for each model
                    with st.expander(f"Model: {model}"):
                        # Check if history contains the 'history' key with metrics
                        if isinstance(history, dict) and 'history' in history and isinstance(history['history'], dict):
                            # Display information about the model
                            if 'model_name' in history:
                                st.write(f"**Model Name:** {history['model_name']}")
                            if 'group_name' in history:
                                st.write(f"**Group:** {history['group_name']}")
                            
                            # Display final metrics if available
                            if 'metrics' in history and isinstance(history['metrics'], dict):
                                st.write("**Final Evaluation Metrics:**")
                                metrics_df = pd.DataFrame({
                                    'Metric': list(history['metrics'].keys()),
                                    'Value': list(history['metrics'].values())
                                })
                                st.dataframe(metrics_df)
                            
                            # Display training history for different metrics
                            history_data = history['history']
                            
                            # Get available metrics
                            available_metrics = [m for m in STANDARD_METRICS if m in history_data]
                            
                            if available_metrics:
                                # Create tabs for different metric categories
                                tab1, tab2 = st.tabs(["Error Metrics", "Performance Metrics"])
                                
                                # Error metrics tab
                                with tab1:
                                    error_metrics = ['rmse', 'mse', 'mae', 'mape', 'max_error', 'median_absolute_error', 'norm_rmse', 'norm_mae']
                                    error_metrics = [m for m in error_metrics if m in available_metrics]
                                    
                                    if error_metrics:
                                        for metric in error_metrics:
                                            # Safely check if metric has data and is not None
                                            if metric in history_data and history_data[metric] is not None and len(history_data[metric]) > 0:
                                                values = history_data[metric]
                                                epochs = list(range(1, len(values) + 1))
                                                
                                                fig = px.line(
                                                    x=epochs,
                                                    y=values,
                                                    title=f"{METRIC_DISPLAY_NAMES.get(metric, metric)} History",
                                                    labels={'x': 'Epoch', 'y': metric}
                                                )
                                                st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.info("No error metrics available for this model.")
                                
                                # Performance metrics tab
                                with tab2:
                                    perf_metrics = ['r2_score', 'explained_variance']
                                    perf_metrics = [m for m in perf_metrics if m in available_metrics]
                                    
                                    if perf_metrics:
                                        for metric in perf_metrics:
                                            # Safely check if metric has data and is not None
                                            if metric in history_data and history_data[metric] is not None and len(history_data[metric]) > 0:
                                                values = history_data[metric]
                                                epochs = list(range(1, len(values) + 1))
                                                
                                                fig = px.line(
                                                    x=epochs,
                                                    y=values,
                                                    title=f"{METRIC_DISPLAY_NAMES.get(metric, metric)} History",
                                                    labels={'x': 'Epoch', 'y': metric}
                                                )
                                                st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.info("No performance metrics available for this model.")
                            else:
                                st.warning("No training history metrics available for this model.")
                        else:
                            st.warning("This model does not have proper training history data.")
            else:
                st.info("Select at least one model to view its training history.")
        else:
            st.warning(f"No models found in group: {selected_group}")
    else:
        st.info("Select a group to view available models.")

def display_model_comparison(all_histories, comparison_df):
    """Display comparison of models across different metrics."""
    st.markdown('<div class="sub-header">Model Comparison</div>', unsafe_allow_html=True)
    
    if comparison_df.empty:
        st.warning("No model metrics available for comparison.")
        return
    
    # Create tabs for different visualization types
    tab1, tab2, tab3 = st.tabs(["Metric Comparison", "Radar Chart", "Group Comparison"])
    
    with tab1:
        # Select metric for comparison
        selected_metric = st.selectbox(
            "Select Metric to Compare:",
            options=STANDARD_METRICS,
            format_func=lambda x: METRIC_DISPLAY_NAMES.get(x, x)
        )
        
        # Create comparison chart
        fig = plot_metric_comparison(comparison_df, selected_metric)
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            
            # Display explanation about the metric
            metric_explanations = {
                'rmse': "**Root Mean Square Error (RMSE)** measures the average magnitude of the prediction errors. Lower values indicate better model performance.",
                'norm_rmse': "**Normalized RMSE** is the RMSE divided by the range of observed values. This helps compare errors across different scales.",
                'r2_score': "**R² Score** (Coefficient of Determination) indicates how well the model fits the data. Values closer to 1 indicate better fit.",
                'explained_variance': "**Explained Variance** measures the proportion of variance in the dependent variable that is predictable from the independent variables.",
                'mae': "**Mean Absolute Error (MAE)** measures the average magnitude of errors without considering their direction. Lower values are better.",
                'mape': "**Mean Absolute Percentage Error (MAPE)** measures prediction accuracy as a percentage. Lower values indicate better accuracy.",
                'max_error': "**Maximum Error** shows the maximum residual error, representing the worst case prediction. Lower is better.",
                'median_absolute_error': "**Median Absolute Error** is the median of all absolute differences between the target and the prediction. Less sensitive to outliers.",
                'mse': "**Mean Squared Error (MSE)** is the average of squared differences between predicted and actual values. Lower values indicate better fit.",
                'norm_mae': "**Normalized MAE** is the MAE divided by the range of observed values. Helps compare errors across different scales."
            }
            
            if selected_metric in metric_explanations:
                st.markdown(metric_explanations[selected_metric])
        else:
            st.warning(f"No data available for metric: {selected_metric}")
    
    with tab2:
        # Select metrics for radar chart
        selected_metrics = st.multiselect(
            "Select Metrics for Radar Chart:",
            options=STANDARD_METRICS,
            default=STANDARD_METRICS[:5],  # Default to first 5 metrics
            format_func=lambda x: METRIC_DISPLAY_NAMES.get(x, x)
        )
        
        # Select models to include
        all_models = []
        for group in all_histories:
            for model in all_histories[group]:
                if not (isinstance(all_histories[group][model], dict) and 
                   all_histories[group][model].get('__placeholder__', False)):
                    all_models.append((group, model))
        
            selected_models = st.multiselect(
            "Select Models to Include (max 5):",
            options=[(f"{model} ({group})") for group, model in all_models],
            default=[f"{all_models[i][1]} ({all_models[i][0]})" for i in range(min(3, len(all_models)))]
        )
        
        # Limit to max 5 models for readability
        if len(selected_models) > 5:
            st.warning("For better readability, only the first 5 selected models will be shown in the radar chart.")
            selected_models = selected_models[:5]
        
        if selected_metrics and selected_models:
            # Filter comparison_df to include only selected models
            selected_model_tuples = []
            for model_str in selected_models:
                # Extract model and group from the combined string
                model, group = model_str.split(" (")
                group = group.rstrip(")")
                selected_model_tuples.append((group, model))
            
            filtered_df = comparison_df[
                comparison_df.apply(
                    lambda row: (row['Group'], row['Model']) in [(group, model) for group, model in selected_model_tuples], 
                    axis=1
                )
            ]
            
            if not filtered_df.empty and len(selected_metrics) > 0:
                # Create radar chart
                fig = create_radar_chart(filtered_df, metrics=selected_metrics)
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add explanation about normalization
                    st.markdown("""
                    **Note about the Radar Chart:**
                    - The metrics are normalized to a scale of 0-1, where 1 always represents the best performance.
                    - For error metrics (RMSE, MAE, etc.), lower original values are better, so the scale is inverted.
                    - For performance metrics (R², explained variance), higher values are better.
                    """)
                else:
                    st.warning("Unable to create radar chart with the selected data.")
            else:
                st.warning("No data available for the selected models and metrics.")
        else:
            st.info("Select at least one metric and one model to create a radar chart.")
    
    with tab3:
        # Group comparison
        st.subheader("Compare Model Groups")
        
        # Calculate group averages
        if 'Group' in comparison_df.columns:
            # Create group metrics for each standardized metric
            group_metrics = []
            
            for group in comparison_df['Group'].unique():
                group_data = {'Group': group}
                
                # Filter dataframe for this group
                group_df = comparison_df[comparison_df['Group'] == group]
                
                # Calculate average for each metric
                for metric in STANDARD_METRICS:
                    if metric in group_df.columns:
                        group_data[f'avg_{metric}'] = group_df[metric].mean()
                        group_data[f'best_{metric}'] = group_df[metric].min() if not HIGHER_IS_BETTER.get(metric, False) else group_df[metric].max()
                
                # Add count of models
                group_data['model_count'] = len(group_df)
                
                group_metrics.append(group_data)
            
            # Create dataframe
            if group_metrics:
                group_df = pd.DataFrame(group_metrics)
                
                # Display model counts
                st.write("**Number of Models per Group:**")
                
                # Create bar chart for model counts
                fig = px.bar(
                    group_df,
                    x='Group',
                    y='model_count',
                    title='Models per Group',
                    labels={'model_count': 'Number of Models', 'Group': 'Group'},
                    color='Group'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Select metric for group comparison
                selected_metric = st.selectbox(
                    "Select Metric for Group Comparison:",
                    options=STANDARD_METRICS,
                    format_func=lambda x: METRIC_DISPLAY_NAMES.get(x, x),
                    key="group_comparison_metric"
                )
                
                if selected_metric:
                    # Create tabs for different comparison types
                    tab_avg, tab_best = st.tabs(["Average Performance", "Best Performance"])
                    
                    with tab_avg:
                        # Average performance
                        avg_col = f'avg_{selected_metric}'
                        
                        if avg_col in group_df.columns:
            # Create bar chart
            fig = px.bar(
                                group_df,
                                x='Group',
                                y=avg_col,
                                title=f'Average {METRIC_DISPLAY_NAMES.get(selected_metric, selected_metric)} by Group',
                                labels={avg_col: f'Average {selected_metric}', 'Group': 'Group'},
                                color='Group'
                            )
                            
                            # Adjust y-axis (lower bound for better metrics)
                            if not HIGHER_IS_BETTER.get(selected_metric, False):
                                fig.update_layout(yaxis_range=[0, group_df[avg_col].max() * 1.1])
                            
            st.plotly_chart(fig, use_container_width=True)
        else:
                            st.warning(f"No average data available for metric: {selected_metric}")
                    
                    with tab_best:
                        # Best performance
                        best_col = f'best_{selected_metric}'
                        
                        if best_col in group_df.columns:
                            # Create bar chart
                            fig = px.bar(
                                group_df,
                                x='Group',
                                y=best_col,
                                title=f'Best {METRIC_DISPLAY_NAMES.get(selected_metric, selected_metric)} by Group',
                                labels={best_col: f'Best {selected_metric}', 'Group': 'Group'},
                                color='Group'
                            )
                            
                            # Adjust y-axis (lower bound for better metrics)
                            if not HIGHER_IS_BETTER.get(selected_metric, False):
                                fig.update_layout(yaxis_range=[0, group_df[best_col].max() * 1.1])
                            
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                            st.warning(f"No best performance data available for metric: {selected_metric}")
            else:
                st.warning("No group data available for comparison.")
        else:
            st.warning("Group information not available in the metrics data.")

def display_best_models(all_histories):
    """Display the best performing models based on different metrics."""
    st.markdown('<div class="sub-header">Best Performing Models</div>', unsafe_allow_html=True)
    
    # Select metric for determining best models
    selected_metric = st.selectbox(
        "Select Metric for Ranking:",
        options=STANDARD_METRICS,
        format_func=lambda x: METRIC_DISPLAY_NAMES.get(x, x)
    )
    
    # Determine if higher is better for this metric
    is_higher_better = HIGHER_IS_BETTER.get(selected_metric, False)
    
    # Get best models based on the selected metric
        best_models = get_best_models(all_histories, metric=selected_metric, is_higher_better=is_higher_better)
        
        if best_models:
        # Create dataframe for best models
        best_df = pd.DataFrame([
            {
                'Group': group,
                'Model': info['model'],
                selected_metric: info['value']
            }
            for group, info in best_models.items()
        ])
        
        # Sort based on metric (ascending or descending based on is_higher_better)
        best_df = best_df.sort_values(by=selected_metric, ascending=not is_higher_better)
        
        # Display as table
        st.write(f"**Best Models by {METRIC_DISPLAY_NAMES.get(selected_metric, selected_metric)}:**")
            st.dataframe(best_df)
            
        # Create bar chart
            fig = px.bar(
                best_df,
            x='Group',
                y=selected_metric,
            color='Model',
            title=f'Best Models by {METRIC_DISPLAY_NAMES.get(selected_metric, selected_metric)}',
            barmode='group',
            text='Model'
        )
        
        # Adjust label positions
        fig.update_traces(textposition='outside')
        
        # Adjust y-axis (lower bound for better metrics)
        if not is_higher_better:
            fig.update_layout(yaxis_range=[0, best_df[selected_metric].max() * 1.1])
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display information about each best model
        st.markdown('<div class="sub-header">Detailed Information</div>', unsafe_allow_html=True)
        
        for group, info in best_models.items():
            model_name = info['model']
            
            with st.expander(f"Best model in {group}: {model_name}"):
                if group in all_histories and model_name in all_histories[group]:
                    history = all_histories[group][model_name]
                    
                    # Check if history contains the necessary information
                    if isinstance(history, dict):
                        if 'model_name' in history:
                            st.write(f"**Model Name:** {history['model_name']}")
                        if 'group_name' in history:
                            st.write(f"**Group:** {history['group_name']}")
                        
                        # Display final metrics if available
                        if 'metrics' in history and isinstance(history['metrics'], dict):
                            st.write("**Final Evaluation Metrics:**")
                            
                            # Create two columns for metrics display
                            col1, col2 = st.columns(2)
                            
                            metrics = sorted(history['metrics'].keys())
                            half = len(metrics) // 2
                            
                            # First column of metrics
                            with col1:
                                for metric in metrics[:half]:
                                    display_name = METRIC_DISPLAY_NAMES.get(metric, metric)
                                    value = history['metrics'][metric]
                                    st.metric(label=display_name, value=f"{value:.4f}")
                            
                            # Second column of metrics
                            with col2:
                                for metric in metrics[half:]:
                                    display_name = METRIC_DISPLAY_NAMES.get(metric, metric)
                                    value = history['metrics'][metric]
                                    st.metric(label=display_name, value=f"{value:.4f}")
                        
                        # Display training history for best metric
                        if 'history' in history and isinstance(history['history'], dict):
                            if selected_metric in history['history'] and history['history'][selected_metric] is not None and len(history['history'][selected_metric]) > 0:
                                st.write(f"**Training History for {METRIC_DISPLAY_NAMES.get(selected_metric, selected_metric)}:**")
                                
                                values = history['history'][selected_metric]
                                epochs = list(range(1, len(values) + 1))
                                
                                fig = px.line(
                                    x=epochs,
                                    y=values,
                                    title=f"{METRIC_DISPLAY_NAMES.get(selected_metric, selected_metric)} History",
                                    labels={'x': 'Epoch', 'y': selected_metric}
                                )
            st.plotly_chart(fig, use_container_width=True)
        else:
                        st.warning("Model history data is not in the expected format.")
                else:
                    st.warning(f"Model information not found for {model_name} in {group}.")
    else:
        st.warning("No best models found for the selected metric.")

if __name__ == "__main__":
    main() 
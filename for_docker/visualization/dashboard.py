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
import logging
import datetime
import configparser
from logging.handlers import RotatingFileHandler

# Загрузка конфигурации
config = configparser.ConfigParser()
config_path = Path(__file__).parent / "config.ini"
if os.path.exists(config_path):
    # Используем кодировку UTF-8 для поддержки кириллицы
    config.read(config_path, encoding='utf-8')
else:
    # Создаем конфигурацию по умолчанию если файл не существует
    config["Paths"] = {"models_directory": "../model_save_preset"}
    config["Logging"] = {"log_level": "INFO"}
    # Сохраняем конфигурацию по умолчанию с кодировкой UTF-8
    with open(config_path, 'w', encoding='utf-8') as f:
        config.write(f)
    print(f"Создан файл конфигурации по умолчанию: {config_path}")

# Настройка логирования
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

log_file = log_dir / f"dashboard_{datetime.datetime.now().strftime('%Y%m%d')}.log"
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

file_handler = RotatingFileHandler(
    log_file, 
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5,
    encoding='utf-8'
)
file_handler.setFormatter(log_formatter)

logger = logging.getLogger("dashboard")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

# Функция для логирования и отображения ошибок
def log_error(error, section=None, context=None):
    """Логирует ошибку и показывает информацию пользователю."""
    error_msg = str(error)
    tb = traceback.format_exc()
    
    # Логируем ошибку
    if section:
        logger.error(f"Ошибка в секции '{section}': {error_msg}")
    else:
        logger.error(f"Ошибка: {error_msg}")
    
    if context:
        logger.error(f"Контекст: {context}")
    
    logger.error(f"Traceback: {tb}")
    
    # Показываем информацию пользователю
    st.error(f"Произошла ошибка: {error_msg}")
    
    # Опционально показываем traceback для разработчиков
    with st.expander("Техническая информация"):
        st.code(tb)
        if context:
            st.write(f"Контекст: {context}")

# Логируем запуск приложения
logger.info("Дашборд запущен")

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
BASE_PATH = Path(config["Paths"]["models_directory"])
MODELS_PATH = BASE_PATH / "models"
HISTORY_PATH = BASE_PATH / "history"

# Define standard metrics used in the updated history format
STANDARD_METRICS = [
    'rmse',
    'explained_variance',
    'mae',
    'max_error',
    'mse'
]

# Define metrics where higher values are better
HIGHER_IS_BETTER = {
    'explained_variance': True,
    'rmse': False,
    'mse': False,
    'mae': False,
    'max_error': False
}

# Friendly display names for metrics
METRIC_DISPLAY_NAMES = {
    'rmse': 'Корень из среднеквадратичной ошибки (RMSE)',
    'explained_variance': 'Объяснённая дисперсия',
    'mae': 'Средняя абсолютная ошибка (MAE)',
    'max_error': 'Максимальная ошибка',
    'mse': 'Среднеквадратичная ошибка (MSE)'
}

# Функция для сбора системной информации
def log_system_info():
    """Собирает и логирует информацию о системе."""
    try:
        import platform
        import sys
        import tensorflow as tf
        import numpy as np
        import pandas as pd
        import streamlit as st
        
        # Используем .resolve() для корректного отображения путей с кириллицей
        models_path = MODELS_PATH.resolve()
        history_path = HISTORY_PATH.resolve()
        
        system_info = {
            "OS": platform.platform(),
            "Python": platform.python_version(),
            "TensorFlow": tf.__version__ if 'tf' in locals() else "Не установлен",
            "NumPy": np.__version__,
            "Pandas": pd.__version__,
            "Streamlit": st.__version__,
            "Работающий каталог": os.getcwd(),
            "Конфигурационный файл": str(config_path),
            "PATH к моделям из конфигурации": config["Paths"]["models_directory"],
            "PATH к моделям (полный)": str(models_path),
            "PATH к истории (полный)": str(history_path)
        }
        
        logger.info("Системная информация:")
        for key, value in system_info.items():
            logger.info(f"  {key}: {value}")
        
        # Проверяем наличие директорий с данными
        if os.path.exists(models_path):
            try:
                groups = [d for d in os.listdir(models_path) if os.path.isdir(models_path / d)]
                logger.info(f"Найдены группы моделей ({len(groups)}): {', '.join(groups)}")
                
                # Считаем файлы моделей в каждой группе
                for group in groups:
                    try:
                        group_path = (models_path / group).resolve()
                        model_files = [f for f in os.listdir(group_path) if f.endswith('.h5')]
                        logger.info(f"  Группа {group}: {len(model_files)} файлов модели")
                    except Exception as group_err:
                        logger.error(f"  Ошибка при чтении группы {group}: {group_err}")
            except Exception as dir_err:
                logger.error(f"Ошибка при чтении директории моделей: {dir_err}")
        else:
            logger.warning(f"Директория моделей не найдена: {models_path}")
        
        if os.path.exists(history_path):
            try:
                groups = [d for d in os.listdir(history_path) if os.path.isdir(history_path / d)]
                logger.info(f"Найдены группы историй ({len(groups)}): {', '.join(groups)}")
                
                # Считаем файлы историй в каждой группе
                for group in groups:
                    try:
                        group_path = (history_path / group).resolve()
                        history_files = [f for f in os.listdir(group_path) if f.endswith(('.json', '.pkl'))]
                        logger.info(f"  Группа {group}: {len(history_files)} файлов истории")
                    except Exception as group_err:
                        logger.error(f"  Ошибка при чтении группы {group}: {group_err}")
            except Exception as dir_err:
                logger.error(f"Ошибка при чтении директории историй: {dir_err}")
        else:
            logger.warning(f"Директория историй не найдена: {history_path}")
            
    except Exception as e:
        logger.error(f"Ошибка при сборе системной информации: {e}")
        logger.error(traceback.format_exc())

# Логируем системную информацию при запуске
log_system_info()

def main():
    try:
        logger.info("Запуск основной функции дашборда")
        
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
        st.markdown('<div class="main-header">Панель управления моделями нейронных сетей</div>', unsafe_allow_html=True)
        
        # Показываем текущий путь из конфигурации
        st.info(f"Текущий путь к моделям: {config['Paths']['models_directory']}")
        
        # Load all model histories
        logger.info(f"Загрузка историй моделей из директории: {BASE_PATH}")
        all_histories = load_all_histories(str(BASE_PATH))
        
        if not all_histories:
            logger.warning("Истории моделей не найдены")
            st.error("Истории моделей не найдены в указанной директории.")
            return
        
        available_groups = list(all_histories.keys())
        logger.info(f"Загружено {sum(len(models) for models in all_histories.values())} моделей из {len(all_histories)} групп")
        
        # Удаляем статическое предупреждение о несовместимости групп
        # и заменяем на динамическое, которое будет показываться только если есть проблемные группы
        problematic_groups = []
        for group, models in all_histories.items():
            # Проверяем, есть ли в группе модели-заглушки с флагом __placeholder__
            if all(isinstance(history, dict) and history.get('__placeholder__', False) for history in models.values()):
                problematic_groups.append(group)
        
        if problematic_groups:
            st.warning(f"""
                **Примечание:** Некоторые группы моделей не могут быть загружены полностью: {', '.join(problematic_groups)}.
                Это может быть вызвано несовместимостью форматов данных или отсутствием файлов.
            """)
        
        # Sidebar
        st.sidebar.title("Навигация")
        
        # Navigation
        pages = [
            "Обзор моделей",
        #    "История обучения",
            "Сравнение моделей",
            "Лучшие модели"
        ]
        
        selected_page = st.sidebar.radio("Выберите раздел", pages)
        logger.info(f"Переход на страницу: {selected_page}")
        
        # Добавляем секцию для диагностики
        st.sidebar.markdown("---")
        st.sidebar.title("Диагностика")
        
        # Добавляем информацию о пути к моделям
        with st.sidebar.expander("Настройки путей"):
            # Показываем текущий путь из конфигурации
            current_path = config["Paths"]["models_directory"]
            st.text_input("Текущий путь к моделям:", value=current_path, disabled=True)
            
            # Поле для ввода нового пути
            new_path = st.text_input("Новый путь к моделям:", value=current_path)
            
            # Кнопка для сохранения нового пути
            if st.button("Сохранить путь"):
                try:
                    # Проверяем, существует ли указанный путь
                    path_to_check = Path(new_path)
                    if os.path.exists(path_to_check) and os.path.exists(path_to_check / "models") and os.path.exists(path_to_check / "history"):
                        # Обновляем конфигурацию
                        config["Paths"]["models_directory"] = new_path
                        with open(config_path, 'w', encoding='utf-8') as f:
                            config.write(f)
                        st.success(f"Путь обновлен. Перезагрузите страницу для применения изменений.")
                        logger.info(f"Путь к моделям обновлен: {new_path}")
                    else:
                        st.error(f"Ошибка: Путь {new_path} не существует или не содержит необходимых директорий 'models' и 'history'")
                        logger.warning(f"Попытка установить неверный путь: {new_path}")
                except Exception as e:
                    st.error(f"Ошибка при обновлении пути: {e}")
                    logger.error(f"Ошибка при обновлении пути: {e}")
        
        # Добавляем кнопку для выгрузки лог-файлов
        if st.sidebar.button("Выгрузить логи"):
            try:
                log_files = list(log_dir.glob("*.log"))
                
                if log_files:
                    # Создаем selectbox для выбора лог-файла
                    selected_log = st.sidebar.selectbox(
                        "Выберите лог-файл:", 
                        options=log_files,
                        format_func=lambda x: x.name
                    )
                    
                    if selected_log:
                        # Читаем содержимое файла
                        with open(selected_log, 'r', encoding='utf-8') as f:
                            log_content = f.read()
                        
                        # Создаем кнопку для скачивания лог-файла
                        st.sidebar.download_button(
                            label="Скачать лог-файл",
                            data=log_content,
                            file_name=selected_log.name,
                            mime="text/plain"
                        )
                        
                        # Показываем последние 20 строк лога
                        st.sidebar.markdown("### Последние записи лога:")
                        log_lines = log_content.splitlines()
                        
                        if len(log_lines) > 20:
                            log_preview = "\n".join(log_lines[-20:])
                        else:
                            log_preview = log_content
                        
                        st.sidebar.code(log_preview, language="text")
                else:
                    st.sidebar.warning("Лог-файлы не найдены")
                    logger.warning("Попытка выгрузки логов: файлы не найдены")
            except Exception as e:
                st.sidebar.error(f"Ошибка при чтении лог-файлов: {e}")
                logger.error(f"Ошибка при чтении лог-файлов: {e}")
        
        # Create metrics comparison dataframe
        try:
            logger.info("Создание таблицы сравнения метрик")
            comparison_df = create_metrics_comparison(all_histories, metrics=STANDARD_METRICS)
            logger.info(f"Создана таблица сравнения с {len(comparison_df)} моделями")
        except Exception as e:
            log_error(e, section="create_metrics_comparison", context=f"metrics={STANDARD_METRICS}")
            comparison_df = pd.DataFrame()
        
        # Display selected page
        try:
            if selected_page == "Обзор моделей":
                display_model_overview(all_histories, comparison_df)
            elif selected_page == "История обучения":
                display_training_history(all_histories)
            elif selected_page == "Сравнение моделей":
                display_model_comparison(all_histories, comparison_df)
            elif selected_page == "Лучшие модели":
                display_best_models(all_histories)
        except Exception as e:
            log_error(e, section=selected_page)

    except Exception as e:
        log_error(e, section="main")

def display_model_overview(all_histories, comparison_df):
    """Display overview of all models."""
    st.markdown('<div class="sub-header">Обзор моделей</div>', unsafe_allow_html=True)
    
    # Get actual groups dynamically from all_histories
    available_groups = list(all_histories.keys())
    
    # Count models by group
    model_counts = {}
    for group in available_groups:
        # Count non-placeholder models
        real_models = [model for model, history in all_histories[group].items() 
                    if not (isinstance(history, dict) and history.get('__placeholder__', False))]
        model_counts[group] = len(real_models)
    
    # Create metrics for display
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Всего групп моделей", len(available_groups))
        
        # Display models by group
        st.subheader("Модели по группам")
        for group in available_groups:
            count = model_counts.get(group, 0)
            st.write(f"**{group}:** {count} моделей")
    
    with col2:
        total_models = sum(model_counts.values())
        st.metric("Всего моделей", total_models)
        
        # Pie chart of models by group
        if model_counts:
            # Filter out groups with zero models for the chart
            non_empty_groups = {k: v for k, v in model_counts.items() if v > 0}
            
            # Create pie chart
            if non_empty_groups:
                fig = px.pie(
                    values=list(non_empty_groups.values()),
                    names=list(non_empty_groups.keys()),
                    title="Распределение моделей по группам"
                )
                st.plotly_chart(fig)
            else:
                st.warning("Не найдено моделей ни в одной группе.")
    
    # Display model metrics table
    st.markdown('<div class="sub-header">Обзор метрик моделей</div>', unsafe_allow_html=True)
    
    if not comparison_df.empty:
        # Make sure comparison dataframe has entries for all groups
        if 'group' in comparison_df.columns:
            metrics_groups = comparison_df['group'].unique().tolist()
            missing_groups = [g for g in available_groups if g not in metrics_groups]
            
            if missing_groups:
                st.warning(f"Примечание: Для следующих групп моделей нет доступных метрик: {', '.join(missing_groups)}")
        
        # Удаляем столбцы mape, r2, norm_rmse, norm_mae и median_absolute_error из таблицы, если они есть
        columns_to_drop = []
        if 'mape' in comparison_df.columns:
            columns_to_drop.append('mape')
        if 'r2' in comparison_df.columns:
            columns_to_drop.append('r2')
        if 'r2_score' in comparison_df.columns:
            columns_to_drop.append('r2_score')
        if 'norm_rmse' in comparison_df.columns:
            columns_to_drop.append('norm_rmse')
        if 'norm_mae' in comparison_df.columns:
            columns_to_drop.append('norm_mae')
        if 'median_absolute_error' in comparison_df.columns:
            columns_to_drop.append('median_absolute_error')
        if 'explained_variance' in comparison_df.columns:
            columns_to_drop.append('explained_variance')
        if 'max_error' in comparison_df.columns:
            columns_to_drop.append('max_error')
            
        if columns_to_drop:
            comparison_df = comparison_df.drop(columns=columns_to_drop)
            
        # Allow sorting by different columns, исключая только group и model
        sort_by = st.selectbox(
            "Сортировать по метрике:", 
            [col for col in comparison_df.columns if col not in ["group", "model"]]
        )
        
        ascending = st.checkbox("Сортировать по возрастанию", value=True)
        
        # Sort and display
        if sort_by in comparison_df.columns:
            sorted_df = comparison_df.sort_values(by=sort_by, ascending=ascending)
        else:
            # Если выбранного столбца нет, не сортируем или используем первый доступный столбец
            st.warning(f"Столбец '{sort_by}' не найден в данных. Отображение без сортировки.")
            sorted_df = comparison_df
        
        st.dataframe(sorted_df)
        
        # Download button for the dataframe
        csv = sorted_df.to_csv(index=False)
        st.download_button(
            label="Скачать метрики в CSV",
            data=csv,
            file_name="model_metrics.csv",
            mime="text/csv"
        )
    else:
        st.warning("Нет доступных метрик для сравнения моделей.")

def display_model_architecture(all_histories):
    """Display architecture visualization for selected model."""
    st.markdown('<div class="sub-header">Model Architecture Visualization</div>', unsafe_allow_html=True)
    
    try:
        logger.info("Отображение архитектуры модели")
        
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
            logger.info(f"Выбрана модель: {selected_model} из группы {selected_group}")
            
            # Try to find the matching model file in the models directory
            model_found = False
            model_files = []
            model_path = None
            
            # Different model naming patterns for different groups
            # Используем .resolve() для получения полного пути с поддержкой Unicode/кириллицы
            group_path = (MODELS_PATH / selected_group).resolve()
            
            if os.path.exists(group_path):
                logger.info(f"Поиск файла модели в директории: {group_path}")
                try:
                    # Get all model files in the group directory
                    model_files = [f for f in os.listdir(group_path) if f.endswith('.h5')]
                    logger.info(f"Найдено файлов моделей: {len(model_files)}")
                except Exception as e:
                    logger.error(f"Ошибка при чтении файлов из директории {group_path}: {e}")
                    st.error(f"Ошибка при чтении файлов из директории: {e}")
                    model_files = []
                
                # Определяем шаблоны поиска файлов в зависимости от группы
                if selected_group in ['1 old', '2 new']:
                    # For groups 1 and 2, format is typically just "name.h5"
                    if f"{selected_model}.h5" in model_files:
                        model_path = (group_path / f"{selected_model}.h5").resolve()
                        model_found = True
                        logger.info(f"Файл модели найден: {model_path}")
                else:
                    # For groups 3 and 4, format is typically "model_name_timestamp.h5"
                    # First attempt: Direct match with model name
                    if f"{selected_model}.h5" in model_files:
                        model_path = (group_path / f"{selected_model}.h5").resolve()
                        model_found = True
                        logger.info(f"Файл модели найден: {model_path}")
                    # Second attempt: Model name with "model_" prefix
                    elif f"model_{selected_model}.h5" in model_files:
                        model_path = (group_path / f"model_{selected_model}.h5").resolve()
                        model_found = True
                        logger.info(f"Файл модели найден: {model_path}")
                    # If still not found, search for partial matches
                    else:
                        # Partial match based on model name components
                        model_name_parts = selected_model.split('_')
                        for model_file in model_files:
                            if any(part in model_file for part in model_name_parts if len(part) > 2):
                                model_path = (group_path / model_file).resolve()
                                model_found = True
                                logger.info(f"Найден подходящий файл модели: {model_path}")
                                break
                
                if not model_found:
                    logger.warning(f"Не удалось найти файл модели для '{selected_model}' в группе '{selected_group}'")
                    st.warning(f"Не удалось найти файл модели для '{selected_model}' в группе '{selected_group}'")
                    st.write("Доступные файлы моделей в этой группе:")
                    st.write(", ".join(model_files) if model_files else "Нет файлов")
            else:
                logger.warning(f"Директория группы не найдена: {group_path}")
                st.warning(f"Директория группы не найдена: {group_path}")
            
            # Display model info if found
            if model_found and model_path:
                try:
                    logger.info(f"Попытка визуализации модели: {model_path}")
                    # Convert model_path to string with proper encoding for visualization
                    model_path_str = str(model_path)
                    # Try to load and visualize the model
                    visualize_keras_model(model_path_str)
                    
                    # Display model file location
                    st.info(f"Model file: {model_path}")
                    logger.info(f"Модель успешно визуализирована: {model_path}")
                except Exception as e:
                    log_error(e, section="визуализация модели", context=f"файл: {model_path}")
            else:
                st.warning(f"No matching model file found for {selected_model} in {selected_group}.")
                
                if model_files:
                    st.info(f"Available model files in {selected_group}: {', '.join(model_files)}")
                else:
                    st.info(f"No model files found in {selected_group}.")
        else:
            st.info("Select a model group and model to visualize its architecture.")
    except Exception as e:
        log_error(e, section="display_model_architecture")

def display_training_history(all_histories):
    """Display training history for selected models."""
    st.markdown('<div class="sub-header">История обучения</div>', unsafe_allow_html=True)
    
    # Allow selection of multiple models for comparison
    col1, col2 = st.columns(2)
    
    with col1:
        # Select group
        selected_group = st.selectbox(
            "Выберите группу:",
            options=list(all_histories.keys())
        )
    
    with col2:
        # Select metric
        selected_metric = st.selectbox(
            "Выберите метрику:",
            options=STANDARD_METRICS,
            format_func=lambda x: METRIC_DISPLAY_NAMES.get(x, x)
        )
    
    fig = None
    
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
                "Выберите модели для сравнения:",
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
        st.warning(f"Нет данных истории обучения для выбранных моделей и метрики: {selected_metric}")
    
    # Show detailed metrics for each selected model if selected_models exists and is not empty
    if selected_group and 'selected_models' in locals() and selected_models:
        st.markdown('<div class="sub-header">Детали метрик модели</div>', unsafe_allow_html=True)
        
        for model in selected_models:
            history = all_histories[selected_group][model]
            
            # Create expander for each model
            with st.expander(f"Модель: {model}"):
                # Check if history contains the 'history' key with metrics
                if isinstance(history, dict) and 'history' in history and isinstance(history['history'], dict):
                    # Display information about the model
                    if 'model_name' in history:
                        st.write(f"**Название модели:** {history['model_name']}")
                    if 'group_name' in history:
                        st.write(f"**Группа:** {history['group_name']}")
                    
                    # Display final metrics if available
                    if 'metrics' in history and isinstance(history['metrics'], dict):
                        st.write("**Итоговые метрики оценки:**")
                        
                        # Create two columns for metrics display
                        col1, col2 = st.columns(2)
                        
                        # Исключаем mape, r2, norm_rmse, norm_mae и median_absolute_error из отображения
                        metrics_to_exclude = ['mape', 'r2', 'r2_score', 'norm_rmse', 'norm_mae', 'median_absolute_error']
                        metrics = [m for m in sorted(history['metrics'].keys()) if m not in metrics_to_exclude]
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
                    
                    # Display training history for different metrics
                    history_data = history['history']
                    
                    # Get available metrics
                    available_metrics = [m for m in STANDARD_METRICS if m in history_data]
                    
                    if available_metrics:
                        # Create tabs for different metric categories
                        tab1, tab2 = st.tabs(["Метрики ошибок", "Метрики производительности"])
                        
                        # Error metrics tab
                        with tab1:
                            error_metrics = ['rmse', 'mse', 'mae', 'max_error']
                            error_metrics = [m for m in error_metrics if m in available_metrics]
                            
                            if error_metrics:
                                for metric in error_metrics:
                                    # Safely check if metric has data and is not None
                                    if metric in history_data and history_data[metric] is not None:
                                        values = history_data[metric]
                                        
                                        # Ensure values is a list
                                        if not isinstance(values, list):
                                            # Для скалярного значения отображаем только конечное значение
                                            st.metric(
                                                label=f"Итоговое {METRIC_DISPLAY_NAMES.get(metric, metric)}", 
                                                value=f"{values:.4f}"
                                            )
                                            continue
                                        
                                        if len(values) > 0:
                                            epochs = list(range(1, len(values) + 1))
                                            
                                            fig = px.line(
                                                x=epochs,
                                                y=values,
                                                title=f"История {METRIC_DISPLAY_NAMES.get(metric, metric)}",
                                                labels={'x': 'Эпоха', 'y': metric}
                                            )
                                            st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("Для этой модели нет доступных метрик ошибок.")
                        
                        # Performance metrics tab
                        with tab2:
                            perf_metrics = ['explained_variance']
                            perf_metrics = [m for m in perf_metrics if m in available_metrics]
                            
                            if perf_metrics:
                                for metric in perf_metrics:
                                    # Safely check if metric has data and is not None
                                    if metric in history_data and history_data[metric] is not None:
                                        values = history_data[metric]
                                        
                                        # Ensure values is a list
                                        if not isinstance(values, list):
                                            # Для скалярного значения отображаем только конечное значение
                                            st.metric(
                                                label=f"Итоговое {METRIC_DISPLAY_NAMES.get(metric, metric)}", 
                                                value=f"{values:.4f}"
                                            )
                                            continue
                                        
                                        if len(values) > 0:
                                            epochs = list(range(1, len(values) + 1))
                                            
                                            fig = px.line(
                                                x=epochs,
                                                y=values,
                                                title=f"История {METRIC_DISPLAY_NAMES.get(metric, metric)}",
                                                labels={'x': 'Эпоха', 'y': metric}
                                            )
                                            st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("Для этой модели нет доступных метрик производительности.")
                    else:
                        st.info("Для этой модели нет доступных метрик истории обучения.")
                else:
                    st.warning("Формат данных истории не соответствует ожидаемому.")
        else:
            st.warning(f"В группе '{selected_group}' не найдено действительных моделей.")
    else:
        st.info("Выберите группу моделей для просмотра истории обучения.")

def display_model_comparison(all_histories, comparison_df):
    """Display comparison of models across different metrics."""
    st.markdown('<div class="sub-header">Сравнение моделей</div>', unsafe_allow_html=True)
    
    # Create tabs for different visualization types
    tab1, tab2, tab3 = st.tabs(["Сравнение метрик", "Лепестковая диаграмма", "Сравнение групп"])
    
    with tab1:
        # Select metric for comparison
        selected_metric = st.selectbox(
            "Выберите метрику для сравнения:",
            options=[m for m in STANDARD_METRICS if m not in ['mape', 'r2', 'r2_score', 'norm_rmse', 'norm_mae', 'median_absolute_error', 'max_error', 'explained_variance']],
            format_func=lambda x: METRIC_DISPLAY_NAMES.get(x, x)
        )
        
        # Create comparison chart
        fig = plot_metric_comparison(comparison_df, selected_metric)
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            
            # Display explanation about the metric
            metric_explanations = {
                'rmse': "**Корень из среднеквадратичной ошибки (RMSE)** измеряет среднюю величину ошибок прогноза. Меньшие значения указывают на лучшую производительность модели.",
                'explained_variance': "**Объяснённая дисперсия** измеряет долю дисперсии в зависимой переменной, которую можно предсказать на основе независимых переменных.",
                'mae': "**Средняя абсолютная ошибка (MAE)** измеряет среднюю величину ошибок без учета их направления. Меньшие значения лучше.",
                'max_error': "**Максимальная ошибка** показывает максимальную остаточную ошибку, представляющую наихудший случай прогноза. Меньше значение - лучше.",
                'mse': "**Среднеквадратичная ошибка (MSE)** - это среднее значение квадратов разностей между предсказанными и фактическими значениями. Меньшие значения указывают на лучшее соответствие."
            }
            
            if selected_metric in metric_explanations:
                st.markdown(metric_explanations[selected_metric])
        else:
            st.warning(f"Нет данных для метрики: {selected_metric}")
    
    with tab2:
        # Select metrics for radar chart
        selected_metrics = st.multiselect(
            "Выберите метрики для лепестковой диаграммы:",
            options=[m for m in STANDARD_METRICS if m not in ['mape', 'r2', 'r2_score', 'norm_rmse', 'norm_mae', 'median_absolute_error', 'max_error', 'explained_variance']],
            default=[m for m in STANDARD_METRICS[:5] if m not in ['mape', 'r2', 'r2_score', 'norm_rmse', 'norm_mae', 'median_absolute_error', 'max_error', 'explained_variance']],
            format_func=lambda x: METRIC_DISPLAY_NAMES.get(x, x)
        )
        
        # Подготовим список моделей для выбора
        all_models = []
        for group in all_histories:
            for model in all_histories[group]:
                if not (isinstance(all_histories[group][model], dict) and 
                   all_histories[group][model].get('__placeholder__', False)):
                    all_models.append((group, model))
        
        # Формируем варианты для выбора
        model_options = [f"{model} ({group})" for group, model in all_models]
        default_selections = model_options[:min(3, len(model_options))] if model_options else []
        
        # Выбор моделей для сравнения
        selected_models = st.multiselect(
            "Выберите модели для включения (максимум 5):",
            options=model_options,
            default=default_selections
        )
        
        # Limit to max 5 models for readability
        if len(selected_models) > 5:
            st.warning("Для лучшей читаемости на лепестковой диаграмме будут показаны только первые 5 выбранных моделей.")
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
                    lambda row: (row['group'], row['model']) in [(group, model) for group, model in selected_model_tuples], 
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
                    **Примечание о лепестковой диаграмме:**
                    - Метрики нормализованы до шкалы 0-1, где 1 всегда представляет наилучшую производительность.
                    - Для метрик ошибок (RMSE, MAE и т.д.), меньшие исходные значения лучше, поэтому шкала инвертируется.
                    - Для метрик производительности (объяснённая дисперсия), большие значения лучше.
                    """)
                else:
                    st.warning("Не удалось создать лепестковую диаграмму с выбранными данными.")
            else:
                st.warning("Нет данных для выбранных моделей и метрик.")
        else:
            st.info("Выберите хотя бы одну метрику и одну модель для создания лепестковой диаграммы.")
    
    with tab3:
        # Group comparison
        st.subheader("Сравнение групп моделей")
        
        # Calculate group averages
        if 'group' in comparison_df.columns:
            # Create group metrics for each standardized metric
            group_metrics = []
            
            for group in comparison_df['group'].unique():
                group_data = {'group': group}  # Use lowercase 'group' for consistency
                
                # Filter dataframe for this group
                group_df = comparison_df[comparison_df['group'] == group]
                
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
                st.write("**Количество моделей по группам:**")
                
                # Create bar chart for model counts
                fig = px.bar(
                    group_df,
                    x='group',  # Use lowercase column name
                    y='model_count',
                    title='Модели по группам',
                    labels={'model_count': 'Количество моделей', 'group': 'Группа'},
                    color='group'  # Use lowercase column name
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Select metric for group comparison
                selected_metric = st.selectbox(
                    "Выберите метрику для сравнения групп:",
                    options=[m for m in STANDARD_METRICS if m not in ['mape', 'r2', 'r2_score', 'norm_rmse', 'norm_mae', 'median_absolute_error', 'max_error', 'explained_variance']],
                    format_func=lambda x: METRIC_DISPLAY_NAMES.get(x, x),
                    key="group_comparison_metric"
                )
                
                if selected_metric:
                    # Create tabs for different comparison types
                    tab_avg, tab_best = st.tabs(["Средняя производительность", "Лучшая производительность"])
                    
                    with tab_avg:
                        # Average performance
                        avg_col = f'avg_{selected_metric}'
                        
                        if avg_col in group_df.columns:
                            # Create bar chart
                            fig = px.bar(
                                group_df,
                                x='group',  # Use lowercase column name
                                y=avg_col,
                                title=f'Среднее {METRIC_DISPLAY_NAMES.get(selected_metric, selected_metric)} по группам',
                                labels={avg_col: f'Среднее {selected_metric}', 'group': 'Группа'},
                                color='group'  # Use lowercase column name
                            )
                            
                            # Adjust y-axis (lower bound for better metrics)
                            if not HIGHER_IS_BETTER.get(selected_metric, False):
                                fig.update_layout(yaxis_range=[0, group_df[avg_col].max() * 1.1])
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning(f"Нет данных для средней метрики: {selected_metric}")
                    
                    with tab_best:
                        # Best performance
                        best_col = f'best_{selected_metric}'
                        
                        if best_col in group_df.columns:
                            # Create bar chart
                            fig = px.bar(
                                group_df,
                                x='group',  # Use lowercase column name
                                y=best_col,
                                title=f'Лучшее {METRIC_DISPLAY_NAMES.get(selected_metric, selected_metric)} по группам',
                                labels={best_col: f'Лучшее {selected_metric}', 'group': 'Группа'},
                                color='group'  # Use lowercase column name
                            )
                            
                            # Adjust y-axis (lower bound for better metrics)
                            if not HIGHER_IS_BETTER.get(selected_metric, False):
                                fig.update_layout(yaxis_range=[0, group_df[best_col].max() * 1.1])
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning(f"Нет данных для лучшей метрики: {selected_metric}")
                
                if not selected_metric:
                    st.warning("Не выбрана метрика для сравнения групп.")
                else:
                    st.warning("Нет данных для сравнения групп.")
            else:
                st.warning("Информация о группах недоступна в данных метрик.")

def display_best_models(all_histories):
    """Display the best performing models based on different metrics."""
    st.markdown('<div class="sub-header">Лучшие модели</div>', unsafe_allow_html=True)
    
    # Select metric for determining best models
    selected_metric = st.selectbox(
        "Выберите метрику для ранжирования:",
        options=[m for m in STANDARD_METRICS if m not in ['mape', 'r2', 'r2_score', 'norm_rmse', 'norm_mae', 'median_absolute_error', 'max_error', 'explained_variance']],
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
                'group': group,
                'model': info['model_name'],
                selected_metric: info['value']
            }
            for group, info in best_models.items()
        ])
        
        # Sort based on metric (ascending or descending based on is_higher_better)
        best_df = best_df.sort_values(by=selected_metric, ascending=not is_higher_better)
        
        # Display as table
        st.write(f"**Лучшие модели по {METRIC_DISPLAY_NAMES.get(selected_metric, selected_metric)}:**")
        st.dataframe(best_df)
        
        # Create bar chart
        fig = px.bar(
            best_df,
            x='group',
            y=selected_metric,
            color='model',
            title=f'Лучшие модели по {METRIC_DISPLAY_NAMES.get(selected_metric, selected_metric)}',
            barmode='group',
            text='model'
        )
        
        # Adjust label positions
        fig.update_traces(textposition='outside')
        
        # Adjust y-axis (lower bound for better metrics)
        if not is_higher_better:
            fig.update_layout(yaxis_range=[0, best_df[selected_metric].max() * 1.1])
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display information about each best model
        st.markdown('<div class="sub-header">Подробная информация</div>', unsafe_allow_html=True)
        
        for group, info in best_models.items():
            model_name = info['model_name']
            
            with st.expander(f"Лучшая модель в группе {group}: {model_name}"):
                if group in all_histories and model_name in all_histories[group]:
                    history = all_histories[group][model_name]
                    
                    # Check if history contains the necessary information
                    if isinstance(history, dict):
                        if 'model_name' in history:
                            st.write(f"**Название модели:** {history['model_name']}")
                        if 'group_name' in history:
                            st.write(f"**Группа:** {history['group_name']}")
                        
                        # Display final metrics if available
                        if 'metrics' in history and isinstance(history['metrics'], dict):
                            st.write("**Итоговые метрики оценки:**")
                            
                            # Create two columns for metrics display
                            col1, col2 = st.columns(2)
                            
                            # Исключаем mape, r2, norm_rmse, norm_mae и median_absolute_error из отображения
                            metrics_to_exclude = ['mape', 'r2', 'r2_score', 'norm_rmse', 'norm_mae', 'median_absolute_error']
                            metrics = [m for m in sorted(history['metrics'].keys()) if m not in metrics_to_exclude]
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
                            if selected_metric in history['history'] and history['history'][selected_metric] is not None:
                                # Проверяем, является ли значение списком или скаляром
                                metric_values = history['history'][selected_metric]
                                if isinstance(metric_values, list) and len(metric_values) > 0:
                                    # Для списка значений (история по эпохам)
                                    st.write(f"**История обучения для {METRIC_DISPLAY_NAMES.get(selected_metric, selected_metric)}:**")
                                    epochs = list(range(1, len(metric_values) + 1))
                                    
                                    # Создаем график внутри блока, где определена переменная epochs
                                    fig = px.line(
                                        x=epochs,
                                        y=metric_values,
                                        title=f"История {METRIC_DISPLAY_NAMES.get(selected_metric, selected_metric)}",
                                        labels={'x': 'Эпоха', 'y': selected_metric}
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.warning("История метрики не содержит данных для построения графика.")
                            else:
                                st.warning("Данные истории модели не в ожидаемом формате.")
                        else:
                            st.warning(f"Информация о модели не найдена для {model_name} в группе {group}.")
                    else:
                        st.warning(f"Информация о модели не найдена для {model_name} в группе {group}.")
                else:
                    st.warning("Лучшие модели для выбранной метрики не найдены.")
    else:
        st.warning("Не удалось найти лучшие модели для выбранной метрики.")

if __name__ == "__main__":
    main() 
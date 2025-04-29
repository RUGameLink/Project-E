import os
import json
import shutil
import uuid
import tempfile
from pathlib import Path
import traceback
import datetime
import logging
import configparser
from logging.handlers import RotatingFileHandler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
import uvicorn
from pydantic import BaseModel, Field
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error, explained_variance_score

# Настройка логирования
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

log_file = log_dir / f"evaluator_{datetime.datetime.now().strftime('%Y%m%d')}.log"
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

file_handler = RotatingFileHandler(
    log_file, 
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5,
    encoding='utf-8'
)
file_handler.setFormatter(log_formatter)

logger = logging.getLogger("evaluator")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

# Загрузка конфигурации
config = configparser.ConfigParser()
config_path = Path(__file__).parent / "config.ini"
if os.path.exists(config_path):
    # Используем кодировку UTF-8 для поддержки кириллицы
    config.read(config_path, encoding='utf-8')
else:
    # Создаем конфигурацию по умолчанию если файл не существует
    config["Paths"] = {"models_directory": "../model_save_preset"}
    config["Data"] = {"test_data_path": "test_data.csv"}
    config["Logging"] = {"log_level": "INFO"}
    # Сохраняем конфигурацию по умолчанию с кодировкой UTF-8
    with open(config_path, 'w', encoding='utf-8') as f:
        config.write(f)
    logger.info(f"Создан файл конфигурации по умолчанию: {config_path}")

# Настройка путей к моделям и данным
BASE_PATH = Path(config["Paths"]["models_directory"])
MODELS_PATH = BASE_PATH / "models"
HISTORY_PATH = BASE_PATH / "history"
TEMP_PATH = Path("temp_uploads")
TEMP_PATH.mkdir(exist_ok=True)

# Создание папок для моделей и истории, если они не существуют
MODELS_PATH.mkdir(exist_ok=True, parents=True)
HISTORY_PATH.mkdir(exist_ok=True, parents=True)

# Метрики для оценки моделей
METRICS = {
    'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
    'mse': mean_squared_error,
    'mae': mean_absolute_error,
    'max_error': max_error,
    'explained_variance': explained_variance_score,
}

# Определение моделей данных через Pydantic
class ModelGroup(BaseModel):
    name: str
    models_count: int
    history_count: int

class ModelEvalResult(BaseModel):
    model_name: str
    group_name: str
    metrics: Dict[str, float]
    
class EvaluationRequest(BaseModel):
    model_path: str
    test_data_path: str
    group_name: str
    model_name: Optional[str] = None

# Инициализация FastAPI
app = FastAPI(
    title="Модуль оценки моделей нейронных сетей",
    description="API для оценки моделей нейронных сетей и управления группами моделей",
    version="1.0.0"
)

# Добавление CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Монтирование статических файлов
app.mount("/static", StaticFiles(directory="static"), name="static")

# Настройка шаблонов Jinja2
templates = Jinja2Templates(directory="templates")

# Функция для проверки существования пути
def ensure_dir_exists(dir_path: Path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logger.info(f"Создана директория: {dir_path}")

# Функция получения списка групп моделей
def get_model_groups():
    ensure_dir_exists(MODELS_PATH)
    ensure_dir_exists(HISTORY_PATH)
    
    groups = []
    
    model_dirs = {d.name: d for d in MODELS_PATH.iterdir() if d.is_dir()}
    history_dirs = {d.name: d for d in HISTORY_PATH.iterdir() if d.is_dir()}
    
    all_groups = set(list(model_dirs.keys()) + list(history_dirs.keys()))
    
    for group_name in sorted(all_groups):
        model_count = 0
        if group_name in model_dirs:
            model_files = [f for f in model_dirs[group_name].glob("*.h5")]
            model_count = len(model_files)
        
        history_count = 0
        if group_name in history_dirs:
            history_files = [f for f in history_dirs[group_name].glob("*.json")]
            history_count = len(history_files)
        
        groups.append(ModelGroup(
            name=group_name,
            models_count=model_count,
            history_count=history_count
        ))
    
    return groups

# Функция оценки модели
def evaluate_model(model_path: Path, test_data_path: Path) -> Dict[str, float]:
    logger.info(f"Оценка модели: {model_path}")
    try:
        # Загрузка модели
        model = load_model(model_path)
        
        # Загрузка тестовых данных
        test_data = pd.read_csv(test_data_path)
        
        # Предполагаем, что последний столбец - целевая переменная
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values
        
        # Выполнение предсказания
        y_pred = model.predict(X_test).flatten()
        
        # Расчет метрик
        results = {}
        for metric_name, metric_func in METRICS.items():
            results[metric_name] = float(metric_func(y_test, y_pred))
        
        return results
    except Exception as e:
        logger.error(f"Ошибка при оценке модели {model_path}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Ошибка при оценке модели: {e}")

# Функция сохранения результатов оценки
def save_evaluation_results(model_name: str, group_name: str, metrics: Dict[str, float]):
    logger.info(f"Сохранение результатов оценки для модели {model_name} в группе {group_name}")
    
    # Создание директории группы, если она не существует
    history_group_path = HISTORY_PATH / group_name
    ensure_dir_exists(history_group_path)
    
    # Формируем данные для сохранения
    history_data = {
        "model_name": model_name,
        "group_name": group_name,
        "metrics": metrics,
        "history": {metric: [value] for metric, value in metrics.items()},
        "evaluation_date": datetime.datetime.now().isoformat()
    }
    
    # Имя файла для истории
    history_file = f"model_save_preset_history_{group_name}_{model_name}.json"
    history_path = history_group_path / history_file
    
    # Сохраняем историю в формате JSON
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Результаты оценки сохранены в {history_path}")
    return str(history_path)

# Функция для копирования модели в структуру хранения
def copy_model_to_storage(temp_model_path: Path, group_name: str, model_name: str) -> Path:
    logger.info(f"Копирование модели в хранилище: {temp_model_path} -> группа {group_name}, модель {model_name}")
    
    # Создание директории группы, если она не существует
    models_group_path = MODELS_PATH / group_name
    ensure_dir_exists(models_group_path)
    
    # Целевой путь для сохранения модели
    target_path = models_group_path / f"{model_name}.h5"
    
    # Копирование файла модели
    shutil.copy2(temp_model_path, target_path)
    logger.info(f"Модель скопирована в {target_path}")
    
    return target_path

# Роуты API
@app.get("/", response_class=HTMLResponse)
async def get_index():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Модуль оценки моделей</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { padding: 20px; }
            .container { max-width: 800px; margin: 0 auto; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="mb-4">Модуль оценки моделей нейронных сетей</h1>
            
            <div class="alert alert-success">
                <strong>Статус:</strong> Система готова к работе
            </div>
            
            <div class="card mb-4">
                <div class="card-header">Загрузка и оценка модели</div>
                <div class="card-body">
                    <form id="uploadForm" enctype="multipart/form-data">
                        <div class="mb-3">
                            <label for="model" class="form-label">Модель (.h5)</label>
                            <input type="file" class="form-control" id="model" name="model" accept=".h5" required>
                        </div>
                        <div class="mb-3">
                            <label for="testData" class="form-label">Тестовые данные (.csv)</label>
                            <input type="file" class="form-control" id="testData" name="testData" accept=".csv" required>
                        </div>
                        <div class="mb-3">
                            <label for="groupName" class="form-label">Группа</label>
                            <input type="text" class="form-control" id="groupName" name="groupName" required>
                        </div>
                        <div class="mb-3">
                            <label for="modelName" class="form-label">Название модели</label>
                            <input type="text" class="form-control" id="modelName" name="modelName" placeholder="По умолчанию генерируется автоматически">
                        </div>
                        <button type="submit" class="btn btn-primary">Оценить модель</button>
                    </form>
                </div>
            </div>
            
            <div class="card">
                <div class="card-header">Группы моделей</div>
                <div class="card-body">
                    <div id="groupsList">
                        <p>Загрузка групп...</p>
                    </div>
                </div>
            </div>
            
            <div id="resultContainer" class="mt-4" style="display: none;">
                <div class="card">
                    <div class="card-header">Результаты оценки</div>
                    <div class="card-body">
                        <div id="evaluationResults"></div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            // Загрузка списка групп при загрузке страницы
            document.addEventListener('DOMContentLoaded', function() {
                fetchGroups();
                
                // Обработчик формы загрузки
                document.getElementById('uploadForm').addEventListener('submit', function(e) {
                    e.preventDefault();
                    uploadAndEvaluate();
                });
            });
            
            // Получение списка групп
            function fetchGroups() {
                fetch('/api/groups')
                    .then(response => response.json())
                    .then(data => {
                        const groupsList = document.getElementById('groupsList');
                        if (data.length === 0) {
                            groupsList.innerHTML = '<p>Нет доступных групп</p>';
                            return;
                        }
                        
                        let html = '<table class="table table-striped">';
                        html += '<thead><tr><th>Группа</th><th>Моделей</th><th>Файлов истории</th><th>Действия</th></tr></thead>';
                        html += '<tbody>';
                        
                        data.forEach(group => {
                            html += `<tr>
                                <td>${group.name}</td>
                                <td>${group.models_count}</td>
                                <td>${group.history_count}</td>
                                <td>
                                    <button class="btn btn-sm btn-danger" onclick="deleteGroup('${group.name}')">Удалить</button>
                                </td>
                            </tr>`;
                        });
                        
                        html += '</tbody></table>';
                        groupsList.innerHTML = html;
                    })
                    .catch(error => {
                        console.error('Ошибка при получении групп:', error);
                        document.getElementById('groupsList').innerHTML = '<p class="text-danger">Ошибка при загрузке групп</p>';
                    });
            }
            
            // Загрузка и оценка модели
            function uploadAndEvaluate() {
                const form = document.getElementById('uploadForm');
                const formData = new FormData(form);
                
                // Отображение индикатора загрузки
                const submitBtn = form.querySelector('button[type="submit"]');
                const originalBtnText = submitBtn.textContent;
                submitBtn.disabled = true;
                submitBtn.textContent = 'Загрузка и оценка...';
                
                fetch('/api/evaluate', {
                    method: 'POST',
                    body: formData
                })
                .then(response => {
                    if (!response.ok) {
                        return response.json().then(err => {
                            throw new Error(err.detail || 'Произошла ошибка при оценке модели');
                        });
                    }
                    return response.json();
                })
                .then(data => {
                    // Отображение результатов
                    const resultContainer = document.getElementById('resultContainer');
                    const resultsDiv = document.getElementById('evaluationResults');
                    
                    resultContainer.style.display = 'block';
                    
                    let html = `<h5>Модель: ${data.model_name} (Группа: ${data.group_name})</h5>`;
                    html += '<table class="table table-bordered">';
                    html += '<thead><tr><th>Метрика</th><th>Значение</th></tr></thead>';
                    html += '<tbody>';
                    
                    for (const [metric, value] of Object.entries(data.metrics)) {
                        let metricDisplay = metric;
                        if (metric === 'rmse') metricDisplay = 'Корень из среднеквадратичной ошибки (RMSE)';
                        if (metric === 'explained_variance') metricDisplay = 'Объяснённая дисперсия';
                        if (metric === 'mae') metricDisplay = 'Средняя абсолютная ошибка (MAE)';
                        if (metric === 'max_error') metricDisplay = 'Максимальная ошибка';
                        if (metric === 'mse') metricDisplay = 'Среднеквадратичная ошибка (MSE)';
                        
                        html += `<tr><td>${metricDisplay}</td><td>${value.toFixed(4)}</td></tr>`;
                    }
                    
                    html += '</tbody></table>';
                    html += '<p class="text-success">Модель успешно оценена и сохранена</p>';
                    
                    resultsDiv.innerHTML = html;
                    
                    // Обновление списка групп
                    fetchGroups();
                })
                .catch(error => {
                    console.error('Ошибка:', error);
                    const resultContainer = document.getElementById('resultContainer');
                    const resultsDiv = document.getElementById('evaluationResults');
                    
                    resultContainer.style.display = 'block';
                    resultsDiv.innerHTML = `<div class="alert alert-danger">${error.message || 'Произошла ошибка при оценке модели'}</div>`;
                })
                .finally(() => {
                    // Восстановление кнопки
                    submitBtn.disabled = false;
                    submitBtn.textContent = originalBtnText;
                });
            }
            
            // Удаление группы
            function deleteGroup(groupName) {
                if (!confirm(`Вы уверены, что хотите удалить группу "${groupName}" и все её модели?`)) {
                    return;
                }
                
                fetch(`/api/groups/${encodeURIComponent(groupName)}`, {
                    method: 'DELETE'
                })
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Не удалось удалить группу');
                    }
                    return response.json();
                })
                .then(data => {
                    alert(`Группа "${groupName}" успешно удалена`);
                    fetchGroups();
                })
                .catch(error => {
                    console.error('Ошибка при удалении группы:', error);
                    alert('Произошла ошибка при удалении группы');
                });
            }
        </script>
        
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """
    
    return html

@app.get("/api/groups", response_model=List[ModelGroup])
async def api_get_groups():
    return get_model_groups()

@app.delete("/api/groups/{group_name}")
async def api_delete_group(group_name: str):
    # Удаление группы моделей и истории
    logger.info(f"Удаление группы: {group_name}")
    
    models_group_path = MODELS_PATH / group_name
    history_group_path = HISTORY_PATH / group_name
    
    deleted_files = 0
    
    if os.path.exists(models_group_path):
        try:
            # Удаление всех файлов в директории группы моделей
            for file_path in models_group_path.glob("*"):
                file_path.unlink()
                deleted_files += 1
            
            # Удаление директории группы моделей
            models_group_path.rmdir()
            logger.info(f"Удалена директория моделей группы: {models_group_path}")
        except Exception as e:
            logger.error(f"Ошибка при удалении директории моделей группы {group_name}: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Ошибка при удалении группы моделей: {e}")
    
    if os.path.exists(history_group_path):
        try:
            # Удаление всех файлов в директории группы истории
            for file_path in history_group_path.glob("*"):
                file_path.unlink()
                deleted_files += 1
            
            # Удаление директории группы истории
            history_group_path.rmdir()
            logger.info(f"Удалена директория истории группы: {history_group_path}")
        except Exception as e:
            logger.error(f"Ошибка при удалении директории истории группы {group_name}: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Ошибка при удалении группы истории: {e}")
    
    return {"status": "success", "message": f"Группа {group_name} успешно удалена", "deleted_files": deleted_files}

@app.post("/api/evaluate", response_model=ModelEvalResult)
async def api_evaluate_model(
    model: UploadFile = File(...),
    testData: UploadFile = File(...),
    groupName: str = Form(...),
    modelName: Optional[str] = Form(None)
):        
    logger.info(f"Получен запрос на оценку модели для группы: {groupName}")
    
    # Создаем временные файлы для загрузки
    temp_model_file = TEMP_PATH / f"{uuid.uuid4()}.h5"
    temp_data_file = TEMP_PATH / f"{uuid.uuid4()}.csv"
    
    try:
        # Генерируем имя модели если не указано
        if not modelName:
            now = datetime.datetime.now()
            modelName = f"model_{now.strftime('%Y%m%d_%H%M%S')}"
        
        # Сохраняем загруженные файлы
        with open(temp_model_file, "wb") as buffer:
            shutil.copyfileobj(model.file, buffer)
        
        with open(temp_data_file, "wb") as buffer:
            shutil.copyfileobj(testData.file, buffer)
        
        # Оцениваем модель
        metrics = evaluate_model(temp_model_file, temp_data_file)
        
        # Копируем модель в хранилище
        copy_model_to_storage(temp_model_file, groupName, modelName)
        
        # Сохраняем результаты оценки
        save_evaluation_results(modelName, groupName, metrics)
        
        return ModelEvalResult(
            model_name=modelName,
            group_name=groupName,
            metrics=metrics
        )
    
    except Exception as e:
        logger.error(f"Ошибка при оценке модели: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Удаляем временные файлы
        if os.path.exists(temp_model_file):
            os.unlink(temp_model_file)
        if os.path.exists(temp_data_file):
            os.unlink(temp_data_file)

def run_server():
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    run_server() 
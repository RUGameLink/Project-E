import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict, Any, Tuple, List, Optional, Union
from sklearn.base import BaseEstimator, RegressorMixin
from pandas import DataFrame, Series

def load_data(data_path: str) -> Tuple[DataFrame, Series]:
    """
    Загружает набор данных и разделяет на признаки и целевую переменную
    
    Args:
        data_path: путь к файлу данных (csv, xlsx)
        
    Returns:
        кортеж из X (признаки) и y (целевая переменная)
    """
    file_extension = os.path.splitext(data_path)[1].lower()
    
    if file_extension == '.csv':
        df = pd.read_csv(data_path)
    elif file_extension in ['.xlsx', '.xls']:
        df = pd.read_excel(data_path)
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {file_extension}")
    
    # Предполагаем, что целевая переменная находится в колонке "target"
    # Если имя целевой колонки отличается, нужно изменить здесь
    target_column = "target"
    
    if target_column not in df.columns:
        raise ValueError(f"Колонка {target_column} не найдена в данных")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    return X, y

class ModelManager:
    """
    Класс для управления моделями машинного обучения:
    обучение, сохранение, загрузка и оценка
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Инициализация менеджера моделей
        
        Args:
            models_dir: директория для хранения моделей
        """
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
    
    def _save_model(self, model: BaseEstimator, model_name: str) -> str:
        """
        Сохраняет модель в файл
        
        Args:
            model: обученная модель
            model_name: имя модели
            
        Returns:
            путь к сохраненной модели
        """
        model_path = os.path.join(self.models_dir, f"{model_name}.joblib")
        joblib.dump(model, model_path)
        return model_path
    
    def load_model(self, model_name: str) -> Union[LinearRegression, RandomForestRegressor, GradientBoostingRegressor]:
        """
        Загружает модель из файла
        
        Args:
            model_name: имя модели
            
        Returns:
            загруженная модель
        """
        model_path = os.path.join(self.models_dir, f"{model_name}.joblib")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель {model_name} не найдена в {model_path}")
        
        return joblib.load(model_path)
    
    def list_available_models(self) -> List[str]:
        """
        Возвращает список доступных моделей
        
        Returns:
            список имен моделей
        """
        models = []
        for filename in os.listdir(self.models_dir):
            if filename.endswith('.joblib'):
                models.append(os.path.splitext(filename)[0])
        return models
    
    def train_linear_regression(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> LinearRegression:
        """
        Обучает и сохраняет модель линейной регрессии
        
        Args:
            X: признаки
            y: целевая переменная
            **kwargs: параметры модели
            
        Returns:
            обученная модель
        """
        model = LinearRegression(**kwargs)
        model.fit(X, y)
        self._save_model(model, "linear_regression")
        return model
    
    def train_random_forest(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> RandomForestRegressor:
        """
        Обучает и сохраняет модель случайного леса
        
        Args:
            X: признаки
            y: целевая переменная
            **kwargs: параметры модели
            
        Returns:
            обученная модель
        """
        model = RandomForestRegressor(**kwargs)
        model.fit(X, y)
        self._save_model(model, "random_forest")
        return model
    
    def train_gradient_boosting(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> GradientBoostingRegressor:
        """
        Обучает и сохраняет модель градиентного бустинга
        
        Args:
            X: признаки
            y: целевая переменная
            **kwargs: параметры модели
            
        Returns:
            обученная модель
        """
        model = GradientBoostingRegressor(**kwargs)
        model.fit(X, y)
        self._save_model(model, "gradient_boosting")
        return model
    
    def evaluate_model(self, model_name: str, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Оценивает производительность модели на данных
        
        Args:
            model_name: имя модели
            X: признаки
            y: целевая переменная
            
        Returns:
            словарь с метриками
        """
        model = self.load_model(model_name)
        y_pred = model.predict(X)
        
        metrics = {
            "mse": mean_squared_error(y, y_pred),
            "rmse": np.sqrt(mean_squared_error(y, y_pred)),
            "mae": mean_absolute_error(y, y_pred),
            "r2": r2_score(y, y_pred)
        }
        
        return metrics
    
    def predict(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """
        Делает предсказания с помощью модели
        
        Args:
            model_name: имя модели
            X: признаки
            
        Returns:
            массив предсказаний
        """
        model = self.load_model(model_name)
        return model.predict(X)

    def train_all_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict[str, Any]]:
        """
        Обучает все доступные модели
        
        Args:
            X: признаки
            y: целевая переменная
            
        Returns:
            словарь с результатами обучения
        """
        results = {}
        
        # Линейная регрессия
        try:
            lr_model = self.train_linear_regression(X, y)
            lr_metrics = self.evaluate_model("linear_regression", X, y)
            results["linear_regression"] = {"success": True, "metrics": lr_metrics}
        except Exception as e:
            results["linear_regression"] = {"success": False, "error": str(e)}
        
        # Случайный лес
        try:
            rf_model = self.train_random_forest(X, y)
            rf_metrics = self.evaluate_model("random_forest", X, y)
            results["random_forest"] = {"success": True, "metrics": rf_metrics}
        except Exception as e:
            results["random_forest"] = {"success": False, "error": str(e)}
        
        # Градиентный бустинг
        try:
            gb_model = self.train_gradient_boosting(X, y)
            gb_metrics = self.evaluate_model("gradient_boosting", X, y)
            results["gradient_boosting"] = {"success": True, "metrics": gb_metrics}
        except Exception as e:
            results["gradient_boosting"] = {"success": False, "error": str(e)}
        
        return results 
import pandas as pd
import numpy as np
import os
import json
import argparse
from pathlib import Path
from models import ModelManager, load_data
from typing import Dict, Any, List, Optional

def evaluate_models(test_data_path: str, 
                   models_dir: str = "models", 
                   output_dir: str = "results") -> Dict[str, Any]:
    """
    Оценка всех моделей на тестовых данных
    
    Args:
        test_data_path: путь к тестовому набору данных
        models_dir: директория с сохраненными моделями
        output_dir: директория для сохранения результатов
        
    Returns:
        словарь с результатами оценки для каждой модели
    """
    # Создаем директорию для результатов если ее нет
    os.makedirs(output_dir, exist_ok=True)
    
    # Загружаем тестовые данные
    X_test, y_test = load_data(test_data_path)
    
    # Инициализируем менеджер моделей
    model_manager = ModelManager(models_dir=models_dir)
    
    # Список доступных моделей
    model_names = ["linear_regression", "random_forest", "gradient_boosting"]
    
    # Словарь для хранения результатов
    results = {}
    
    # Оцениваем каждую модель
    for model_name in model_names:
        try:
            evaluation = model_manager.evaluate_model(model_name, X_test, y_test)
            
            # Сохраняем предсказания в CSV
            predictions_df = pd.DataFrame({
                "true": y_test.values,
                "predicted": evaluation["predictions"]
            })
            predictions_path = os.path.join(output_dir, f"{model_name}_predictions.csv")
            predictions_df.to_csv(predictions_path, index=False)
            
            # Сохраняем метрики в JSON
            metrics_path = os.path.join(output_dir, f"{model_name}_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(evaluation["metrics"], f, indent=4)
            
            # Добавляем результаты в общий словарь
            results[model_name] = {
                "metrics": evaluation["metrics"],
                "predictions_path": predictions_path,
                "metrics_path": metrics_path
            }
            
            print(f"Модель {model_name} успешно оценена.")
            print(f"Метрики: {evaluation['metrics']}")
            print("=" * 50)
            
        except Exception as e:
            print(f"Ошибка при оценке модели {model_name}: {str(e)}")
            results[model_name] = {"error": str(e)}
    
    # Сохраняем общие результаты
    summary_path = os.path.join(output_dir, "summary.json")
    summary = {
        "models_evaluated": len(results),
        "best_model": find_best_model(results),
        "all_metrics": {name: info.get("metrics", {}) for name, info in results.items() if "metrics" in info}
    }
    
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    
    return results

def find_best_model(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Находит лучшую модель на основе метрик
    
    Args:
        results: словарь с результатами оценки моделей
        
    Returns:
        информация о лучшей модели
    """
    best_model = None
    best_r2 = -float("inf")
    
    for model_name, info in results.items():
        if "metrics" in info and "r2" in info["metrics"]:
            r2 = info["metrics"]["r2"]
            if r2 > best_r2:
                best_r2 = r2
                best_model = {
                    "name": model_name,
                    "r2": r2,
                    "metrics": info["metrics"]
                }
    
    return best_model

def main():
    """Основная функция для запуска оценки моделей из командной строки"""
    parser = argparse.ArgumentParser(description="Оценка моделей на тестовых данных")
    parser.add_argument("--test_data", type=str, required=True, 
                        help="Путь к тестовому набору данных")
    parser.add_argument("--models_dir", type=str, default="models",
                        help="Директория с сохраненными моделями")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Директория для сохранения результатов")
    
    args = parser.parse_args()
    
    results = evaluate_models(
        test_data_path=args.test_data,
        models_dir=args.models_dir,
        output_dir=args.output_dir
    )
    
    # Выводим информацию о лучшей модели
    best_model = find_best_model(results)
    if best_model:
        print("\nЛучшая модель:")
        print(f"Название: {best_model['name']}")
        print(f"R-квадрат: {best_model['r2']}")
        print(f"Все метрики: {best_model['metrics']}")
    else:
        print("\nНе удалось определить лучшую модель.")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python
"""
Скрипт для настройки конфигурации визуализатора моделей.
Позволяет настроить пути к директориям с моделями и другие параметры.
"""

import configparser
import os
import sys
from pathlib import Path
import argparse


def main():
    """
    Основная функция для настройки конфигурации.
    """
    parser = argparse.ArgumentParser(description="Настройка конфигурации визуализатора моделей")
    parser.add_argument("--models-dir", "-m", help="Путь к директории с моделями",
                      default="../model_save_preset")
    parser.add_argument("--log-level", "-l", help="Уровень логирования",
                      choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                      default="INFO")
    
    args = parser.parse_args()
    
    # Получаем путь к config.ini в той же директории, где находится скрипт
    script_dir = Path(__file__).parent
    config_path = script_dir / "config.ini"
    
    # Создаем или загружаем конфигурацию
    config = configparser.ConfigParser()
    
    if os.path.exists(config_path):
        print(f"Загрузка существующего файла конфигурации: {config_path}")
        # Используем кодировку UTF-8 для поддержки кириллицы
        config.read(config_path, encoding='utf-8')
    else:
        print(f"Создание нового файла конфигурации: {config_path}")
        config["Paths"] = {}
        config["Logging"] = {}
    
    # Проверяем существует ли директория моделей
    models_dir = Path(args.models_dir)
    if not os.path.exists(models_dir):
        print(f"Предупреждение: Директория {models_dir} не существует.")
        user_input = input("Хотите создать эту директорию? (y/n): ")
        if user_input.lower() == 'y':
            try:
                os.makedirs(models_dir, exist_ok=True)
                os.makedirs(models_dir / "models", exist_ok=True)
                os.makedirs(models_dir / "history", exist_ok=True)
                print(f"Созданы директории: {models_dir}, {models_dir}/models, {models_dir}/history")
            except Exception as e:
                print(f"Ошибка при создании директории: {e}")
                return
        else:
            print("Директория не создана. Укажите существующий путь.")
            return
    
    # Проверяем структуру директории
    if not os.path.exists(models_dir / "models") or not os.path.exists(models_dir / "history"):
        print(f"Предупреждение: В директории {models_dir} отсутствуют необходимые поддиректории 'models' и/или 'history'.")
        user_input = input("Хотите создать недостающие директории? (y/n): ")
        if user_input.lower() == 'y':
            try:
                os.makedirs(models_dir / "models", exist_ok=True)
                os.makedirs(models_dir / "history", exist_ok=True)
                print(f"Созданы необходимые поддиректории")
            except Exception as e:
                print(f"Ошибка при создании директорий: {e}")
                return
        else:
            print("Директории не созданы. Конфигурация может работать некорректно.")
    
    # Обновляем конфигурацию
    config["Paths"]["models_directory"] = str(models_dir)
    config["Logging"]["log_level"] = args.log_level
    
    # Сохраняем конфигурацию
    try:
        # Используем кодировку UTF-8 для поддержки кириллицы при записи
        with open(config_path, 'w', encoding='utf-8') as f:
            config.write(f)
        print(f"Конфигурация сохранена в {config_path}")
        print(f"Настройки:")
        print(f"  Директория моделей: {config['Paths']['models_directory']}")
        print(f"  Уровень логирования: {config['Logging']['log_level']}")
    except Exception as e:
        print(f"Ошибка при сохранении конфигурации: {e}")


if __name__ == "__main__":
    main() 
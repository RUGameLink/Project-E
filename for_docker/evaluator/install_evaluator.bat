@echo off
chcp 65001 > nul
echo =======================================================
echo Установка модуля оценки моделей нейронных сетей
echo =======================================================
cd /d "%~dp0"

echo Начинаем установку...

:: Проверка наличия Python 3.11
set PYTHON_PATH=%~dp0python-3.11.0-amd64.exe
if exist "%PYTHON_PATH%" (
    echo Найден установщик Python 3.11 в текущей директории.
    echo Если Python 3.11 еще не установлен, установите его запустив файл "%PYTHON_PATH%"
    echo.
)

:: Пытаемся найти Python 3.11
echo Поиск Python 3.11...
set PYTHON_CMD=python
set PYTHON311_PATH=

:: Проверяем стандартные пути установки Python 3.11
if exist "C:\Python311\python.exe" (
    echo Найден Python 3.11 в C:\Python311
    set PYTHON311_PATH=C:\Python311\python.exe
) else if exist "C:\Program Files\Python311\python.exe" (
    echo Найден Python 3.11 в C:\Program Files\Python311
    set PYTHON311_PATH="C:\Program Files\Python311\python.exe"
) else if exist "C:\Program Files (x86)\Python311\python.exe" (
    echo Найден Python 3.11 в C:\Program Files (x86)\Python311
    set PYTHON311_PATH="C:\Program Files (x86)\Python311\python.exe"
) else if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    echo Найден Python 3.11 в %LOCALAPPDATA%\Programs\Python\Python311
    set PYTHON311_PATH="%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
)

:: Если нашли Python 3.11 в стандартных путях, используем его
if defined PYTHON311_PATH (
    echo Будет использован Python 3.11: %PYTHON311_PATH%
    set PYTHON_CMD=%PYTHON311_PATH%
    goto :CheckPython
)

:: Пытаемся найти любую версию Python 3.11.x через py лаунчер
echo Проверка наличия Python 3.11 через лаунчер py...
py -3.11 --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set PYTHON_CMD=py -3.11
    echo Найден Python 3.11 через лаунчер py.
    goto :CheckPython
)

:: Если дошли сюда, пробуем использовать обычный python
echo Проверка стандартной команды python...
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ОШИБКА: Python не найден!
    echo Пожалуйста, установите Python 3.11 из файла "%PYTHON_PATH%" или c официального сайта.
    echo Нажмите любую клавишу для выхода...
    pause >nul
    exit /b 1
)

:: Проверяем версию стандартного python
python --version > temp_python_version.txt 2>nul
set /p PYTHON_VERSION=<temp_python_version.txt
del temp_python_version.txt 2>nul

echo Найден: %PYTHON_VERSION%

:: Проверяем, что это Python 3.11
echo %PYTHON_VERSION% | findstr /C:"Python 3.11" >nul
if %ERRORLEVEL% NEQ 0 (
    echo ВНИМАНИЕ: Найденная версия Python не является 3.11!
    echo Для корректной работы рекомендуется использовать Python 3.11
    echo.
    echo Нажмите любую клавишу, чтобы продолжить с найденной версией
    echo или закройте окно для отмены установки.
    pause >nul
)

:CheckPython
:: Проверка доступности выбранной версии Python
echo Проверка доступности выбранной версии Python...
%PYTHON_CMD% --version
if %ERRORLEVEL% NEQ 0 (
    echo ОШИБКА: Не удалось запустить команду: %PYTHON_CMD% --version
    echo Пожалуйста, проверьте установку Python 3.11.
    echo Нажмите любую клавишу для выхода...
    pause >nul
    exit /b 1
)

:: Удаление существующего окружения
if exist venv (
    echo Обнаружено существующее виртуальное окружение. Удаляем...
    call venv\Scripts\deactivate.bat 2>nul
    timeout /t 2 > nul
    rmdir /S /Q venv
    if exist venv (
        echo Не удалось удалить существующее окружение.
        echo Закройте все процессы Python и повторите попытку.
        echo Нажмите любую клавишу для выхода...
        pause >nul
        exit /b 1
    )
)

:: Создание виртуального окружения с выбранной версией Python
echo Создание виртуального окружения...
%PYTHON_CMD% -m venv venv
if %ERRORLEVEL% NEQ 0 (
    echo ОШИБКА: Не удалось создать виртуальное окружение!
    echo Нажмите любую клавишу для выхода...
    pause >nul
    exit /b 1
)

:: Активация окружения
echo Активация виртуального окружения...
call venv\Scripts\activate.bat
if %ERRORLEVEL% NEQ 0 (
    echo ОШИБКА: Не удалось активировать виртуальное окружение!
    echo Нажмите любую клавишу для выхода...
    pause >nul
    exit /b 1
)

:: Обновление pip и установка базовых инструментов
echo Обновление pip и установка базовых инструментов...
python -m pip install --upgrade pip
python -m pip install --upgrade setuptools wheel

:: Установка зависимостей
echo Установка необходимых библиотек...
python -m pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo ОШИБКА: Не удалось установить зависимости!
    echo Нажмите любую клавишу для выхода...
    pause >nul
    exit /b 1
)

:: Установка TensorFlow
echo Установка TensorFlow...
python -m pip install tensorflow==2.13.0
if %ERRORLEVEL% NEQ 0 (
    echo ПРЕДУПРЕЖДЕНИЕ: Не удалось установить TensorFlow 2.13.0.
    echo Пробуем установить совместимую версию...
    python -m pip install tensorflow
)

:: Проверка установленных пакетов
echo Проверка установленных пакетов...
python -c "import numpy; print('NumPy:', numpy.__version__)" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ПРЕДУПРЕЖДЕНИЕ: NumPy не установлен корректно.
)

python -c "import fastapi; print('FastAPI:', fastapi.__version__)" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ПРЕДУПРЕЖДЕНИЕ: FastAPI не установлен корректно.
)

python -c "import tensorflow; print('TensorFlow:', tensorflow.__version__)" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ПРЕДУПРЕЖДЕНИЕ: TensorFlow не установлен корректно.
    echo Если некоторые пакеты не установлены, попробуйте запустить скрипт еще раз.
)

:: Создание подкаталогов проекта (если не существуют)
echo Создание необходимых директорий...
if not exist "templates" mkdir templates
if not exist "static" mkdir static
if not exist "logs" mkdir logs
if not exist "temp_uploads" mkdir temp_uploads

:: Проверка существования директории для моделей
set MODEL_DIR=..\model_save_preset
if not exist "%MODEL_DIR%" (
    echo Создание директории для моделей: %MODEL_DIR%
    mkdir "%MODEL_DIR%"
    mkdir "%MODEL_DIR%\models"
    mkdir "%MODEL_DIR%\history"
)

:: Создаем тестовый файл с данными
echo Создание примера тестовых данных...
echo feature1,feature2,feature3,target > test_data.csv
echo 0.1,0.2,0.3,0.15 >> test_data.csv
echo 0.2,0.3,0.4,0.25 >> test_data.csv
echo 0.3,0.4,0.5,0.35 >> test_data.csv
echo 0.4,0.5,0.6,0.45 >> test_data.csv
echo 0.5,0.6,0.7,0.55 >> test_data.csv
echo 0.6,0.7,0.8,0.65 >> test_data.csv
echo 0.7,0.8,0.9,0.75 >> test_data.csv
echo 0.8,0.9,1.0,0.85 >> test_data.csv
echo 0.9,1.0,0.1,0.95 >> test_data.csv
echo 1.0,0.1,0.2,1.05 >> test_data.csv

:: Создаем конфигурационный файл
echo Создание файла конфигурации...
echo [Paths]> config.ini
echo models_directory = ../model_save_preset>> config.ini
echo.>> config.ini
echo [Data]>> config.ini
echo test_data_path = test_data.csv>> config.ini
echo.>> config.ini
echo [Logging]>> config.ini
echo log_level = INFO>> config.ini

echo.
echo =======================================================
echo Установка завершена успешно!
echo Для запуска сервера выполните: run_evaluator_fixed.bat
echo =======================================================
echo.

call venv\Scripts\deactivate.bat

echo Нажмите любую клавишу для выхода...
pause >nul 
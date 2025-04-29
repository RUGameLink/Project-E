@echo off 
chcp 65001 > nul 
cd /d "%~dp0" 
echo =======================================================
echo Запуск модуля оценки моделей нейронных сетей
echo =======================================================

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
    goto :CheckVenv
)

:: Пытаемся найти любую версию Python 3.11.x через py лаунчер
echo Проверка наличия Python 3.11 через лаунчер py...
py -3.11 --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set PYTHON_CMD=py -3.11
    echo Найден Python 3.11 через лаунчер py.
    goto :CheckVenv
)

:: Если дошли сюда, используем стандартный python
echo Будет использована стандартная команда python.

:CheckVenv
:: Проверяем наличие виртуального окружения
echo Проверка наличия виртуального окружения...
if not exist "venv" (
    echo Виртуальное окружение не найдено!
    echo Запустите сначала install_evaluator.bat для установки окружения.
    echo Нажмите любую клавишу для выхода...
    pause >nul
    exit /b 1
)

echo Активация виртуального окружения...
call venv\Scripts\activate.bat 
if %ERRORLEVEL% NEQ 0 (
    echo ОШИБКА: Не удалось активировать виртуальное окружение!
    echo Повторите запуск или переустановите окружение с помощью install_evaluator.bat
    echo Нажмите любую клавишу для выхода...
    pause >nul
    exit /b 1
)

:: Проверяем наличие config.ini
if not exist config.ini (
  echo Файл конфигурации config.ini не найден.
  echo Создание конфигурации по умолчанию...
  
  echo [Paths]> config.ini
  echo models_directory = ../model_save_preset>> config.ini
  echo.>> config.ini
  echo [Data]>> config.ini
  echo test_data_path = test_data.csv>> config.ini
  echo.>> config.ini
  echo [Logging]>> config.ini
  echo log_level = INFO>> config.ini
  
  echo Файл конфигурации config.ini создан с настройками по умолчанию.
  echo.
)

:: Создаем тестовые данные, если их нет
if not exist test_data.csv (
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
  echo Файл test_data.csv создан.
  echo.
)

:: Создаем директории, если они не существуют
echo Проверка наличия необходимых директорий...
if not exist "templates" mkdir templates
if not exist "static" mkdir static
if not exist "logs" mkdir logs
if not exist "temp_uploads" mkdir temp_uploads

:: Проверяем установленные пакеты перед запуском
echo Проверка критических зависимостей...
python -c "import fastapi; import uvicorn; print('FastAPI и uvicorn найдены')" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ПРЕДУПРЕЖДЕНИЕ: Отсутствуют необходимые пакеты (FastAPI или uvicorn)
    echo Рекомендуется переустановить окружение с помощью install_evaluator.bat
    echo.
    echo Нажмите любую клавишу, чтобы продолжить (сервер может не запуститься)...
    pause >nul
)

echo Запуск модуля оценки моделей... 
echo Веб-интерфейс будет доступен по адресу: http://localhost:8000/
echo.
echo Для остановки сервера нажмите Ctrl+C
echo.

venv\Scripts\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
if %ERRORLEVEL% NEQ 0 ( 
  echo. 
  echo ОШИБКА: Не удалось запустить сервер! 
  echo Рекомендуется запустить install_evaluator.bat для исправления проблемы.
  echo.
  echo Нажмите любую клавишу для выхода... 
  pause >nul
  exit /b 1 
) 

call venv\Scripts\deactivate.bat
echo.
echo Сервер остановлен. Нажмите любую клавишу для выхода...
pause >nul 
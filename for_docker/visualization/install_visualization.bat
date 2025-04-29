@echo off
chcp 65001 > nul
echo =======================================================
echo Установка визуализатора нейронных сетей
echo =======================================================
cd /d "%~dp0"

:: Проверка наличия Python
echo Проверка наличия Python...
python --version > temp_python_version.txt
set /p PYTHON_VERSION=<temp_python_version.txt
del temp_python_version.txt
echo %PYTHON_VERSION%

if %ERRORLEVEL% NEQ 0 (
    echo ОШИБКА: Python не найден!
    echo Пожалуйста, установите Python 3.8 или выше.
    pause
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
        echo Запустите remove_all.bat и повторите попытку.
        pause
        exit /b 1
    )
)

:: Создание виртуального окружения
echo Создание виртуального окружения...
python -m venv venv
if %ERRORLEVEL% NEQ 0 (
    echo ОШИБКА: Не удалось создать виртуальное окружение!
    pause
    exit /b 1
)

:: Активация окружения
call venv\Scripts\activate.bat

:: Обновление pip и установка базовых инструментов
echo Обновление pip и установка базовых инструментов...
python -m pip install --upgrade pip
python -m pip install --upgrade setuptools wheel

:: Установка зависимостей
echo Установка необходимых библиотек...
echo Использование предкомпилированных пакетов (wheels) для ускорения установки...

:: Проверка версии Python для выбора совместимого NumPy
echo %PYTHON_VERSION% | findstr "3.12" > nul
if %ERRORLEVEL% EQU 0 (
    echo - Обнаружен Python 3.12, установка NumPy 1.26.0...
    python -m pip install --only-binary :all: numpy==1.26.0 || python -m pip install numpy==1.26.0
) else (
    echo - Установка NumPy 1.24.3...
    python -m pip install --only-binary :all: numpy==1.24.3 || python -m pip install numpy==1.24.3
)

if %ERRORLEVEL% NEQ 0 (
    echo ОШИБКА: Не удалось установить NumPy!
    echo Попробуйте запустить:
    echo python -m pip install numpy
    pause
    exit /b 1
)

:: Затем устанавливаем основные зависимости
echo - Установка основных библиотек...
echo %PYTHON_VERSION% | findstr "3.12" > nul
if %ERRORLEVEL% EQU 0 (
    echo - Обнаружен Python 3.12, установка matplotlib 3.8.4...
    python -m pip install --only-binary :all: matplotlib==3.8.4 || python -m pip install matplotlib==3.8.4
) else (
    python -m pip install --only-binary :all: matplotlib==3.7.2 || python -m pip install matplotlib==3.7.2
)
echo Установлен matplotlib

:: Проверка версии Python для выбора совместимого pandas
echo %PYTHON_VERSION% | findstr "3.12" > nul
if %ERRORLEVEL% EQU 0 (
    echo - Обнаружен Python 3.12, установка pandas 2.1.4...
    python -m pip install --only-binary :all: pandas==2.1.4 || python -m pip install pandas==2.1.4
) else (
    echo - Установка pandas 2.0.3...
    python -m pip install --only-binary :all: pandas==2.0.3 || python -m pip install pandas==2.0.3
)
echo Установлен pandas

echo %PYTHON_VERSION% | findstr "3.12" > nul
if %ERRORLEVEL% EQU 0 (
    echo - Обнаружен Python 3.12, установка scipy 1.15.2...
    python -m pip install --only-binary :all: scipy==1.15.2 || python -m pip install scipy==1.15.2
) else (
    python -m pip install --only-binary :all: scipy==1.11.1 || python -m pip install scipy==1.11.1
)
echo Установлен scipy

echo %PYTHON_VERSION% | findstr "3.12" > nul
if %ERRORLEVEL% EQU 0 (
    echo - Обнаружен Python 3.12, установка scikit-learn 1.4.2...
    python -m pip install --only-binary :all: scikit-learn==1.4.2 || python -m pip install scikit-learn==1.4.2
) else (
    python -m pip install --only-binary :all: scikit-learn==1.3.0 || python -m pip install scikit-learn==1.3.0
)
echo Установлен scikit-learn

:: Установка TensorFlow
echo - Установка TensorFlow...
echo %PYTHON_VERSION% | findstr "3.12" > nul
if %ERRORLEVEL% EQU 0 (
    echo - Обнаружен Python 3.12, установка tensorflow 2.16.1...
    python -m pip install tensorflow==2.16.1
    if %ERRORLEVEL% NEQ 0 (
        echo Попытка установки tensorflow с другими параметрами...
        python -m pip install tensorflow
    )
) else (
    python -m pip install --only-binary :all: tensorflow-cpu==2.13.0 || python -m pip install tensorflow-cpu==2.13.0
)
echo Установлен TensorFlow

:: Другие библиотеки
echo - Установка дополнительных библиотек...
python -m pip install --only-binary :all: seaborn==0.12.2 || python -m pip install seaborn==0.12.2
python -m pip install --only-binary :all: plotly==5.15.0 || python -m pip install plotly==5.15.0

echo %PYTHON_VERSION% | findstr "3.12" > nul
if %ERRORLEVEL% EQU 0 (
    echo - Обнаружен Python 3.12, установка h5py 3.10.0...
    python -m pip install --only-binary :all: h5py==3.10.0 || python -m pip install h5py==3.10.0
) else (
    python -m pip install --only-binary :all: h5py==3.9.0 || python -m pip install h5py==3.9.0
)

:: Установка Streamlit (последняя, так как она зависит от других пакетов)
echo - Установка Streamlit...
python -m pip install --only-binary :all: streamlit==1.22.0 || python -m pip install streamlit==1.22.0

:: Проверка установленных пакетов
echo Проверка установленных пакетов...
python -c "import numpy; print('NumPy:', numpy.__version__)" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ПРЕДУПРЕЖДЕНИЕ: NumPy не установлен корректно.
)

python -c "import matplotlib; print('Matplotlib:', matplotlib.__version__)" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ПРЕДУПРЕЖДЕНИЕ: Matplotlib не установлен корректно.
)

python -c "import tensorflow; print('TensorFlow:', tensorflow.__version__)" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ПРЕДУПРЕЖДЕНИЕ: TensorFlow не установлен корректно.
)

python -c "import streamlit; print('Streamlit:', streamlit.__version__)" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ПРЕДУПРЕЖДЕНИЕ: Streamlit не установлен корректно.
    echo Если некоторые пакеты не установлены, попробуйте запустить скрипт еще раз.
)

:: Создание скриптов запуска
echo Создание скриптов запуска...

:: Создание скрипта запуска дашборда
echo @echo off > run_dashboard.bat
echo chcp 65001 ^> nul >> run_dashboard.bat
echo cd /d "%%~dp0" >> run_dashboard.bat
echo call venv\Scripts\activate.bat >> run_dashboard.bat
echo echo Запуск дашборда визуализации моделей... >> run_dashboard.bat
echo venv\Scripts\python.exe -m streamlit run dashboard.py >> run_dashboard.bat
echo if %%ERRORLEVEL%% NEQ 0 ( >> run_dashboard.bat
echo   echo. >> run_dashboard.bat
echo   echo ОШИБКА: Не удалось запустить дашборд! >> run_dashboard.bat
echo   echo Попробуйте запустить install_visualization.bat для исправления проблемы. >> run_dashboard.bat
echo   pause >> run_dashboard.bat
echo   exit /b 1 >> run_dashboard.bat
echo ) >> run_dashboard.bat
echo call venv\Scripts\deactivate.bat >> run_dashboard.bat

:: Создание скрипта запуска визуализатора моделей
echo @echo off > run_visualizer.bat
echo chcp 65001 ^> nul >> run_visualizer.bat
echo cd /d "%%~dp0" >> run_visualizer.bat
echo call venv\Scripts\activate.bat >> run_visualizer.bat
echo echo Запуск визуализатора моделей... >> run_visualizer.bat
echo venv\Scripts\python.exe -m streamlit run model_visualizer.py >> run_visualizer.bat
echo if %%ERRORLEVEL%% NEQ 0 ( >> run_visualizer.bat
echo   echo. >> run_visualizer.bat
echo   echo ОШИБКА: Не удалось запустить визуализатор! >> run_visualizer.bat
echo   echo Попробуйте запустить install_visualization.bat для исправления проблемы. >> run_visualizer.bat
echo   pause >> run_visualizer.bat
echo   exit /b 1 >> run_visualizer.bat
echo ) >> run_visualizer.bat
echo call venv\Scripts\deactivate.bat >> run_visualizer.bat

:: Деактивация окружения
call venv\Scripts\deactivate.bat

echo.
echo =======================================================
echo Установка успешно завершена!
echo.
echo Созданы скрипты запуска:
echo - run_dashboard.bat - для запуска дашборда (полная версия)
echo - run_visualizer.bat - для запуска визуализатора (базовая версия)
echo =======================================================
pause 
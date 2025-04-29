@echo off 
chcp 65001 > nul 
cd /d "%~dp0" 
call venv\Scripts\activate.bat 
echo Запуск визуализатора моделей... 
venv\Scripts\python.exe -m streamlit run model_visualizer.py 
if %ERRORLEVEL% NEQ 0 ( 
  echo. 
  echo ОШИБКА: Не удалось запустить визуализатор! 
  echo Попробуйте запустить install_visualization.bat для исправления проблемы. 
  pause 
  exit /b 1 
) 
call venv\Scripts\deactivate.bat 

@echo off 
chcp 65001 > nul 
cd /d "%~dp0" 
call venv\Scripts\activate.bat 

:: Проверяем наличие config.ini
if not exist config.ini (
  echo Файл конфигурации config.ini не найден.
  echo Создание конфигурации по умолчанию...
  
  :: Используем PowerShell для создания файла с кодировкой UTF-8 (поддержка кириллицы)
  powershell -Command "& {
    $content = @'
[Paths]
models_directory = ../model_save_preset

[Logging]
log_level = INFO
'@
    [System.IO.File]::WriteAllText('%~dp0config.ini', $content, [System.Text.Encoding]::UTF8)
  }"

  :: Проверяем успешность создания файла
  if exist config.ini (
    echo Файл config.ini создан с настройками по умолчанию.
    echo Для изменения настроек запустите setup_config.bat или отредактируйте файл config.ini.
    echo.
  ) else (
    echo ОШИБКА: Не удалось создать файл config.ini с помощью PowerShell.
    echo Попытка создания через эхо команды...
    
    echo [Paths]> config.ini
    echo models_directory = ../model_save_preset>> config.ini
    echo.>> config.ini
    echo [Logging]>> config.ini
    echo log_level = INFO>> config.ini
    
    echo Файл config.ini создан с настройками по умолчанию.
    echo Для изменения настроек запустите setup_config.bat или отредактируйте файл config.ini.
    echo.
  )
)

echo Запуск дашборда визуализации моделей... 
venv\Scripts\python.exe -m streamlit run dashboard.py 
if %ERRORLEVEL% NEQ 0 ( 
  echo. 
  echo ОШИБКА: Не удалось запустить дашборд! 
  echo Попробуйте запустить install_visualization.bat для исправления проблемы. 
  pause 
  exit /b 1 
) 
call venv\Scripts\deactivate.bat 

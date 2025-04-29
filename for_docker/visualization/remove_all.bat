@echo off
chcp 65001 > nul
echo =======================================================
echo Удаление виртуального окружения визуализатора
echo =======================================================
cd /d "%~dp0"

echo Остановка процессов Python и Streamlit...
taskkill /F /IM streamlit.exe >nul 2>&1
taskkill /F /IM python.exe >nul 2>&1

echo Деактивация виртуального окружения (если активно)...
call venv\Scripts\deactivate.bat 2>nul

echo Удаление папки виртуального окружения...
if exist venv (
    rmdir /S /Q venv
    if exist venv (
        echo ВНИМАНИЕ: Не удалось автоматически удалить папку venv.
        echo Возможно, некоторые файлы используются другими процессами.
        echo Попробуйте закрыть все программы и выполнить скрипт снова,
        echo или удалите папку вручную.
    ) else (
        echo Виртуальное окружение успешно удалено.
    )
) else (
    echo Виртуальное окружение не найдено.
)

echo Удаление сгенерированных скриптов запуска...
if exist run_dashboard.bat del run_dashboard.bat
if exist run_visualizer.bat del run_visualizer.bat

echo =======================================================
echo Очистка завершена.
echo Для установки визуализатора запустите install_visualization.bat
echo =======================================================
pause 
@echo off
title PREDICT/90 - daily predictions
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
python --version >nul 2>nul
if errorlevel 1 (
  echo.
  echo  Python is not installed.
  echo  Install Python 3.10 or newer from https://www.python.org/downloads/
  echo  and tick "Add python.exe to PATH" during the install. Then run this file again.
  echo.
  pause
  exit /b
)
if not exist ".installed" (
  echo First run: installing the needed packages. This takes 2-5 minutes, only once.
  python -m pip install --upgrade pip >nul
  python -m pip install -r requirements.txt
  if errorlevel 1 (
    echo Package install failed. Check your internet and run this file again.
    pause
    exit /b
  )
  echo ok> .installed
)
python run_daily.py %*
echo.
pause

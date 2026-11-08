@echo off
title PREDICT/90 - notebook
cd /d "%~dp0"
if not exist ".installed" (
  echo Run RUN_ME.bat once first, it installs everything.
  pause
  exit /b
)
python -m notebook Football_Predictor.ipynb

@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
python run_daily.py --quiet >> output\scheduled_run.log 2>&1

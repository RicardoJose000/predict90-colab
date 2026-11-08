@echo off
title PREDICT/90 - daily schedule
cd /d "%~dp0"
set /p T="Time for the daily run, 24h format (example 08:00): "
if "%T%"=="" set T=08:00
schtasks /Create /SC DAILY /TN "PREDICT90 daily" /TR "\"%~dp0run_silent.bat\"" /ST %T% /F
if errorlevel 1 (echo Could not create the task.) else (echo Done. Every day at %T% the predictions are made by themselves. The Excel files appear in the output folder.)
echo To remove it later: schtasks /Delete /TN "PREDICT90 daily" /F
pause

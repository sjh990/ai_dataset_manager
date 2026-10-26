@echo off
REM ==== YOLO Dataset Augmentor - Windows launcher ====
REM First run: creates the venv and installs packages (needs internet). Later runs start instantly.
setlocal
call "%~dp0_env.bat" ensure
if errorlevel 1 ( pause & exit /b 1 )
cd /d "%ROOT%"
start "" "%VPYW%" main.py %*
endlocal

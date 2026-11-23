@echo off
rem Facility labeling tool
rem Uses .venv made by setup.bat. Falls back to conda env ai_label if there is no .venv.
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" main.py
) else if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\anaconda3\Scripts\activate.bat" ai_label
    python main.py
) else (
    echo No Python environment found. Run setup.bat first.
    pause
    exit /b 1
)
if errorlevel 1 pause

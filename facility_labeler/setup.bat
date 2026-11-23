@echo off
rem ============================================================
rem  Facility labeling tool - first-time setup (no Anaconda needed)
rem
rem  Double-click once per PC. Creates .venv in this folder with
rem  Python 3.11 + all packages (same versions as the labeling PC),
rem  then downloads the AI start weights into models\ (~720MB).
rem    setup.bat        GPU version (NVIDIA, CUDA 12.8)
rem    setup.bat cpu    CPU only (PC without NVIDIA GPU)
rem  Needs internet. Takes about 5-15 minutes (torch is ~3GB).
rem ============================================================
setlocal
cd /d "%~dp0"

set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
if /i "%~1"=="cpu" set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"

rem ---- 1. uv (Python + package installer, no admin rights needed) ----
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
where uv >nul 2>nul
if errorlevel 1 (
    echo [1/5] Installing uv ...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    where uv >nul 2>nul
    if errorlevel 1 goto :fail
) else (
    echo [1/5] uv found
)

rem ---- 2. Python 3.11 virtual environment (.venv) ----
if exist ".venv\Scripts\python.exe" (
    echo [2/5] .venv already exists, reusing it
) else (
    echo [2/5] Creating .venv with Python 3.11 ...
    uv venv .venv --python 3.11 --seed
    if errorlevel 1 goto :fail
)
set "PY=.venv\Scripts\python.exe"

rem ---- 3. torch / torchvision (GPU or CPU build) ----
echo [3/5] Installing torch from %TORCH_INDEX% ...
uv pip install --python "%PY%" torch==2.11.0 torchvision==0.26.0 --index-url "%TORCH_INDEX%"
if errorlevel 1 goto :fail

rem ---- 4. everything else, pinned versions ----
echo [4/5] Installing packages from requirements.txt ...
uv pip install --python "%PY%" -r requirements.txt
if errorlevel 1 goto :fail

echo.
"%PY%" tools\check_env.py
if errorlevel 1 goto :fail

rem ---- 5. AI start weights (not on GitHub, over 100MB each) ----
echo.
echo [5/5] Downloading AI models into models\ ...
"%PY%" tools\download_models.py
if errorlevel 1 goto :fail
echo.
echo Setup done. Start the tool with run.bat
pause
exit /b 0

:fail
echo.
echo Setup FAILED. Check the error message above.
echo (Run setup.bat again: finished steps are skipped. To start clean, delete .venv first.)
pause
exit /b 1

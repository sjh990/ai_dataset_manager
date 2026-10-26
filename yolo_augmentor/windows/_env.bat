@echo off
REM ==== Shared environment for YOLO Dataset Augmentor Windows scripts ====
REM   call "%~dp0_env.bat"          -> sets ROOT, PY, VENV, VPY, VPYW
REM   call "%~dp0_env.bat" ensure   -> also creates/repairs the venv and installs base packages
REM
REM The venv is placed OUTSIDE OneDrive when the project is inside a OneDrive folder:
REM OneDrive sync (Files On-Demand) corrupts venv files, e.g.
REM   ModuleNotFoundError: No module named 'pip._vendor.rich.markup'
REM Override the location with:  set YOLOAUG_VENV=D:\venvs\yoloaug

set "ROOT=%~dp0.."
for %%I in ("%ROOT%") do set "ROOT=%%~fI"

set "ONEDRIVE="
echo %ROOT%| findstr /i "OneDrive" >nul
if not errorlevel 1 set "ONEDRIVE=1"
set "VENV=%ROOT%\.venv"
if defined ONEDRIVE set "VENV=%LOCALAPPDATA%\YOLOAugmentor\venv"
if defined YOLOAUG_VENV set "VENV=%YOLOAUG_VENV%"
set "VPY=%VENV%\Scripts\python.exe"
set "VPYW=%VENV%\Scripts\pythonw.exe"

REM ---- pick Python (3.12 / 3.11 / 3.10 / 3.13 preferred: PyTorch CUDA wheels exist for these)
if defined PY goto py_done
where py >nul 2>nul
if errorlevel 1 goto py_plain
for %%V in (3.12 3.11 3.10 3.13) do if not defined PY py -%%V -c "import sys" >nul 2>nul && set "PY=py -%%V"
if not defined PY py -3 -c "import sys" >nul 2>nul && set "PY=py -3"
:py_plain
if not defined PY where python >nul 2>nul && set "PY=python"
:py_done

if /i not "%~1"=="ensure" exit /b 0

REM ================================================================ ensure
if not defined PY (
    echo [ERROR] Python 3.10-3.12 not found. Install from https://www.python.org/downloads/ ^(check "Add python.exe to PATH"^).
    exit /b 1
)
echo Python : %PY%
echo venv   : %VENV%
REM an old venv left inside OneDrive by v1.3 or earlier is not used any more
if defined ONEDRIVE if /i not "%VENV%"=="%ROOT%\.venv" if exist "%ROOT%\.venv" (
    echo [cleanup] Removing the old .venv inside the OneDrive folder ^(not used any more^)...
    rmdir /s /q "%ROOT%\.venv" >nul 2>nul
)

REM venv exists but pip is broken (partial sync / interrupted upgrade) -> rebuild it
if not exist "%VPY%" goto create
"%VPY%" -c "import pip._internal.cli.main_parser" >nul 2>nul
if not errorlevel 1 goto base
echo [repair] The virtual environment is damaged. Recreating it...
rmdir /s /q "%VENV%"

:create
echo [setup] Creating virtual environment...
%PY% -m venv "%VENV%"
if errorlevel 1 (
    echo [ERROR] venv creation failed.
    exit /b 1
)
"%VPY%" -c "import pip._internal.cli.main_parser" >nul 2>nul
if errorlevel 1 (
    "%VPY%" -m ensurepip --upgrade >nul 2>nul
)

:base
"%VPY%" -c "import PySide6, cv2, numpy, yaml, PIL" >nul 2>nul
if not errorlevel 1 exit /b 0
echo [setup] Installing base packages ^(1-3 min^)...
"%VPY%" -m pip install --disable-pip-version-check -r "%ROOT%\requirements.txt"
if errorlevel 1 (
    echo [ERROR] base package install failed. Check the internet connection / proxy and run again.
    exit /b 1
)
exit /b 0

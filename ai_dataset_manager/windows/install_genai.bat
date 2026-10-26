@echo off
REM ==== Install AI dependencies (auto labeling + project model training) on an internet-connected PC ====
REM PyTorch CUDA 12.8 build: RTX 30 / 40 / 50 series (RTX 3080, 4060, 5050 ...), NVIDIA driver 570+ required
setlocal
set CUDA=cu128
if not "%~1"=="" set CUDA=%~1
call "%~dp0_env.bat" ensure
if errorlevel 1 ( pause & exit /b 1 )
cd /d "%ROOT%"

"%VPY%" -c "import torch,torchvision,sys; sys.exit(0 if torch.version.cuda else 1)" >nul 2>nul
if not errorlevel 1 goto have_torch
echo Installing PyTorch + torchvision (%CUDA%) ... about 3GB
"%VPY%" -m pip install --disable-pip-version-check torch torchvision --index-url https://download.pytorch.org/whl/%CUDA%
if errorlevel 1 ( echo [ERROR] torch install failed & pause & exit /b 1 )
:have_torch
"%VPY%" -m pip install --disable-pip-version-check -r requirements-genai.txt
if errorlevel 1 ( echo [ERROR] AI packages install failed & pause & exit /b 1 )
echo.
set /p UL=Also install Ultralytics YOLO (optional engine, AGPL-3.0, internal use only)? [y/N]:
if /i "%UL%"=="y" "%VPY%" -m pip install --disable-pip-version-check -r requirements-ultralytics.txt
"%VPY%" -c "import torch; ok=torch.cuda.is_available(); print('torch', torch.__version__, '| CUDA available:', ok, torch.cuda.get_device_name(0) if ok else '')"

echo.
set /p DL=Download required models now (about 2.3GB)? [y/N]:
if /i "%DL%"=="y" "%VPY%" tools\download_models.py --required
echo Done. Start with windows\run.bat
pause
endlocal

@echo off
REM ==== Build standalone Windows executable (PyInstaller) ====
REM Output: dist\YOLOAugmentor\YOLOAugmentor.exe  (copy the whole YOLOAugmentor folder to other PCs)
setlocal
call "%~dp0_env.bat" ensure
if errorlevel 1 ( pause & exit /b 1 )
cd /d "%ROOT%"
"%VPY%" -m pip install --disable-pip-version-check pyinstaller
if errorlevel 1 ( echo [ERROR] pyinstaller install failed & pause & exit /b 1 )
"%VPY%" -m PyInstaller --noconfirm --clean YOLOAugmentor.spec
if errorlevel 1 ( echo [ERROR] build failed & pause & exit /b 1 )

echo.
echo ===========================================================
echo  Build complete: dist\YOLOAugmentor\YOLOAugmentor.exe
echo ===========================================================
explorer "dist\YOLOAugmentor"
pause
endlocal

@echo off
rem Facility labeling tool (uses conda env: ai_label)
cd /d "%~dp0"
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" ai_label
python main.py
if errorlevel 1 pause

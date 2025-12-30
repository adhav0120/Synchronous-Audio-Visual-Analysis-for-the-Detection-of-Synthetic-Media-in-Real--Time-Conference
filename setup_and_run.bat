@echo off
TITLE Deepfake Voice Detector Setup & Run
ECHO ===================================================
ECHO    Deepfake Voice Detector - One-Click Installer
ECHO ===================================================
ECHO.

REM Check if Python is installed
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    ECHO [ERROR] Python is not installed or not in your PATH.
    ECHO Please install Python 3.10+ from: https://www.python.org/downloads/
    ECHO IMPORTANT: Check "Add Python to PATH" during installation.
    PAUSE
    EXIT /B
)

REM Create virtual environment if it doesn't exist
IF NOT EXIST ".venv" (
    ECHO [1/3] Creating isolated environment (.venv)...
    python -m venv .venv
) ELSE (
    ECHO [1/3] Environment found.
)

REM Install dependencies
ECHO [2/3] Installing/Updating dependencies (this may take a minute)...
.venv\Scripts\pip install -r requirements.txt --upgrade

REM Run the application
ECHO [3/3] Launching Deepfake Detector...
ECHO.
.venv\Scripts\python src\desktop_live_capture.py

PAUSE

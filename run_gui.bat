@echo off
REM Vision-Fit Windows Launcher Script
REM Run this file to launch the Vision-Fit GUI application

echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║         Vision-Fit: AI-Based Body Measurement              ║
echo ║                  GUI Application                           ║
echo ╚════════════════════════════════════════════════════════════╝
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python is not installed or not in PATH
    echo.
    echo   Please install Python 3.8+ from: https://www.python.org
    echo   Make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)

echo ✓ Python detected
echo.

REM Check if required packages are installed
echo Checking dependencies...
python -m pip show opencv-python >nul 2>&1
if errorlevel 1 (
    echo.
    echo ⚠️  Missing dependency: opencv-python
    echo Installing required packages...
    echo.
    pip install opencv-python mediapipe numpy pillow
    if errorlevel 1 (
        echo ❌ Failed to install dependencies
        pause
        exit /b 1
    )
)

echo ✓ Dependencies verified
echo.

REM Launch the GUI
echo Launching Vision-Fit GUI...
echo.

python app_gui.py

if errorlevel 1 (
    echo.
    echo ❌ Error launching GUI
    echo Please check the error message above
    echo.
    pause
    exit /b 1
)

exit /b 0
@echo off
echo ===============================================
echo PolyVision Executable Build Setup
echo ===============================================

:: This script lives in packaging\; run everything from the project root
pushd "%~dp0.."

:: Check if virtual environment exists
if not exist "venv" (
    echo Error: Virtual environment 'venv' not found!
    echo Please ensure your virtual environment is named 'venv' and located in the project root.
    popd
    pause
    exit /b 1
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Install build requirements
echo Installing build requirements...
pip install -r "%~dp0build_requirements.txt"
if %errorlevel% neq 0 (
    echo Error: Build requirements installation failed!
    popd
    pause
    exit /b 1
)

:: Verify PyInstaller installation
echo Verifying PyInstaller installation...
pyinstaller --version
if %errorlevel% neq 0 (
    echo Error: PyInstaller installation failed!
    popd
    pause
    exit /b 1
)

echo.
echo ===============================================
echo Setup complete!
echo Before running build_exe.bat, make sure this GPU build machine passes:
echo venv\Scripts\python.exe ui\PolyVisionMain.py --diagnose-retraining --require-gpu --json
echo ===============================================
popd
pause

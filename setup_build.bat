@echo off
echo ===============================================
echo PolyVision Executable Build Setup
echo ===============================================

:: Check if virtual environment exists
if not exist "venv" (
    echo Error: Virtual environment 'venv' not found!
    echo Please ensure your virtual environment is named 'venv' and located in the project root.
    pause
    exit /b 1
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Install build requirements
echo Installing build requirements...
pip install -r build_requirements.txt

:: Verify PyInstaller installation
echo Verifying PyInstaller installation...
pyinstaller --version
if %errorlevel% neq 0 (
    echo Error: PyInstaller installation failed!
    pause
    exit /b 1
)

echo.
echo ===============================================
echo Setup complete! You can now run build_exe.bat
echo ===============================================
pause
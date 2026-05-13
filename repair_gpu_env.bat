@echo off
setlocal

cd /d "%~dp0"
set "LOG_FILE=%CD%\repair_gpu_env.log"
set "PYTHONNOUSERSITE=1"

echo ===============================================
echo PolyVision NVIDIA GPU Repair
echo ===============================================
echo Repair log: %LOG_FILE%
echo.
echo This repair targets the source/venv release of PolyVision.
echo It requires internet access and will reinstall PyTorch with CUDA 11.8 support.
echo.
echo IMPORTANT:
echo   1. Close PolyVision before continuing.
echo   2. Leave this repair window open.
echo   3. Restart PolyVision after the repair completes.
echo.
pause

echo PolyVision GPU repair started at %DATE% %TIME% > "%LOG_FILE%"

if not exist "venv\Scripts\activate.bat" (
    echo.
    echo ERROR: venv\Scripts\activate.bat was not found.
    echo Run this repair from the PolyVision project root with a venv folder present.
    goto fail
)

if not exist "detectron2" (
    echo.
    echo ERROR: detectron2 folder was not found in the project root.
    echo Clone Detectron2 into this folder before running the repair.
    goto fail
)

echo.
echo Activating virtual environment...
call "venv\Scripts\activate.bat" >> "%LOG_FILE%" 2>&1
if errorlevel 1 goto fail

echo.
echo Updating pip...
python -m pip install --upgrade pip >> "%LOG_FILE%" 2>&1
if errorlevel 1 goto fail

echo.
echo Installing CUDA-capable PyTorch packages...
python -m pip install --upgrade --force-reinstall torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118 >> "%LOG_FILE%" 2>&1
if errorlevel 1 goto fail

echo.
echo Restoring PolyVision dependency pins after GPU package installation...
python -m pip install --force-reinstall numpy==1.25.2 Pillow==8.4.0 idna==2.10 requests==2.31.0 urllib3==2.0.4 charset-normalizer==3.2.0 certifi==2022.12.7 >> "%LOG_FILE%" 2>&1
if errorlevel 1 goto fail

echo.
echo Reinstalling local Detectron2 against the current PyTorch build...
if exist "detectron2\build" rmdir /s /q "detectron2\build"
del /s /q "detectron2\detectron2\*.pyd" >nul 2>nul
python -m pip install --no-build-isolation -e detectron2 >> "%LOG_FILE%" 2>&1
if errorlevel 1 goto fail

echo.
echo Re-checking dependency consistency...
python -m pip check >> "%LOG_FILE%" 2>&1
if errorlevel 1 goto fail

echo.
echo Verifying GPU environment...
python repair_gpu_env.py >> "%LOG_FILE%" 2>&1
if errorlevel 1 goto fail

echo.
echo ===============================================
echo GPU repair completed successfully.
echo Restart PolyVision before starting retraining.
echo ===============================================
type "%LOG_FILE%"
pause
exit /b 0

:fail
echo.
echo ===============================================
echo GPU repair failed.
echo Review %LOG_FILE% and keep using CPU retraining until the environment is fixed.
echo ===============================================
if exist "%LOG_FILE%" type "%LOG_FILE%"
pause
exit /b 1

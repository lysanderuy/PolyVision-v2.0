@echo off
setlocal EnableExtensions

:: This script lives in packaging\; the project root is one level up
cd /d "%~dp0.."
set "PROJECT_ROOT=%CD%"
if not exist "logs" mkdir "logs"
set "LOG_FILE=%CD%\logs\repair_gpu_env.log"
set "ORIGINAL_PYTHONPATH=%PYTHONPATH%"
set "PYTHONNOUSERSITE=1"
set "TEMP_REPAIR_DIR=%TEMP%\PolyVisionGpuRepair_%RANDOM%_%RANDOM%"
set "TEMP_VENV=%TEMP_REPAIR_DIR%\venv"
set "TEMP_PYTHON=%TEMP_VENV%\Scripts\python.exe"
set "WHEEL_DIR=%TEMP_REPAIR_DIR%\wheel"
set "DETECTRON2_WHEEL="
set "COMPAT_PACKAGES=numpy==1.25.2 Pillow==8.4.0 idna==2.10 requests==2.31.0 urllib3==2.0.4 charset-normalizer==3.2.0 certifi==2022.12.7"

echo ===============================================
echo PolyVision NVIDIA GPU Repair
echo ===============================================
echo Repair log: %LOG_FILE%
echo.
if /I "%~1"=="--preflight-only" (
    echo This check validates source GPU repair prerequisites.
    echo No packages or active runtime files will be changed.
) else (
    echo TECHNICAL ADMINISTRATOR TOOL - not for packaged application users.
    echo If you normally open PolyVision.exe, close this window and contact support.
    echo Do not install CUDA Toolkit or repair Python packages yourself.
    echo.
    echo This repair targets only the source/venv release of PolyVision.
    echo It requires internet access and will reinstall PyTorch with CUDA 11.8 support.
    echo.
    echo IMPORTANT:
    echo   1. Close PolyVision before continuing.
    echo   2. Leave this repair window open.
    echo   3. Restart PolyVision after the repair completes.
    echo   4. Installed MSVC build tools and CUDA 11.8 will be detected automatically.
)
echo.
if /I not "%~1"=="--preflight-only" pause

echo PolyVision GPU repair started at %DATE% %TIME% > "%LOG_FILE%"

if not exist "venv\Scripts\activate.bat" (
    echo.
    echo ERROR: venv\Scripts\activate.bat was not found.
    echo Run this repair from the PolyVision project root with a venv folder present.
    goto fail
)

if not exist "detectron2\setup.py" (
    echo.
    echo ERROR: detectron2 source was not found in the project root.
    echo Clone Detectron2 into this folder before running the repair.
    goto fail
)

echo.
echo Preparing MSVC and CUDA build environment...
call :prepare_build_environment

echo.
echo Running repair preflight before changing the active environment...
"venv\Scripts\python.exe" "%~dp0repair_gpu_env.py" --preflight-repair >> "%LOG_FILE%" 2>&1
if errorlevel 1 goto fail

if /I "%~1"=="--preflight-only" (
    echo.
    echo GPU repair preflight passed. No packages were changed.
    type "%LOG_FILE%"
    exit /b 0
)

echo.
echo Creating temporary validation environment...
"venv\Scripts\python.exe" -m venv "%TEMP_VENV%" >> "%LOG_FILE%" 2>&1
if errorlevel 1 goto fail

echo.
echo Preparing temporary build tools...
"%TEMP_PYTHON%" -m pip install --upgrade pip wheel ninja setuptools==65.5.0 >> "%LOG_FILE%" 2>&1
if errorlevel 1 goto fail

echo.
echo Installing CUDA-capable PyTorch packages in the temporary environment...
"%TEMP_PYTHON%" -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118 >> "%LOG_FILE%" 2>&1
if errorlevel 1 goto fail

echo.
echo Applying project compatibility pins in the temporary environment...
"%TEMP_PYTHON%" -m pip install --force-reinstall %COMPAT_PACKAGES% >> "%LOG_FILE%" 2>&1
if errorlevel 1 goto fail

echo.
echo Copying Detectron2 source into the temporary build workspace...
xcopy /E /I /Q /Y "%PROJECT_ROOT%\detectron2" "%TEMP_REPAIR_DIR%\detectron2" >> "%LOG_FILE%" 2>&1
if errorlevel 1 goto fail

echo.
echo Removing stale native artifacts from the temporary Detectron2 copy...
if exist "%TEMP_REPAIR_DIR%\detectron2\build" rmdir /s /q "%TEMP_REPAIR_DIR%\detectron2\build"
del /s /q "%TEMP_REPAIR_DIR%\detectron2\detectron2\*.pyd" >nul 2>nul

echo.
echo Building a replacement Detectron2 wheel without touching the active environment...
mkdir "%WHEEL_DIR%" >> "%LOG_FILE%" 2>&1
set "FORCE_CUDA=1"
"%TEMP_PYTHON%" -m pip wheel --no-build-isolation --no-deps "%TEMP_REPAIR_DIR%\detectron2" --wheel-dir "%WHEEL_DIR%" >> "%LOG_FILE%" 2>&1
if errorlevel 1 goto fail

for %%F in ("%WHEEL_DIR%\detectron2-*.whl") do set "DETECTRON2_WHEEL=%%~fF"
if not defined DETECTRON2_WHEEL (
    echo ERROR: Detectron2 wheel was not created. >> "%LOG_FILE%"
    goto fail
)

echo.
echo Validating the replacement wheel in the temporary environment...
"%TEMP_PYTHON%" -m pip install "%DETECTRON2_WHEEL%" >> "%LOG_FILE%" 2>&1
if errorlevel 1 goto fail

pushd "%TEMP_REPAIR_DIR%"
set "PYTHONPATH=%PROJECT_ROOT%\UI"
"%TEMP_PYTHON%" -m retraining_runtime.diagnostic_cli --diagnose-retraining --require-gpu >> "%LOG_FILE%" 2>&1
set "TEMP_DIAGNOSTIC_ERROR=%ERRORLEVEL%"
popd
set "PYTHONPATH=%ORIGINAL_PYTHONPATH%"
if not "%TEMP_DIAGNOSTIC_ERROR%"=="0" goto fail

echo.
echo Temporary validation passed. Updating the active environment...
"venv\Scripts\python.exe" -m pip install --upgrade pip >> "%LOG_FILE%" 2>&1
if errorlevel 1 goto fail

"venv\Scripts\python.exe" -m pip install --upgrade --force-reinstall torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118 >> "%LOG_FILE%" 2>&1
if errorlevel 1 goto fail

"venv\Scripts\python.exe" -m pip install --force-reinstall %COMPAT_PACKAGES% >> "%LOG_FILE%" 2>&1
if errorlevel 1 goto fail

"venv\Scripts\python.exe" -m pip install --force-reinstall --no-deps "%DETECTRON2_WHEEL%" >> "%LOG_FILE%" 2>&1
if errorlevel 1 goto fail

echo.
echo Re-checking dependency consistency...
"venv\Scripts\python.exe" -m pip check >> "%LOG_FILE%" 2>&1
if errorlevel 1 goto fail

echo.
echo Verifying the active retraining runtime...
pushd "%TEMP_REPAIR_DIR%"
set "PYTHONPATH=%PROJECT_ROOT%\UI"
"%PROJECT_ROOT%\venv\Scripts\python.exe" -m retraining_runtime.diagnostic_cli --diagnose-retraining --require-gpu >> "%LOG_FILE%" 2>&1
set "ACTIVE_DIAGNOSTIC_ERROR=%ERRORLEVEL%"
popd
set "PYTHONPATH=%ORIGINAL_PYTHONPATH%"
if not "%ACTIVE_DIAGNOSTIC_ERROR%"=="0" goto fail

echo.
echo Cleaning temporary validation environment...
echo %TEMP_REPAIR_DIR% | findstr /B /I /C:"%TEMP%\PolyVisionGpuRepair_" >nul
if errorlevel 1 goto fail
rmdir /s /q "%TEMP_REPAIR_DIR%"
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
if /I "%~1"=="--preflight-only" goto preflight_fail
echo.
echo ===============================================
echo GPU repair failed.
echo Review %LOG_FILE%. Do not retrain unless the diagnostic confirms CPU fallback is safe.
echo Temporary validation files, if created, are at:
echo %TEMP_REPAIR_DIR%
echo ===============================================
if exist "%LOG_FILE%" type "%LOG_FILE%"
if /I not "%~1"=="--preflight-only" pause
exit /b 1

:preflight_fail
echo.
echo ===============================================
echo GPU repair preflight blocked. No packages were changed.
echo Review %LOG_FILE% and resolve the reported prerequisites before running the repair.
echo ===============================================
if exist "%LOG_FILE%" type "%LOG_FILE%"
exit /b 1

:prepare_build_environment
if not defined CUDA_PATH if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\nvcc.exe" set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
if defined CUDA_PATH set "PATH=%CUDA_PATH%\bin;%PATH%"

set "DISTUTILS_USE_SDK=1"

set "CL_ALREADY_AVAILABLE="
where cl.exe >nul 2>&1
if not errorlevel 1 set "CL_ALREADY_AVAILABLE=1"

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" exit /b 0

set "VS_INSTALL="
for /f "usebackq tokens=*" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS_INSTALL=%%I"
if not defined VS_INSTALL exit /b 0
if not exist "%VS_INSTALL%\Common7\Tools\VsDevCmd.bat" exit /b 0

set "COMPATIBLE_MSVC_TOOLSET="
for /f "delims=" %%T in ('dir /b /ad /o-n "%VS_INSTALL%\VC\Tools\MSVC\14.3*" 2^>nul') do if not defined COMPATIBLE_MSVC_TOOLSET set "COMPATIBLE_MSVC_TOOLSET=%%T"
if defined COMPATIBLE_MSVC_TOOLSET (
    call "%VS_INSTALL%\Common7\Tools\VsDevCmd.bat" -no_logo -arch=x64 -host_arch=x64 -vcvars_ver=%COMPATIBLE_MSVC_TOOLSET%
) else if not defined CL_ALREADY_AVAILABLE (
    call "%VS_INSTALL%\Common7\Tools\VsDevCmd.bat" -no_logo -arch=x64 -host_arch=x64
)
exit /b 0

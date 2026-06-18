@echo off
echo ===============================================
echo Building PolyVision Executable
echo ===============================================

:: This script lives in packaging\; run everything from the project root
pushd "%~dp0.."

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 goto fail

echo.
echo Verifying source GPU retraining runtime before packaging...
venv\Scripts\python.exe ui\PolyVisionMain.py --diagnose-retraining --require-gpu --json
if errorlevel 1 (
    echo.
    echo ERROR: Source runtime is not GPU-ready. Do not package from this environment.
    echo Run repair_gpu_env.bat on a GPU build machine and verify the diagnostic first.
    goto fail
)

pyinstaller --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: PyInstaller is not installed in this virtual environment.
    echo Run setup_build.bat first.
    goto fail
)

:: Clean previous builds
echo Cleaning previous builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

:: Build the executable
echo Building executable with PyInstaller...
pyinstaller --clean --noconfirm "%~dp0PolyVision.spec"
if errorlevel 1 goto fail

:: Check if build was successful
if exist "dist\PolyVision\PolyVision.exe" (
    echo.
    echo ===============================================
    echo PyInstaller build completed.
    echo Executable location: dist\PolyVision\PolyVision.exe
    echo ===============================================
    echo.

    if exist "models" (
        :: Copy models/ into dist\PolyVision\ so it sits next to the exe
        echo Copying models folder into dist\PolyVision\...
        xcopy /E /I /Y "models" "dist\PolyVision\models"
        if errorlevel 1 goto fail
        echo models folder copied.
    ) else (
        echo WARNING: models folder was not found. The packaged app needs a writable models folder next to PolyVision.exe.
    )
    echo.

    echo Verifying packaged GPU retraining runtime...
    dist\PolyVision\PolyVision.exe --diagnose-retraining --require-gpu --json
    if errorlevel 1 (
        echo.
        echo ERROR: Packaged executable failed GPU retraining diagnostics.
        echo Fix missing native DLLs, hidden imports, or Detectron2 CUDA architecture coverage before distribution.
        goto fail
    )

    echo.
    echo ===============================================
    echo Build and packaged GPU diagnostics completed successfully.
    echo Keep dist\PolyVision and models in a user-writable install location.
    echo ===============================================

    :: Optional: Create a shortcut on desktop
    set /p create_shortcut="Create desktop shortcut? (y/n): "
    if /i "%create_shortcut%"=="y" (
        echo Creating desktop shortcut...
        powershell "$WshShell = New-Object -ComObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%USERPROFILE%\Desktop\PolyVision.lnk'); $Shortcut.TargetPath = '%CD%\dist\PolyVision\PolyVision.exe'; $Shortcut.WorkingDirectory = '%CD%\dist\PolyVision'; $Shortcut.IconLocation = '%CD%\ui\res\PolyVisionLogo.ico'; $Shortcut.Save()"
        echo Desktop shortcut created!
    )

) else (
    echo.
    echo ===============================================
    echo Build failed! Check the output above for errors.
    echo ===============================================
    goto fail
)

popd
pause
exit /b 0

:fail
echo.
echo ===============================================
echo Build failed.
echo ===============================================
popd
pause
exit /b 1

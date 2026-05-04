@echo off
echo ===============================================
echo Building PolyVision Executable
echo ===============================================

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Clean previous builds
echo Cleaning previous builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

:: Build the executable
echo Building executable with PyInstaller...
pyinstaller --clean PolyVision.spec

:: Check if build was successful
if exist "dist\PolyVision\PolyVision.exe" (
    echo.
    echo ===============================================
    echo Build completed successfully!
    echo Executable location: dist\PolyVision\PolyVision.exe
    echo ===============================================
    echo.

    :: Copy Models/ into dist\PolyVision\ so it sits next to the exe
    echo Copying Models folder into dist\PolyVision\...
    xcopy /E /I /Y "Models" "dist\PolyVision\Models"
    echo Models folder copied.
    echo.

    :: Optional: Create a shortcut on desktop
    set /p create_shortcut="Create desktop shortcut? (y/n): "
    if /i "%create_shortcut%"=="y" (
        echo Creating desktop shortcut...
        powershell "$WshShell = New-Object -ComObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%USERPROFILE%\Desktop\PolyVision.lnk'); $Shortcut.TargetPath = '%CD%\dist\PolyVision\PolyVision.exe'; $Shortcut.WorkingDirectory = '%CD%\dist\PolyVision'; $Shortcut.IconLocation = '%CD%\UI\res\PolyVisionLogo.ico'; $Shortcut.Save()"
        echo Desktop shortcut created!
    )

) else (
    echo.
    echo ===============================================
    echo Build failed! Check the output above for errors.
    echo ===============================================
)

pause

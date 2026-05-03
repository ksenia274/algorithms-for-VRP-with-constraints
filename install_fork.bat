@echo off
echo Activating Visual Studio 2022 Build Tools (x64)...
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
if errorlevel 1 (
    echo ERROR: Could not initialize Visual Studio environment.
    pause
    exit /b 1
)

echo.
echo Installing build tools for PyVRP fork...
cd /d "%~dp0"
call venv\Scripts\pip.exe install poetry-core meson ninja docblock
if errorlevel 1 (
    echo ERROR: Failed to install build tools.
    pause
    exit /b 1
)

echo.
echo Installing dependencies from requirements.txt...
call venv\Scripts\pip.exe install -r requirements.txt --no-build-isolation
if errorlevel 1 (
    echo.
    echo ERROR: Installation failed.
    pause
    exit /b 1
)

echo.
echo Verifying pyvrp fork installation...
call venv\Scripts\python.exe -c "import pyvrp; from pyvrp.adaptive_objective import AdaptiveObjective; from importlib.metadata import version; print('pyvrp version:', version('pyvrp')); print('AdaptiveObjective: OK')"

echo.
pause

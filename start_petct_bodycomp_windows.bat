@echo off
setlocal
cd /d "%~dp0"
title PETCT BodyComp Extractor Windows Launcher
chcp 65001 >nul

set "ENV_NAME=petct_bodycomp"
set "CONDA_CMD="
set "EXIT_CODE=0"

where conda >nul 2>nul
if %errorlevel%==0 (
    for /f "delims=" %%I in ('where conda 2^>nul') do (
        set "CONDA_CMD=%%I"
        goto :run_conda
    )
)

if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" (
    set "CONDA_CMD=%USERPROFILE%\anaconda3\condabin\conda.bat"
    goto :run_conda
)

if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" (
    set "CONDA_CMD=%USERPROFILE%\miniconda3\condabin\conda.bat"
    goto :run_conda
)

if exist "%ProgramData%\anaconda3\condabin\conda.bat" (
    set "CONDA_CMD=%ProgramData%\anaconda3\condabin\conda.bat"
    goto :run_conda
)

if exist "%ProgramData%\miniconda3\condabin\conda.bat" (
    set "CONDA_CMD=%ProgramData%\miniconda3\condabin\conda.bat"
    goto :run_conda
)

goto :run_venv

:run_conda
echo [INFO] Conda detected.
echo [INFO] Checking environment "%ENV_NAME%"...
call "%CONDA_CMD%" env list | findstr /R /C:"^%ENV_NAME%[ ]" >nul 2>nul
if not %errorlevel%==0 (
    echo [INFO] Creating conda environment "%ENV_NAME%" with Python 3.10...
    call "%CONDA_CMD%" create -y -n "%ENV_NAME%" python=3.10
    if not %errorlevel%==0 (
        echo [ERROR] Failed to create conda environment.
        set "EXIT_CODE=1"
        goto :finish
    )
)

echo [INFO] Launching PETCT BodyComp Extractor in conda environment "%ENV_NAME%"...
call "%CONDA_CMD%" run -n "%ENV_NAME%" python "%~dp0bootstrap_windows.py"
set "EXIT_CODE=%errorlevel%"
goto :finish

:run_venv
echo [INFO] Conda was not detected. Falling back to a local .venv environment.
set "PYTHON_CMD="

where py >nul 2>nul
if %errorlevel%==0 (
    set "PYTHON_CMD=py -3"
    goto :ensure_venv
)

where python >nul 2>nul
if %errorlevel%==0 (
    set "PYTHON_CMD=python"
    goto :ensure_venv
)

echo [ERROR] Python was not found.
echo Please install Anaconda/Miniconda or Python 3.10+ and run this launcher again.
set "EXIT_CODE=1"
goto :finish

:ensure_venv
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo [INFO] Creating local .venv with %PYTHON_CMD%...
    %PYTHON_CMD% -m venv "%~dp0.venv"
    if not %errorlevel%==0 (
        echo [ERROR] Failed to create local .venv.
        set "EXIT_CODE=1"
        goto :finish
    )
)

echo [INFO] Launching PETCT BodyComp Extractor in local .venv...
"%~dp0.venv\Scripts\python.exe" "%~dp0bootstrap_windows.py"
set "EXIT_CODE=%errorlevel%"
goto :finish

:finish
if not "%EXIT_CODE%"=="0" (
    echo.
    echo Program exited with errors. Press any key to close this window.
    pause >nul
)
exit /b %EXIT_CODE%

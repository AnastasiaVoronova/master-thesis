@echo off
setlocal

set PY_VER=3.12.0
set REPO_DIR=%USERPROFILE%\eNPS_App
set ENV_DIR=%REPO_DIR%\runtime
set PY_DIR=%ENV_DIR%\python
set VENV_DIR=%ENV_DIR%\venv
set ZIP_URL=https://www.python.org/ftp/python/%PY_VER%/python-%PY_VER%-amd64.zip
set ZIP_FILE=%ENV_DIR%\python.zip

echo ============================================================================
echo Setting up eNPS_App in: %REPO_DIR%
echo ============================================================================

if not exist "%REPO_DIR%" mkdir "%REPO_DIR%"
if not exist "%ENV_DIR%" mkdir "%ENV_DIR%"
if exist "%PY_DIR%" rmdir /s /q "%PY_DIR%"
if exist "%VENV_DIR%" rmdir /s /q "%VENV_DIR%"
mkdir "%PY_DIR%"

echo Downloading Python %PY_VER%...
powershell -Command "Invoke-WebRequest -Uri '%ZIP_URL%' -OutFile '%ZIP_FILE%'"

echo Extracting Python...
powershell -Command "Expand-Archive '%ZIP_FILE%' '%PY_DIR%'"

for /d %%i in ("%PY_DIR%\python-*") do (
    xcopy "%%i\*" "%PY_DIR%\" /s /e /y >nul
    rmdir /s /q "%%i"
)

if exist "%ZIP_FILE%" del /f /q "%ZIP_FILE%"
if exist "%PY_DIR%\python312._pth" del "%PY_DIR%\python312._pth"

echo Verifying Python...
"%PY_DIR%\python.exe" --version || (
    echo ERROR: Python failed to start
    pause
    exit /b 1
)

echo Installing pip...
"%PY_DIR%\python.exe" -m ensurepip
"%PY_DIR%\python.exe" -m pip install --upgrade pip setuptools wheel

echo Creating virtual environment...
"%PY_DIR%\python.exe" -m venv "%VENV_DIR%"

call "%VENV_DIR%\Scripts\activate.bat"

echo Installing dependencies...
IF EXIST "%REPO_DIR%\requirements.txt" (
    pip install -r "%REPO_DIR%\requirements.txt"
) ELSE (
    echo WARNING: requirements.txt not found in %REPO_DIR%
)

REM --- Ярлык на Рабочем столе ---
set "SHORTCUT_NAME=eNPS_App.lnk"
set "TARGET_FILE=%REPO_DIR%\run.bat"
set "VBS_PATH=%REPO_DIR%\CreateShortcut.vbs"

if not exist "%TARGET_FILE%" (
    echo ERROR: target file not found: %TARGET_FILE%
    pause
    exit /b 1
)

cscript //nologo "%VBS_PATH%" "%TARGET_FILE%" "%SHORTCUT_NAME%"

set "SHORTCUT_PATH=%USERPROFILE%\Desktop\%SHORTCUT_NAME%"
if exist "%SHORTCUT_PATH%" (
    echo Link created: %SHORTCUT_PATH%
) else (
    echo Link NOT created. Run app with eNPS_App\run.bat
)

echo ============================================================================
echo Set Up Finished: %REPO_DIR%
echo ============================================================================
pause

@echo off

set ROOT=%CD%
set VENV_DIR=%ROOT%\runtime\venv
set PYTHONPATH=%ROOT%

call "%VENV_DIR%\Scripts\activate.bat"

echo Running eNPS_App...
echo Don't close this window for eNPS_App to work

start /b powershell -Command "while (-not (Invoke-WebRequest -Uri 'http://localhost:8000' -UseBasicParsing -ErrorAction SilentlyContinue)) { Start-Sleep 1 }; Start-Process 'http://localhost:8000'"

"%VENV_DIR%\Scripts\uvicorn.exe" app:app --host 127.0.0.1 --port 8000

call "%VENV_DIR%\Scripts\deactivate.bat"

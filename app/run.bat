@echo off

set ROOT=%CD%
set VENV_DIR=%ROOT%\runtime\venv
set PYTHONPATH=%ROOT%

call "%VENV_DIR%\Scripts\activate.bat"

echo Running eNPS_App...
echo Don't close this window for eNPS_App to work

"%VENV_DIR%\Scripts\python.exe" "%ROOT%\app.py"

call "%VENV_DIR%\Scripts\deactivate.bat"

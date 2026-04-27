@echo off
setlocal

set ROOT=%USERPROFILE%\eNPS_App
set SHORTCUT=%USERPROFILE%\Desktop\eNPS_App.lnk

echo ============================================================================
echo Deleting eNPS_App...
echo ============================================================================

if exist "%ROOT%" (
    echo Deleting project folder...
    rmdir /s /q "%ROOT%"
) else (
    echo Project folder not found: skipping.
)

if exist "%SHORTCUT%" (
    echo Deleting link from Desktop...
    del /f /q "%SHORTCUT%"
) else (
    echo Link not found: skipping.
)

echo ============================================================================
echo eNPS_App successfully deleted
echo ============================================================================
pause

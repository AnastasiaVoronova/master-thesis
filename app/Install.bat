@echo off
setlocal

set ROOT=%USERPROFILE%\eNPS_App
set ZIP_URL=https://github.com/AnastasiaVoronova/master-thesis/archive/refs/heads/main.zip
set ZIP_FILE=%ROOT%\archive.zip

echo ============================================================================
echo Downloading eNPS App from GitHub
echo Target folder: %ROOT%
echo ============================================================================

if not exist "%ROOT%" mkdir "%ROOT%"

set "IS_EMPTY=1"
for /f %%i in ('dir "%ROOT%" /a /b') do set "IS_EMPTY=0"

if "%IS_EMPTY%"=="1" (
    set "DOWNLOADED=1"
    if exist "%ZIP_FILE%" del /f /q "%ZIP_FILE%"
    echo Downloading archive...
    powershell -Command "Invoke-WebRequest -Uri '%ZIP_URL%' -OutFile '%ZIP_FILE%'"
    echo Extracting archive...
    powershell -Command "Expand-Archive -LiteralPath '%ZIP_FILE%' -DestinationPath '%ROOT%' -Force"
    if exist "%ZIP_FILE%" del /f /q "%ZIP_FILE%"

    rem === Flatten the -main folder ===
    for /d %%D in ("%ROOT%\*-main" "%ROOT%\*-master") do (
        echo Moving contents from %%~nxD...

        rem === Delete everything except the app folder ===
        echo Removing files and folders outside app...
        for /f "delims=" %%F in ('dir "%%D" /a-d /b') do del /f /q "%%D\%%F"
        for /d %%S in ("%%D\*") do (
            if /i not "%%~nxS"=="app" rmdir /s /q "%%S"
        )

        rem === Move app contents up to ROOT ===
        echo Moving app contents to %ROOT%...
        xcopy "%%D\app\*" "%ROOT%\" /s /e /y >nul
        rmdir /s /q "%%D"
    )
) else (
    echo Folder is not empty. Uninstall eNPS_App first.
    set "DOWNLOADED=0"
)

echo ============================================================================
echo Repository ready in: %ROOT%
echo ============================================================================

if "%DOWNLOADED%"=="1" (
    if exist "%ROOT%\SetUp.bat" (
        echo Running SetUp.bat...
        call "%ROOT%\SetUp.bat"
    )
)

pause
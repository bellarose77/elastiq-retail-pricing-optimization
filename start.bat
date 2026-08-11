@echo off
setlocal EnableExtensions

cd /d "%~dp0"
set "APP_ROOT=%CD%"
set "WEB_DIR=%APP_ROOT%\app\frontend"
set "RUNTIME_DIR=%APP_ROOT%\.runtime"
set "npm_config_cache=%RUNTIME_DIR%\npm-cache"

echo.
echo ============================================================
echo   ELASTIQ Retail Pricing Decision Studio
echo   DEEP ANALYSIS RUNNER - BUILD 1.2.0
echo ============================================================
echo.

where node >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Node.js is not installed.
  echo Install Node.js 18 or newer from https://nodejs.org and run start.bat again.
  pause
  exit /b 1
)

where npm >nul 2>nul
if errorlevel 1 (
  echo [ERROR] npm is not available. Reinstall Node.js and run start.bat again.
  pause
  exit /b 1
)

if not exist "%RUNTIME_DIR%" mkdir "%RUNTIME_DIR%"

if /I "%~1"=="--refresh-data" goto refresh_data
goto install_web

:refresh_data
echo [1/3] Preparing the Python analytics environment...
where py >nul 2>nul
if not errorlevel 1 (
  set "PYTHON_CMD=py -3"
) else (
  where python >nul 2>nul
  if errorlevel 1 (
    echo [ERROR] Python 3 is required for --refresh-data.
    echo Install Python 3.10 or newer, or run start.bat without --refresh-data.
    pause
    exit /b 1
  )
  set "PYTHON_CMD=python"
)

if not exist "%RUNTIME_DIR%\venv\Scripts\python.exe" (
  %PYTHON_CMD% -m venv "%RUNTIME_DIR%\venv"
  if errorlevel 1 goto failed
)

call "%RUNTIME_DIR%\venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto failed
call "%RUNTIME_DIR%\venv\Scripts\python.exe" scripts\run_pipeline.py
if errorlevel 1 goto failed

:install_web
echo [2/3] Preparing the web application...
cd /d "%WEB_DIR%"
if not exist "node_modules\.package-lock.json" (
  call npm --cache "%npm_config_cache%" ci --no-audit --no-fund
  if errorlevel 1 goto failed
)

echo [3/3] Starting ELASTIQ at http://127.0.0.1:5173
echo Keep this window open while using the application.
echo Press Ctrl+C to stop it.
echo.

node -e "const net=require('net');const s=net.createServer();s.once('error',()=>process.exit(1));s.once('listening',()=>s.close(()=>process.exit(0)));s.listen(5173,'127.0.0.1')"
if errorlevel 1 (
  echo [ERROR] Port 5173 is already in use.
  echo Close the earlier ELASTIQ command window, then run start.bat again.
  pause
  exit /b 1
)

if /I not "%~1"=="--no-browser" (
  start "" /b powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 2; Start-Process 'http://127.0.0.1:5173/?build=DEEP-ANALYSIS-1.2.0'"
)

call npm run dev -- --host 127.0.0.1 --strictPort --force
exit /b %errorlevel%

:failed
echo.
echo [ERROR] ELASTIQ could not be started. Review the message above.
pause
exit /b 1

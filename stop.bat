@echo off
REM stop.bat — Windows shutdown script for rvc-web

setlocal enabledelayedexpansion
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM ---------------------------------------------------------------------------
REM 1. Kill worker subprocesses first
REM ---------------------------------------------------------------------------
echo [stop] Killing worker subprocesses...
wmic process where "commandline like '%_realtime_worker%'" delete >nul 2^&1
wmic process where "commandline like '%spawn_main%'" delete >nul 2^&1

REM ---------------------------------------------------------------------------
REM 2. Attempt clean shutdown via API
REM    Use PowerShell as curl may not be available on all Windows systems
REM ---------------------------------------------------------------------------
echo [stop] Attempting clean shutdown via API...
powershell -Command "try { Invoke-WebRequest -Uri 'http://localhost:8000/api/realtime/stop' -Method POST -Body '{\"session_id\":\"all\"}' -ContentType 'application/json' -UseBasicParsing | Out-Null } catch { }" >nul 2^&1
timeout /t 1 /nobreak >nul

REM ---------------------------------------------------------------------------
REM 3. Kill by port
REM ---------------------------------------------------------------------------
echo [stop] Killing processes on ports 8000, 3000, 3001...
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":8000 "') do (
    if not "%%a"=="0" taskkill /F /PID %%a >nul 2^&1
)
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":3000 "') do (
    if not "%%a"=="0" taskkill /F /PID %%a >nul 2^&1
)
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":3001 "') do (
    if not "%%a"=="0" taskkill /F /PID %%a >nul 2^&1
)

REM ---------------------------------------------------------------------------
REM 4. Kill training/inference processes
REM ---------------------------------------------------------------------------
echo [stop] Cleaning up training processes...
wmic process where "commandline like '%train.py%'" delete >nul 2^&1
wmic process where "commandline like '%extract_f0_print%'" delete >nul 2^&1
wmic process where "commandline like '%extract_feature_print%'" delete >nul 2^&1
wmic process where "commandline like '%preprocess.py%'" delete >nul 2^&1
wmic process where "commandline like '%beatrice_trainer%'" delete >nul 2^&1

REM ---------------------------------------------------------------------------
REM 5. Clean up PID files
REM ---------------------------------------------------------------------------
if exist .pids\backend.pid del /f .pids\backend.pid 2>nul
if exist .pids\frontend.pid del /f .pids\frontend.pid 2>nul

echo [stop] Done.
endlocal
exit /b 0

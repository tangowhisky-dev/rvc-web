@echo off
REM start.bat — Windows startup script for rvc-web
REM Supports: Windows 10/11 with NVIDIA GPU (CUDA) or CPU fallback
REM Requires: Python in PATH, pnpm in PATH, conda (optional)

setlocal enabledelayedexpansion
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

set ERROR_CODE=0

REM ---------------------------------------------------------------------------
REM 1. Detect compute device
REM ---------------------------------------------------------------------------
echo [start] Platform: Windows

python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" > .tmp_device.txt 2>nul
set /p RVC_DEVICE=<.tmp_device.txt
if exist ".tmp_device.txt" del .tmp_device.txt 2>nul
if "!RVC_DEVICE!"=="" set RVC_DEVICE=cpu
echo [start] Compute device: !RVC_DEVICE!

REM ---------------------------------------------------------------------------
REM 2. Universal threading env vars
REM ---------------------------------------------------------------------------
set KMP_DUPLICATE_LIB_OK=TRUE
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1

REM ---------------------------------------------------------------------------
REM 3. CUDA-specific setup
REM ---------------------------------------------------------------------------
if "!RVC_DEVICE!"=="cuda" (
    if "!CUDA_VISIBLE_DEVICES!"=="" (
        set CUDA_VISIBLE_DEVICES=0
        echo [start] CUDA_VISIBLE_DEVICES defaulted to 0 ^(set externally to override^)
    ) else (
        echo [start] CUDA_VISIBLE_DEVICES=!CUDA_VISIBLE_DEVICES! ^(from environment^)
    )
)

REM ---------------------------------------------------------------------------
REM 4. Kill stale processes on ports 8000 and 3000
REM ---------------------------------------------------------------------------
echo [start] Killing stale processes on ports 8000 and 3000...
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":8000 "') do (
    if not "%%a"=="0" taskkill /F /PID %%a >nul 2^&1
)
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":3000 "') do (
    if not "%%a"=="0" taskkill /F /PID %%a >nul 2^&1
)
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":3001 "') do (
    if not "%%a"=="0" taskkill /F /PID %%a >nul 2^&1
)

REM Kill dangling training/inference processes
wmic process where "commandline like '%_realtime_worker%'" delete >nul 2^&1
wmic process where "commandline like '%spawn_main%'" delete >nul 2^&1
wmic process where "commandline like '%extract_f0_print%'" delete >nul 2^&1
wmic process where "commandline like '%extract_feature_print%'" delete >nul 2^&1
wmic process where "commandline like '%preprocess.py%'" delete >nul 2^&1

mkdir .pids 2>nul
mkdir logs 2>nul
mkdir data\profiles 2>nul

REM ---------------------------------------------------------------------------
REM 5. PROJECT_ROOT
REM ---------------------------------------------------------------------------
set PROJECT_ROOT=%CD%
echo [start] PROJECT_ROOT=%PROJECT_ROOT%

REM ---------------------------------------------------------------------------
REM 5b. Detect all IP addresses and let user choose
REM ---------------------------------------------------------------------------
echo [start] Detecting network interfaces...

REM Use Python to detect IPs (cross-platform on Windows)
python -c "
import socket, subprocess, re
ips = set()
# From socket
for i in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
    if i[4][0] != '127.0.0.1' and not i[4][0].startswith('169.254.'):
        ips.add(i[4][0])
# From ipconfig
try:
    out = subprocess.check_output(['ipconfig'], text=True, stderr=subprocess.DEVNULL)
    for m in re.findall(r'IPv4 Address.*?:\s*([\d.]+)', out):
        if m != '127.0.0.1' and not m.startswith('169.254.'):
            ips.add(m)
except:
    pass
print('localhost')
for ip in sorted(ips):
    print(ip)
" > .tmp_ip_choice.txt 2>nul

set /a IP_COUNT=0
for /f "usebackq delims=" %%a in (.tmp_ip_choice.txt) do (
    set /a IP_COUNT+=1
    set "IP_!IP_COUNT!=%%a"
)
if exist ".tmp_ip_choice.txt" del .tmp_ip_choice.txt 2>nul

if !IP_COUNT! leq 1 (
    echo [start] No network interfaces found — using localhost only
    set NEXT_PUBLIC_API_URL=http://localhost:8000
) else (
    echo.
    echo   Available network interfaces for frontend access:
    echo.
    for /l %%i in (1,1,!IP_COUNT!) do (
        if "%%i"=="1" (
            echo     %%i^) !IP_%%i! (local only)
        ) else (
            echo     %%i^) !IP_%%i!
        )
    )
    echo.

    set /p CHOICE="  Select interface [default: 1]: "
    if "!CHOICE!"=="" set CHOICE=1

    if !CHOICE! geq 1 if !CHOICE! leq !IP_COUNT! (
        for /l %%j in (1,1,!IP_COUNT!) do (
            if %%j equ !CHOICE! set "NEXT_PUBLIC_API_URL=http://!IP_%%j!:8000"
        )
        echo [start] Selected: !NEXT_PUBLIC_API_URL!
    ) else (
        echo [start] Invalid selection — falling back to localhost
        set NEXT_PUBLIC_API_URL=http://localhost:8000
    )
    echo.
)

REM ---------------------------------------------------------------------------
REM 6. Print device summary
REM ---------------------------------------------------------------------------
echo [start] -----------------------------------------------
echo [start] Device summary
echo [start]   OS:      Windows
echo [start]   Device:  !RVC_DEVICE!
python -c "
import torch, sys
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    vram = p.total_memory / 1024**3
    print(f'[start]   GPU:  {p.name}  ({vram:.1f} GB VRAM)')
else:
    try:
        import psutil
        ram = psutil.virtual_memory().total / 1024**3
        print(f'[start]   CPU:  {torch.get_num_threads()} threads  ({ram:.1f} GB RAM)')
    except:
        print(f'[start]   CPU:  {torch.get_num_threads()} threads')
" 2>nul
echo [start] -----------------------------------------------

REM ---------------------------------------------------------------------------
REM 7. Start backend
REM    Uses conda env RVC_CONDA_ENV if conda is available, else plain python
REM ---------------------------------------------------------------------------
if "!RVC_CONDA_ENV!"=="" set RVC_CONDA_ENV=rvc

where conda >nul 2^&1
if !errorlevel! equ 0 (
    echo [start] Starting backend via conda env !RVC_CONDA_ENV!...
    start "rvc-backend" /b conda run --no-capture-output -n !RVC_CONDA_ENV! python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
) else (
    echo [start] Starting backend via python...
    start "rvc-backend" /b python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
)

REM Get actual PID of started process using PowerShell
for /f %%a in ('powershell -Command "Get-Process | Where-Object {$_.ProcessName -eq 'python'} | Select-Object -First 1 -ExpandProperty Id"') do (
    set BACKEND_PID=%%a
)
if defined BACKEND_PID (
    echo !BACKEND_PID! > .pids\backend.pid
    echo [start] Backend PID !BACKEND_PID!
) else (
    echo [start] Backend starting... (PID unavailable)
)

REM ---------------------------------------------------------------------------
REM 8. Wait for backend to be ready (up to 30s)
REM    Use PowerShell as curl may not be available on all Windows systems
REM ---------------------------------------------------------------------------
echo [start] Waiting for backend to be ready...
set /a WAITED=0
:wait_backend
powershell -Command "Start-Sleep -Seconds 1" >nul
powershell -Command "try { (New-Object Net.WebClient).DownloadString('http://localhost:8000/health') } catch { exit 1 }" >nul 2^&1
if !errorlevel! equ 0 goto backend_ready
set /a WAITED+=1
if !WAITED! lss 30 goto wait_backend
echo [start] Warning: backend did not respond within 30s — continuing anyway

:backend_ready
if !WAITED! lss 30 (
    echo [start] Backend ready after !WAITED!s
)

REM ---------------------------------------------------------------------------
REM 9. Start frontend
REM ---------------------------------------------------------------------------
echo [start] Starting frontend...
if not exist "frontend\node_modules" (
    echo [start] Installing frontend dependencies...
    cd frontend
    call pnpm install
    cd ..
)
start "rvc-frontend" /b cmd /c "cd frontend ^&^& pnpm dev"

REM Get actual PID of started process
for /f %%a in ('powershell -Command "Get-Process | Where-Object {$_.MainWindowTitle -eq 'rvc-frontend'} | Select-Object -First 1 -ExpandProperty Id" 2^$null') do (
    set FRONTEND_PID=%%a
)
if defined FRONTEND_PID (
    echo !FRONTEND_PID! > .pids\frontend.pid
    echo [start] Frontend PID !FRONTEND_PID!
) else (
    echo [start] Frontend starting... (PID unavailable)
)

REM ---------------------------------------------------------------------------
REM 10. Wait for frontend (up to 60s)
REM ---------------------------------------------------------------------------
echo [start] Waiting for frontend to compile...
set /a WAITED=0
set FRONTEND_URL=
:wait_frontend
powershell -Command "Start-Sleep -Seconds 1" >nul
powershell -Command "try { (New-Object Net.WebClient).DownloadString('http://localhost:3000') } catch { exit 1 }" >nul 2^&1
if !errorlevel! equ 0 ( set FRONTEND_URL=http://localhost:3000 ^& goto frontend_ready )
powershell -Command "try { (New-Object Net.WebClient).DownloadString('http://localhost:3001') } catch { exit 1 }" >nul 2^&1
if !errorlevel! equ 0 ( set FRONTEND_URL=http://localhost:3001 ^& goto frontend_ready )
set /a WAITED+=1
if !WAITED! lss 60 goto wait_frontend
set FRONTEND_URL=http://localhost:3000
echo [start] Warning: frontend did not respond within 60s

:frontend_ready
echo [start] Done — !FRONTEND_URL!
start "" "!FRONTEND_URL!"

endlocal
exit /b !ERROR_CODE!

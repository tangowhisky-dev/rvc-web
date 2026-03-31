@echo off
REM start.bat — Windows startup script for rvc-web
REM Supports: Windows 10/11 with NVIDIA GPU (CUDA) or CPU fallback
REM Requires: Python in PATH, pnpm in PATH, conda (optional)

setlocal enabledelayedexpansion
cd /d "%~dp0"

echo [start] Platform: Windows

REM ---------------------------------------------------------------------------
REM 1. Detect compute device
REM ---------------------------------------------------------------------------
python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" > .tmp_device.txt 2>nul
set /p RVC_DEVICE=<.tmp_device.txt
del .tmp_device.txt 2>nul
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
    if not "%%a"=="0" taskkill /F /PID %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":3000 "') do (
    if not "%%a"=="0" taskkill /F /PID %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":3001 "') do (
    if not "%%a"=="0" taskkill /F /PID %%a >nul 2>&1
)

REM Kill dangling training/inference processes
taskkill /F /IM python.exe /FI "WINDOWTITLE eq train.py*" >nul 2>&1
wmic process where "commandline like '%_realtime_worker%'" delete >nul 2>&1
wmic process where "commandline like '%extract_f0_print%'" delete >nul 2>&1
wmic process where "commandline like '%extract_feature_print%'" delete >nul 2>&1
wmic process where "commandline like '%preprocess.py%'" delete >nul 2>&1

mkdir .pids 2>nul
mkdir logs 2>nul
mkdir data\profiles 2>nul

REM ---------------------------------------------------------------------------
REM 5. PROJECT_ROOT
REM ---------------------------------------------------------------------------
set PROJECT_ROOT=%CD%
echo [start] PROJECT_ROOT=%PROJECT_ROOT%

REM ---------------------------------------------------------------------------
REM 6. Print device summary
REM ---------------------------------------------------------------------------
echo [start] -----------------------------------------------
python -c "
import torch, sys
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    vram = p.total_memory / 1024**3
    print(f'[start]   GPU:  {p.name}  ({vram:.1f} GB VRAM)')
else:
    import psutil
    ram = psutil.virtual_memory().total / 1024**3
    print(f'[start]   CPU:  {torch.get_num_threads()} threads  ({ram:.1f} GB RAM)')
" 2>nul
echo [start] -----------------------------------------------

REM ---------------------------------------------------------------------------
REM 7. Start backend
REM    Uses conda env RVC_CONDA_ENV if conda is available, else plain python
REM ---------------------------------------------------------------------------
if "!RVC_CONDA_ENV!"=="" set RVC_CONDA_ENV=rvc

where conda >nul 2>&1
if !errorlevel!==0 (
    echo [start] Starting backend via conda env !RVC_CONDA_ENV!...
    start "rvc-backend" /b conda run --no-capture-output -n !RVC_CONDA_ENV! python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
) else (
    echo [start] Starting backend via python...
    start "rvc-backend" /b python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
)

REM Write a placeholder PID file (Windows start /b doesn't easily expose PID)
echo %RANDOM% > .pids\backend.pid
echo [start] Backend starting...

REM ---------------------------------------------------------------------------
REM 8. Wait for backend to be ready (up to 30s)
REM ---------------------------------------------------------------------------
echo [start] Waiting for backend to be ready...
set /a WAITED=0
:wait_backend
timeout /t 1 /nobreak >nul
curl -s http://localhost:8000/health >nul 2>&1
if !errorlevel!==0 goto backend_ready
set /a WAITED+=1
if !WAITED! lss 30 goto wait_backend
echo [start] Warning: backend did not respond within 30s — continuing anyway

:backend_ready
echo [start] Backend ready after !WAITED!s

REM ---------------------------------------------------------------------------
REM 9. Start frontend
REM ---------------------------------------------------------------------------
echo [start] Starting frontend...
if not exist "frontend\node_modules" (
    echo [start] Installing frontend dependencies...
    cd frontend && pnpm install && cd ..
)
start "rvc-frontend" /b cmd /c "cd frontend && pnpm dev"
echo %RANDOM% > .pids\frontend.pid
echo [start] Frontend starting...

REM ---------------------------------------------------------------------------
REM 10. Wait for frontend (up to 60s)
REM ---------------------------------------------------------------------------
echo [start] Waiting for frontend to compile...
set /a WAITED=0
set FRONTEND_URL=
:wait_frontend
timeout /t 1 /nobreak >nul
curl -s http://localhost:3000 >nul 2>&1
if !errorlevel!==0 ( set FRONTEND_URL=http://localhost:3000 && goto frontend_ready )
curl -s http://localhost:3001 >nul 2>&1
if !errorlevel!==0 ( set FRONTEND_URL=http://localhost:3001 && goto frontend_ready )
set /a WAITED+=1
if !WAITED! lss 60 goto wait_frontend
set FRONTEND_URL=http://localhost:3000
echo [start] Warning: frontend did not respond within 60s

:frontend_ready
echo [start] Done — !FRONTEND_URL!
start !FRONTEND_URL!

endlocal

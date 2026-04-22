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

REM ---------------------------------------------------------------------------
REM 1. Resolve Python interpreter
REM    Priority: conda env 'rvc' -> uv venv %USERPROFILE%\.rvc
REM ---------------------------------------------------------------------------
set ENV_NAME=%RVC_ENV_NAME%
if "!ENV_NAME!"=="" set ENV_NAME=rvc

set RVC_PYTHON=
set RVC_ENV_SOURCE=

REM Try conda first
where conda >nul 2>&1
if !errorlevel! equ 0 (
    for /f "delims=" %%b in ('conda info --base 2^>nul') do set CONDA_BASE=%%b
    if defined CONDA_BASE (
        if exist "!CONDA_BASE!\envs\!ENV_NAME!\python.exe" (
            set "RVC_PYTHON=!CONDA_BASE!\envs\!ENV_NAME!\python.exe"
            set "RVC_ENV_SOURCE=conda env '!ENV_NAME!'"
        )
    )
)

REM Fall back to uv venv at %USERPROFILE%\.rvc
if not defined RVC_PYTHON (
    if exist "%USERPROFILE%\.!ENV_NAME!\Scripts\python.exe" (
        set "RVC_PYTHON=%USERPROFILE%\.!ENV_NAME!\Scripts\python.exe"
        set "RVC_ENV_SOURCE=uv venv '%USERPROFILE%\.!ENV_NAME!'"
    )
)

if not defined RVC_PYTHON (
    echo [start] ERROR: No Python environment named '!ENV_NAME!' found.
    echo [start]        Tried:
    echo [start]          conda: ^(conda base^)\envs\!ENV_NAME!\python.exe
    echo [start]          uv:    %USERPROFILE%\.!ENV_NAME!\Scripts\python.exe
    echo [start]        Run install.bat to create the environment.
    exit /b 1
)
echo [start] Python: !RVC_PYTHON!  ^(via !RVC_ENV_SOURCE!^)

REM Detect compute device using the resolved interpreter
"!RVC_PYTHON!" -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" > .tmp_device.txt 2>nul
set /p RVC_DEVICE=<.tmp_device.txt
if exist ".tmp_device.txt" del .tmp_device.txt 2>nul
if "!RVC_DEVICE!"=="" set RVC_DEVICE=cpu
echo [start] Compute device: !RVC_DEVICE!

REM ---------------------------------------------------------------------------
REM 2. Asset pre-check
REM    Verify all required model weights are present before launching anything.
REM ---------------------------------------------------------------------------
echo [start] Checking assets...
echo.

set ASSET_ERRORS=0

REM Helper: check a single file exists
REM Usage: call :check_file path label
goto :asset_checks

:check_file
  if not exist "%~1" (
    echo   [MISSING] %~2
    echo             ^-> %~1
    set /a ASSET_ERRORS+=1
  ) else (
    echo   [OK]      %~2
  )
goto :eof

REM Helper: check a directory has at least N files matching a pattern
REM Usage: call :check_dir dir pattern label min
:check_dir
  if not exist "%~1\" (
    echo   [MISSING] %~3 ^(directory not found^)
    echo             ^-> %~1
    set /a ASSET_ERRORS+=1
    goto :eof
  )
  set _COUNT=0
  for %%f in ("%~1\%~2") do set /a _COUNT+=1
  if !_COUNT! LSS %~4 (
    echo   [MISSING] %~3 ^(found !_COUNT!/%~4 files matching %~2^)
    echo             ^-> %~1
    set /a ASSET_ERRORS+=1
  ) else (
    echo   [OK]      %~3 ^(!_COUNT! files^)
  )
goto :eof

:asset_checks
echo   -- RVC core (required for all inference + training) --
call :check_file "assets\hubert\hubert_base.pt"      "HuBERT base model"
call :check_file "assets\rmvpe\rmvpe.pt"             "RMVPE F0 estimator (PyTorch)"
call :check_file "assets\rmvpe\rmvpe.onnx"           "RMVPE F0 estimator (ONNX)"
call :check_dir  "assets\pretrained_v2" "*.pth" "RVC v2 pretrained weights" 12
call :check_dir  "assets\pretrained"   "*.pth" "RVC v1 pretrained weights" 12

echo.
echo   -- Embedder (default: spin-v2) --
call :check_file "assets\spin-v2\config.json"        "SPIN-v2 embedder config"
call :check_file "assets\spin-v2\pytorch_model.bin"  "SPIN-v2 embedder weights"

echo.
echo   -- Speaker analysis (ECAPA-TDNN) --
call :check_file "assets\ecapa\ecapa_tdnn.pt"        "ECAPA-TDNN speaker encoder"

echo.
echo   -- Beatrice 2 (required for B2 pipeline) --
call :check_file "assets\beatrice2\pretrained\122_checkpoint_03000000.pt"                    "B2 phone extractor checkpoint"
call :check_file "assets\beatrice2\pretrained\104_3_checkpoint_00300000.pt"                  "B2 pitch estimator checkpoint"
call :check_file "assets\beatrice2\pretrained\151_checkpoint_libritts_r_200_02750000.pt.gz"  "B2 base model (LibriTTS-R 200)"
call :check_file "assets\beatrice2\utmos\utmos22_strong_step7459_v1.pt"                      "UTMOS MOS scorer"
call :check_dir  "assets\beatrice2\ir"    "*.flac" "Beatrice 2 impulse responses (IR)"  20
call :check_dir  "assets\beatrice2\noise" "*.flac" "Beatrice 2 background noise files"  20

echo.
if !ASSET_ERRORS! GTR 0 (
    echo   +-----------------------------------------------------------------+
    echo   ^|  !ASSET_ERRORS! required asset^(s^) missing -- cannot start          ^|
    echo   +-----------------------------------------------------------------+
    echo.
    echo   To fix:
    echo.
    echo     RVC weights ^(manual download^):
    echo       https://huggingface.co/lj1995/VoiceConversionWebUI
    echo       hubert_base.pt          -^> assets\hubert\
    echo       rmvpe.pt + rmvpe.onnx   -^> assets\rmvpe\
    echo       pretrained\*.pth        -^> assets\pretrained\
    echo       pretrained_v2\*.pth     -^> assets\pretrained_v2\
    echo       spin-v2\ directory      -^> assets\spin-v2\
    echo       ecapa_tdnn.pt           -^> assets\ecapa\
    echo.
    echo     Beatrice 2 weights ^(automated, downloads 20 IR + 20 noise files^):
    echo       "!RVC_PYTHON!" -m backend.beatrice2.download_assets
    echo.
    exit /b 1
)
echo   All assets present -- proceeding.
echo.

REM ---------------------------------------------------------------------------
REM 3. Universal threading env vars (renumbered from 2)
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
REM 4. Kill stale processes on ports 8000, 3000, 3001
REM ---------------------------------------------------------------------------
echo [start] Killing stale processes on ports 8000, 3000, 3001...
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
wmic process where "commandline like '%beatrice_trainer%'" delete >nul 2^&1

mkdir .pids 2>nul
mkdir logs 2>nul
mkdir data\profiles 2>nul

REM ---------------------------------------------------------------------------
REM 4b. Sweep temp/ — delete files older than 1 hour
REM ---------------------------------------------------------------------------
mkdir temp 2>nul
echo [start] Sweeping temp/ (files older than 1h)...
powershell -Command "Get-ChildItem -Path 'temp' -File | Where-Object { $_.LastWriteTime -lt (Get-Date).AddHours(-1) } | Remove-Item -Force" >nul 2^&1

REM ---------------------------------------------------------------------------
REM 5. PROJECT_ROOT
REM ---------------------------------------------------------------------------
set PROJECT_ROOT=%CD%
echo [start] PROJECT_ROOT=%PROJECT_ROOT%

REM ---------------------------------------------------------------------------
REM 6. Detect all IP addresses and let user choose
REM ---------------------------------------------------------------------------
echo [start] Detecting network interfaces...

python -c "
import socket, subprocess, re
ips = set()
for i in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
    ip = i[4][0]
    if ip != '127.0.0.1' and not ip.startswith('169.254.'):
        ips.add(ip)
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
" > .tmp_ip_list.txt 2>nul

set /a IP_COUNT=0
for /f "usebackq delims=" %%a in (.tmp_ip_list.txt) do (
    set /a IP_COUNT+=1
    set "IP_!IP_COUNT!=%%a"
)
if exist ".tmp_ip_list.txt" del .tmp_ip_list.txt 2>nul

if !IP_COUNT! leq 1 (
    echo [start] No network interfaces found — using localhost only
    set NEXT_PUBLIC_API_URL=http://localhost:8000
    set SELECTED_IP=localhost
) else (
    echo.
    echo   Available network interfaces for frontend access:
    echo.
    for /l %%i in (1,1,!IP_COUNT!) do (
        if "!IP_%%i!"=="localhost" (
            echo     %%i^) !IP_%%i! (local only)
        ) else (
            echo     %%i^) !IP_%%i!
        )
    )
    echo.

    REM Default to first non-localhost IP (same logic as start.sh)
    set DEFAULT_CHOICE=1
    for /l %%i in (1,1,!IP_COUNT!) do (
        if not "!IP_%%i!"=="localhost" (
            if "!DEFAULT_CHOICE!"=="1" set DEFAULT_CHOICE=%%i
        )
    )

    set /p CHOICE="  Select interface [default: !DEFAULT_CHOICE!]: "
    if "!CHOICE!"=="" set CHOICE=!DEFAULT_CHOICE!

    set SELECTED_IP=localhost
    if !CHOICE! geq 1 if !CHOICE! leq !IP_COUNT! (
        for /l %%j in (1,1,!IP_COUNT!) do (
            if %%j equ !CHOICE! (
                set "SELECTED_IP=!IP_%%j!"
                set "NEXT_PUBLIC_API_URL=http://!IP_%%j!:8000"
            )
        )
        echo [start] Selected: !SELECTED_IP! ^→ NEXT_PUBLIC_API_URL=!NEXT_PUBLIC_API_URL!
    ) else (
        echo [start] Invalid selection — falling back to localhost
        set NEXT_PUBLIC_API_URL=http://localhost:8000
        set SELECTED_IP=localhost
    )
    echo.
)

REM ---------------------------------------------------------------------------
REM 7. Print device summary
REM ---------------------------------------------------------------------------
echo [start] -----------------------------------------------
echo [start] Device summary
echo [start]   OS:      Windows
echo [start]   Device:  !RVC_DEVICE!
"!RVC_PYTHON!" -c "
import torch
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    vram = p.total_memory / 1024**3
    print(f'[start]   GPU:     {p.name}  ({vram:.1f} GB VRAM)')
else:
    try:
        import psutil
        ram = psutil.virtual_memory().total / 1024**3
        print(f'[start]   CPU:     {torch.get_num_threads()} threads  ({ram:.1f} GB RAM)')
    except:
        print(f'[start]   CPU:     {torch.get_num_threads()} threads')
" 2>nul
echo [start] -----------------------------------------------

REM ---------------------------------------------------------------------------
REM 8. Start backend
REM ---------------------------------------------------------------------------
echo [start] Starting backend via !RVC_ENV_SOURCE!...
start "rvc-backend" /b "!RVC_PYTHON!" -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000

REM Get backend PID
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
REM 9. Wait for backend to be ready (up to 30s)
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
REM 10. Start frontend — pass --hostname so Next.js dev server treats the
REM     selected IP as its canonical origin (prevents cross-origin warnings)
REM ---------------------------------------------------------------------------
echo [start] Starting frontend...
if not exist "frontend\node_modules" (
    echo [start] Installing frontend dependencies...
    cd frontend
    call pnpm install
    cd ..
)

if "!SELECTED_IP!"=="localhost" (
    start "rvc-frontend" /b cmd /c "cd frontend && pnpm dev"
) else (
    start "rvc-frontend" /b cmd /c "cd frontend && pnpm dev --hostname !SELECTED_IP!"
)

REM Get frontend PID
for /f %%a in ('powershell -Command "Get-Process | Where-Object {$_.MainWindowTitle -eq 'rvc-frontend'} | Select-Object -First 1 -ExpandProperty Id" 2^>$null') do (
    set FRONTEND_PID=%%a
)
if defined FRONTEND_PID (
    echo !FRONTEND_PID! > .pids\frontend.pid
    echo [start] Frontend PID !FRONTEND_PID!
) else (
    echo [start] Frontend starting... (PID unavailable)
)

REM ---------------------------------------------------------------------------
REM 11. Wait for frontend (up to 60s)
REM ---------------------------------------------------------------------------
echo [start] Waiting for frontend to compile...
set /a WAITED=0
set FRONTEND_URL=
:wait_frontend
powershell -Command "Start-Sleep -Seconds 1" >nul
powershell -Command "try { (New-Object Net.WebClient).DownloadString('http://localhost:3000') } catch { exit 1 }" >nul 2^&1
if !errorlevel! equ 0 ( set "FRONTEND_URL=http://localhost:3000" & goto frontend_ready )
powershell -Command "try { (New-Object Net.WebClient).DownloadString('http://localhost:3001') } catch { exit 1 }" >nul 2^&1
if !errorlevel! equ 0 ( set "FRONTEND_URL=http://localhost:3001" & goto frontend_ready )
set /a WAITED+=1
if !WAITED! lss 60 goto wait_frontend
set FRONTEND_URL=http://localhost:3000
echo [start] Warning: frontend did not respond within 60s

:frontend_ready
echo [start] Done — !FRONTEND_URL!
start "" "!FRONTEND_URL!"

endlocal
exit /b !ERROR_CODE!

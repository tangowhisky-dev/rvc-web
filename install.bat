@echo off
REM =============================================================================
REM install.bat — rvc-web environment setup for Windows
REM Supports: Windows 10/11, NVIDIA GPU (CUDA) or CPU fallback
REM Requires: uv (https://github.com/astral-sh/uv), Git, Node.js >=18 in PATH
REM   Install uv:  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
REM =============================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

set ENV_NAME=%RVC_ENV_NAME%
if "!ENV_NAME!"=="" set ENV_NAME=rvc
set ENV_DIR=%USERPROFILE%\.!ENV_NAME!
set PYTHON_VERSION=3.11
set NODE_MIN=18

echo [install] Platform: Windows

REM ---------------------------------------------------------------------------
REM 1. Check prerequisites
REM ---------------------------------------------------------------------------
where uv >nul 2>&1
if !errorlevel! NEQ 0 (
    echo [install] ERROR: uv not found.
    echo [install] Install it with:
    echo [install]   powershell -c "irm https://astral.sh/uv/install.ps1 ^| iex"
    echo [install] Then restart your terminal and re-run install.bat.
    exit /b 1
)
for /f "tokens=*" %%v in ('uv --version') do echo [install] %%v ^✓

where git >nul 2>&1 || (echo [install] ERROR: git not found. Install Git for Windows: https://git-scm.com && exit /b 1)

REM Node.js check
where node >nul 2>&1
if !errorlevel!==0 (
    for /f "tokens=1 delims=v." %%v in ('node --version') do set NODE_MAJOR=%%v
    set NODE_MAJOR=!NODE_MAJOR:v=!
    if !NODE_MAJOR! LSS !NODE_MIN! (
        echo [install] WARNING: Node.js !NODE_MAJOR! found but ^>=%NODE_MIN% required.
    ) else (
        echo [install] Node.js !NODE_MAJOR! ^✓
    )
) else (
    echo [install] WARNING: Node.js not found. Install from https://nodejs.org
)

REM pnpm
where pnpm >nul 2>&1
if !errorlevel! NEQ 0 (
    echo [install] Installing pnpm via npm...
    npm install -g pnpm
)

REM ---------------------------------------------------------------------------
REM 2. Detect CUDA
REM ---------------------------------------------------------------------------
set TORCH_INDEX=
set FAISS_PKG=faiss-cpu
set ONNX_PKG=onnxruntime
set CUDA_DEVICE=cpu

where nvidia-smi >nul 2>&1
if !errorlevel!==0 (
    for /f "tokens=9" %%v in ('nvidia-smi ^| findstr "CUDA Version"') do set CUDA_VER=%%v
    if not "!CUDA_VER!"=="" (
        echo [install] CUDA !CUDA_VER! detected
        for /f "tokens=1 delims=." %%m in ("!CUDA_VER!") do set CUDA_MAJOR=%%m
        for /f "tokens=2 delims=." %%n in ("!CUDA_VER!") do set CUDA_MINOR=%%n
        if !CUDA_MAJOR! GEQ 13 (
            set TORCH_INDEX=
            set ONNX_PKG=onnxruntime-gpu
            set CUDA_DEVICE=cuda
            echo [install] CUDA 13.x -- PyPI now ships native CUDA 13 builds
        ) else if !CUDA_MAJOR! EQU 12 (
            if !CUDA_MINOR! GEQ 8 (
                set TORCH_INDEX=https://download.pytorch.org/whl/cu128
                set ONNX_PKG=onnxruntime-gpu
                set CUDA_DEVICE=cuda
                echo [install] Using cu128 index ^(CUDA 12.8^)
            ) else if !CUDA_MINOR! GEQ 6 (
                set TORCH_INDEX=https://download.pytorch.org/whl/cu126
                set ONNX_PKG=onnxruntime-gpu
                set CUDA_DEVICE=cuda
                echo [install] Using cu126 index ^(CUDA 12.6^)
            ) else (
                echo [install] CUDA !CUDA_VER! not supported ^(need 12.6+, 12.8, or 13.x^) -- using CPU build
            )
        ) else (
            echo [install] CUDA !CUDA_VER! not supported ^(need 12.6+, 12.8, or 13.x^) -- using CPU build
        )
    )
) else (
    echo [install] No NVIDIA GPU detected -- using CPU build
)

REM ---------------------------------------------------------------------------
REM 2b. System audio notes (Windows)
REM     sounddevice bundles PortAudio in its wheel on Windows -- no extra install.
REM     soundfile bundles libsndfile in its wheel on Windows -- no extra install.
REM     ffmpeg is NOT required -- `av` bundles its own FFmpeg shared libs.
REM ---------------------------------------------------------------------------
echo [install] Note: PortAudio and libsndfile are bundled in their wheels on Windows -- no system install needed.

REM ---------------------------------------------------------------------------
REM 3. Create / update uv virtual environment at %USERPROFILE%\.rvc
REM ---------------------------------------------------------------------------
echo [install] Setting up uv environment '!ENV_NAME!' at !ENV_DIR!...

if exist "!ENV_DIR!\Scripts\python.exe" (
    echo [install] Environment already exists -- reusing
    echo [install] ^(Delete !ENV_DIR! to start fresh^)
) else (
    uv venv "!ENV_DIR!" --python !PYTHON_VERSION!
    if !errorlevel! NEQ 0 (
        echo [install] ERROR: uv venv creation failed
        exit /b 1
    )
    echo [install] Created uv venv at !ENV_DIR!
)

set RVC_PYTHON=!ENV_DIR!\Scripts\python.exe
if not exist "!RVC_PYTHON!" (
    echo [install] ERROR: Python interpreter not found at !RVC_PYTHON!
    exit /b 1
)
for /f "tokens=*" %%v in ('"!RVC_PYTHON!" --version') do echo [install] %%v

set UV=uv pip install --python "!RVC_PYTHON!"

REM ---------------------------------------------------------------------------
REM 4. Install PyTorch
REM ---------------------------------------------------------------------------
echo [install] Installing PyTorch...
if "!TORCH_INDEX!"=="" (
    call !UV! torch torchaudio
) else (
    echo [install]   Using index: !TORCH_INDEX!
    call !UV! torch torchaudio --index-url !TORCH_INDEX!
)
if !errorlevel! NEQ 0 (echo [install] ERROR: PyTorch install failed && exit /b 1)

REM ---------------------------------------------------------------------------
REM 5. Install faiss
REM ---------------------------------------------------------------------------
echo [install] Installing !FAISS_PKG!...
call !UV! !FAISS_PKG!
if !errorlevel! NEQ 0 (echo [install] ERROR: faiss install failed && exit /b 1)

REM ---------------------------------------------------------------------------
REM 6. Install fairseq (One-sixth fork)
REM ---------------------------------------------------------------------------
echo [install] Installing build prerequisites for fairseq...
call !UV! Cython numpy
if !errorlevel! NEQ 0 (echo [install] ERROR: Cython/numpy install failed && exit /b 1)

echo [install] Installing fairseq ^(One-sixth fork^)...
call !UV! "fairseq @ git+https://github.com/One-sixth/fairseq.git"
if !errorlevel! NEQ 0 (echo [install] ERROR: fairseq install failed && exit /b 1)

REM ---------------------------------------------------------------------------
REM 7. Install remaining requirements
REM ---------------------------------------------------------------------------
echo [install] Installing remaining Python dependencies...

set TMPFILE=%TEMP%\rvc_req_%RANDOM%.txt
type nul > "!TMPFILE!"
for /f "usebackq tokens=* delims=" %%L in (requirements.txt) do (
    set LINE=%%L
    if not "!LINE!"=="" (
        echo !LINE! | findstr /r "^#" >nul 2>&1 && goto :skip_line
        echo !LINE! | findstr /i "^torch " >nul 2>&1 && goto :skip_line
        echo !LINE! | findstr /i "^torchaudio" >nul 2>&1 && goto :skip_line
        echo !LINE! | findstr /i "^faiss" >nul 2>&1 && goto :skip_line
        echo !LINE! | findstr /i "^fairseq" >nul 2>&1 && goto :skip_line
        echo !LINE! | findstr /i "^onnxruntime" >nul 2>&1 && goto :skip_line
        echo !LINE! | findstr /i "sys_platform platform_machine" >nul 2>&1 && goto :skip_line
        echo !LINE! >> "!TMPFILE!"
    )
    :skip_line
)

call !UV! -r "!TMPFILE!"
if !errorlevel! NEQ 0 (echo [install] ERROR: requirements install failed && exit /b 1)
del "!TMPFILE!" 2>nul

echo [install] Installing !ONNX_PKG!...
call !UV! !ONNX_PKG!
if !errorlevel! NEQ 0 (echo [install] ERROR: onnxruntime install failed && exit /b 1)

REM ---------------------------------------------------------------------------
REM 8. Install frontend dependencies
REM ---------------------------------------------------------------------------
echo [install] Installing frontend dependencies...
cd frontend
pnpm install
if !errorlevel! NEQ 0 (echo [install] ERROR: pnpm install failed && exit /b 1)
cd ..

REM ---------------------------------------------------------------------------
REM 9. Verify key imports
REM ---------------------------------------------------------------------------
echo [install] Verifying key imports...
"!RVC_PYTHON!" -c "
import sys
failures = []
checks = [
    'torch','torchaudio','fairseq','faiss',
    'fastapi','uvicorn','librosa','sounddevice',
    'soundfile','numpy','scipy','sklearn',
    'torchcrepe','torchfcpe','av',
    'noisereduce','parselmouth','pyworld',
    'tensorboard','psutil',
]
for mod in checks:
    try:
        __import__(mod)
        print(f'  ok  {mod}')
    except ImportError as e:
        print(f'  ERR {mod}: {e}')
        failures.append(mod)
import torch
print(f'  torch={torch.__version__}')
print(f'  CUDA: {torch.cuda.is_available()}')
if failures:
    print(f'FAILED: {failures}', file=sys.stderr)
    sys.exit(1)
"
if !errorlevel! NEQ 0 (
    echo [install] ERROR: import verification failed -- check output above
    exit /b 1
)

REM ---------------------------------------------------------------------------
REM Done
REM ---------------------------------------------------------------------------
echo.
echo [install] =====================================================
echo [install]   Installation complete
echo [install]   venv   : !ENV_DIR!
echo [install]   Python : !RVC_PYTHON!
echo [install]   Device : !CUDA_DEVICE!
echo [install]   Run:  start.bat
echo [install] =====================================================

endlocal

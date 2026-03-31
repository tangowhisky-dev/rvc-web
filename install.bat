@echo off
REM =============================================================================
REM install.bat — rvc-web environment setup for Windows
REM Supports: Windows 10/11, NVIDIA GPU (CUDA) or CPU fallback
REM Requires: conda (Miniconda/Anaconda), Git, Node.js >=18 in PATH
REM =============================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

set ENV_NAME=%RVC_CONDA_ENV%
if "!ENV_NAME!"=="" set ENV_NAME=rvc
set PYTHON_VERSION=3.11
set NODE_MIN=18

echo [install] Platform: Windows

REM ---------------------------------------------------------------------------
REM 1. Check prerequisites
REM ---------------------------------------------------------------------------
where conda >nul 2>&1 || (echo [install] ERROR: conda not found. Install Miniconda: https://docs.conda.io/en/latest/miniconda.html && exit /b 1)
where git   >nul 2>&1 || (echo [install] ERROR: git not found. Install Git for Windows: https://git-scm.com && exit /b 1)

REM Node.js check
where node >nul 2>&1
if !errorlevel!==0 (
    for /f "tokens=1 delims=v." %%v in ('node --version') do set NODE_MAJOR=%%v
    REM strip leading v if present
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
        REM Parse major version
        for /f "tokens=1 delims=." %%m in ("!CUDA_VER!") do set CUDA_MAJOR=%%m
        if !CUDA_MAJOR! GEQ 12 (
            set TORCH_INDEX=https://download.pytorch.org/whl/cu128
            set FAISS_PKG=faiss-gpu
            set ONNX_PKG=onnxruntime-gpu
            set CUDA_DEVICE=cuda
            echo [install] Using CUDA 12.x PyTorch build
        ) else if !CUDA_MAJOR! EQU 11 (
            set TORCH_INDEX=https://download.pytorch.org/whl/cu118
            set FAISS_PKG=faiss-gpu
            set ONNX_PKG=onnxruntime-gpu
            set CUDA_DEVICE=cuda
            echo [install] Using CUDA 11.8 PyTorch build
        ) else (
            echo [install] CUDA !CUDA_VER! too old ^(need ^>=11.8^) — using CPU build
        )
    )
) else (
    echo [install] No NVIDIA GPU detected — using CPU build
)

REM ---------------------------------------------------------------------------
REM 3. Create / update conda environment
REM ---------------------------------------------------------------------------
conda env list | findstr /C:"!ENV_NAME!" >nul 2>&1
if !errorlevel!==0 (
    echo [install] Conda env '!ENV_NAME!' already exists — updating...
    call conda install -n !ENV_NAME! python=!PYTHON_VERSION! -y -q
) else (
    echo [install] Creating conda env '!ENV_NAME!' with Python !PYTHON_VERSION!...
    call conda create -n !ENV_NAME! python=!PYTHON_VERSION! -y -q
)

REM Activate environment
call conda activate !ENV_NAME!
if !errorlevel! NEQ 0 (
    echo [install] Trying conda init approach...
    call conda run -n !ENV_NAME! python --version >nul 2>&1 || (echo [install] ERROR: Could not activate conda env && exit /b 1)
    REM Use conda run for remaining steps
    set USE_CONDA_RUN=1
)

REM ---------------------------------------------------------------------------
REM 4. Install uv
REM ---------------------------------------------------------------------------
echo [install] Installing uv...
call conda run -n !ENV_NAME! pip install uv -q

REM ---------------------------------------------------------------------------
REM 5. Install PyTorch
REM ---------------------------------------------------------------------------
echo [install] Installing PyTorch...
if "!TORCH_INDEX!"=="" (
    call conda run -n !ENV_NAME! python -m uv pip install torch torchaudio -q
) else (
    call conda run -n !ENV_NAME! python -m uv pip install torch torchaudio --index-url !TORCH_INDEX! -q
)

REM ---------------------------------------------------------------------------
REM 6. Install faiss
REM ---------------------------------------------------------------------------
echo [install] Installing !FAISS_PKG!...
call conda run -n !ENV_NAME! python -m uv pip install !FAISS_PKG! -q

REM ---------------------------------------------------------------------------
REM 7. Install fairseq (One-sixth fork)
REM ---------------------------------------------------------------------------
echo [install] Installing build prerequisites for fairseq...
call conda run -n !ENV_NAME! python -m uv pip install Cython numpy -q

echo [install] Installing fairseq ^(One-sixth fork^)...
call conda run -n !ENV_NAME! python -m uv pip install "fairseq @ git+https://github.com/One-sixth/fairseq.git" -q

REM ---------------------------------------------------------------------------
REM 8. Install remaining requirements
REM ---------------------------------------------------------------------------
echo [install] Installing remaining Python dependencies...

REM Write a filtered requirements file (exclude already-handled packages)
set TMPFILE=%TEMP%\rvc_req_%RANDOM%.txt
type nul > !TMPFILE!
for /f "usebackq tokens=* delims=" %%L in (requirements.txt) do (
    set LINE=%%L
    REM Skip comments, blank lines, and platform-handled packages
    echo !LINE! | findstr /r "^#" >nul && goto :skip_line
    echo !LINE! | findstr /r "^$" >nul && goto :skip_line
    echo !LINE! | findstr /i "^torch ^torchaudio ^faiss ^fairseq ^onnxruntime sys_platform platform_machine" >nul && goto :skip_line
    echo !LINE! >> !TMPFILE!
    :skip_line
)

call conda run -n !ENV_NAME! python -m uv pip install -r !TMPFILE! -q
del !TMPFILE! 2>nul

REM onnxruntime — platform-specific
echo [install] Installing !ONNX_PKG!...
call conda run -n !ENV_NAME! python -m uv pip install !ONNX_PKG! -q

REM ---------------------------------------------------------------------------
REM 9. Install frontend dependencies
REM ---------------------------------------------------------------------------
echo [install] Installing frontend dependencies...
cd frontend
pnpm install
cd ..

REM ---------------------------------------------------------------------------
REM 10. Verify
REM ---------------------------------------------------------------------------
echo [install] Verifying key imports...
call conda run -n !ENV_NAME! python -c "
import sys
failures = []
checks = ['torch','torchaudio','fairseq','faiss','fastapi','uvicorn','librosa','sounddevice','soundfile','numpy','scipy','sklearn','torchcrepe','torchfcpe']
for mod in checks:
    try:
        __import__(mod)
        print(f'  OK  {mod}')
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

REM ---------------------------------------------------------------------------
REM Done
REM ---------------------------------------------------------------------------
echo.
echo [install] =====================================================
echo [install]   Installation complete
echo [install]   Conda env : !ENV_NAME!
echo [install]   Device    : !CUDA_DEVICE!
echo [install]   Run:  start.bat
echo [install] =====================================================

endlocal

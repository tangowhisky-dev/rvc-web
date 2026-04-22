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
            set TORCH_INDEX=https://download.pytorch.org/whl/cu130
            set ONNX_PKG=onnxruntime-gpu
            set CUDA_DEVICE=cuda
            echo [install] Using cu130 index ^(CUDA 13.x^)
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
                echo [install] CUDA !CUDA_VER! not supported ^(need 12.6+, 12.8, or 13.x^) — using CPU build
            )
        ) else (
            echo [install] CUDA !CUDA_VER! not supported ^(need 12.6+, 12.8, or 13.x^) — using CPU build
        )
    )
) else (
    echo [install] No NVIDIA GPU detected — using CPU build
)

REM ---------------------------------------------------------------------------
REM 2b. Install system audio libraries via conda
REM
REM     NOTE: ffmpeg is NOT required as a system package.
REM       - The `av` Python package bundles its own FFmpeg shared libs.
REM       - pydub has been removed (offline output is always WAV now).
REM     sounddevice on Windows bundles PortAudio in the wheel, so only
REM     libsndfile is needed via conda for soundfile to link correctly.
REM ---------------------------------------------------------------------------
echo [install] Installing libsndfile via conda...
call conda install -n !ENV_NAME! -c conda-forge libsndfile -y -q

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
    set USE_CONDA_RUN=1
)

REM ---------------------------------------------------------------------------
REM 4. Resolve uv
REM ---------------------------------------------------------------------------
set UV_BIN=
where uv >nul 2>&1
if !errorlevel!==0 (
    for /f "delims=" %%p in ('where uv') do (
        set UV_BIN=%%p
        goto :uv_found
    )
)
if exist "%USERPROFILE%\.cargo\bin\uv.exe"  set UV_BIN=%USERPROFILE%\.cargo\bin\uv.exe & goto :uv_found
if exist "%USERPROFILE%\.local\bin\uv.exe"  set UV_BIN=%USERPROFILE%\.local\bin\uv.exe & goto :uv_found
if exist "%LOCALAPPDATA%\uv\bin\uv.exe"     set UV_BIN=%LOCALAPPDATA%\uv\bin\uv.exe    & goto :uv_found

echo [install] uv not found globally — installing into conda env...
call conda run -n !ENV_NAME! pip install uv -q
set UV=conda run -n !ENV_NAME! python -m uv pip
goto :uv_done

:uv_found
for /f "delims=" %%p in ('conda run -n !ENV_NAME! python -c "import sys; print(sys.executable)"') do set CONDA_PYTHON=%%p
echo [install] uv found: !UV_BIN! ^(targeting !CONDA_PYTHON!^)
set UV=!UV_BIN! pip --python !CONDA_PYTHON!

:uv_done

REM ---------------------------------------------------------------------------
REM 5. Install PyTorch
REM ---------------------------------------------------------------------------
echo [install] Installing PyTorch...
if "!TORCH_INDEX!"=="" (
    call !UV! install torch torchaudio -q
) else (
    echo [install]   Using index: !TORCH_INDEX!
    call !UV! install torch torchaudio --index-url !TORCH_INDEX! -q
)

REM ---------------------------------------------------------------------------
REM 6. Install faiss
REM ---------------------------------------------------------------------------
echo [install] Installing !FAISS_PKG!...
call !UV! install !FAISS_PKG! -q

REM ---------------------------------------------------------------------------
REM 7. Install fairseq (One-sixth fork)
REM ---------------------------------------------------------------------------
echo [install] Installing build prerequisites for fairseq...
call !UV! install Cython numpy -q

echo [install] Installing fairseq ^(One-sixth fork^)...
call !UV! install "fairseq @ git+https://github.com/One-sixth/fairseq.git" -q

REM ---------------------------------------------------------------------------
REM 8. Install remaining requirements
REM ---------------------------------------------------------------------------
echo [install] Installing remaining Python dependencies...

set TMPFILE=%TEMP%\rvc_req_%RANDOM%.txt
type nul > !TMPFILE!
for /f "usebackq tokens=* delims=" %%L in (requirements.txt) do (
    set LINE=%%L
    echo !LINE! | findstr /r "^#" >nul && goto :skip_line
    echo !LINE! | findstr /r "^$" >nul && goto :skip_line
    echo !LINE! | findstr /i "^torch ^torchaudio ^faiss ^fairseq ^onnxruntime sys_platform platform_machine" >nul && goto :skip_line
    echo !LINE! >> !TMPFILE!
    :skip_line
)

call !UV! install -r !TMPFILE! -q
del !TMPFILE! 2>nul

echo [install] Installing !ONNX_PKG!...
call !UV! install !ONNX_PKG! -q

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

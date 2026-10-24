@echo off
rem One-time setup after cloning, on Windows 10/11 x64 from cmd (no WSL, CPU-only or NVIDIA GPU):
rem   1. Python 3.12 venv with requirements.txt; with an NVIDIA GPU also CUDA 12.6 PyTorch so the
rem      classifier fine-tunes on the GPU (set TORCH=cpu to keep the CPU build)
rem   2. llama.cpp b11840 CPU build -> tools\llama-b11840\, and with an NVIDIA GPU also the
rem      CUDA 12.4 build + CUDA runtime -> tools\llama-b11840-cuda\
rem   3. the three GGUF models -> models\ (about 4.2 GiB)
rem Needs: uv (winget install --id=astral-sh.uv -e). curl and tar come with Windows 10 and 11.
rem Safe to rerun: finished steps are skipped.
setlocal
cd /d "%~dp0.."
set BUILD=b11840
set BASE=https://github.com/ggml-org/llama.cpp/releases/download/%BUILD%
set PYTHONUTF8=1
set PY=.venv\Scripts\python.exe

where uv >nul 2>nul || (echo uv not found: install it with "winget install --id=astral-sh.uv -e", then open a new cmd & exit /b 1)
set GPU=0
nvidia-smi -L >nul 2>nul && set GPU=1
if %GPU%==1 (nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader) else (echo No NVIDIA GPU found: CPU-only setup)

echo == Python environment
if not exist %PY% (uv venv -p 3.12 .venv || exit /b 1)
uv pip install -p .venv --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt || exit /b 1
if %GPU%==1 if /i not "%TORCH%"=="cpu" call :cuda_torch || exit /b 1
%PY% -c "import torch; print('PyTorch', torch.__version__, '| CUDA:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available')" || exit /b 1

echo == llama.cpp %BUILD% (CPU)
call :install_llama tools\llama-%BUILD% llama-%BUILD%-bin-win-cpu-x64.zip || exit /b 1
if %GPU%==1 (
    echo == llama.cpp %BUILD% ^(CUDA 12.4^)
    call :install_llama tools\llama-%BUILD%-cuda llama-%BUILD%-bin-win-cuda-12.4-x64.zip cudart-llama-bin-win-cuda-12.4-x64.zip || exit /b 1
)

echo == models
%PY% serving\download_models.py || exit /b 1

echo.
echo Setup done. Run the app: scripts\run_app.cmd
exit /b 0

:cuda_torch
rem requirements.txt pins torch==X+cpu; install the same version built for CUDA 12.6
for /f "tokens=2 delims==+" %%v in ('findstr /b "torch==" requirements.txt') do set TORCH_VERSION=%%v
uv pip install -p .venv --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu126 "torch==%TORCH_VERSION%+cu126"
exit /b %errorlevel%

:install_llama
rem install_llama <dest> <zip>...: unpack release zips (flat: exe and DLLs at the root) into <dest>
if exist "%~1\llama-server.exe" (
    echo already installed in %~1
) else (
    if not exist "%~1" mkdir "%~1"
    call :fetch "%~1" %2 || exit /b 1
    if not "%~3"=="" call :fetch "%~1" %3 || exit /b 1
)
"%~1\llama-server.exe" --list-devices
exit /b %errorlevel%

:fetch
echo downloading %~2
curl.exe -fL --progress-bar -o "%TEMP%\%~2" "%BASE%/%~2" || exit /b 1
tar.exe -xf "%TEMP%\%~2" -C "%~1" || exit /b 1
del "%TEMP%\%~2"
exit /b 0

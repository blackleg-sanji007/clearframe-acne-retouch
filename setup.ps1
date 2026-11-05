param([switch]$CpuOnly)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$uvExe = Join-Path $PSScriptRoot '.tools\uv.exe'
$env:UV_CACHE_DIR = Join-Path $PSScriptRoot '.cache\uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $PSScriptRoot '.runtime'
if (-not (Test-Path -LiteralPath $uvExe)) {
    New-Item -ItemType Directory -Force .tools | Out-Null
    Invoke-WebRequest -Uri 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' -OutFile '.tools\uv.zip'
    Expand-Archive -LiteralPath '.tools\uv.zip' -DestinationPath '.tools' -Force
}
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    & $uvExe python install 3.12 --no-bin
    if ($LASTEXITCODE -ne 0) { throw 'Python download failed.' }
    & $uvExe venv --python 3.12 .venv
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
}
$torchIndex = if ($CpuOnly) { 'https://download.pytorch.org/whl/cpu' } else { 'https://download.pytorch.org/whl/cu128' }
& $uvExe pip install --python .venv\Scripts\python.exe torch==2.7.1 torchvision==0.22.1 --index-url $torchIndex
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed.' }
& $uvExe pip install --python .venv\Scripts\python.exe -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& .\.venv\Scripts\python.exe check_environment.py
if ($LASTEXITCODE -ne 0) { throw 'Environment check failed.' }
Write-Host 'Next: .venv\Scripts\python.exe download_assets.py --deepfill --dataset'

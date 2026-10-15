$ErrorActionPreference = 'Stop'
$taskRepo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location -LiteralPath $taskRepo
[void][IO.Directory]::CreateDirectory((Join-Path $taskRepo '.cache/native-position-audit'))
$env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'

$compilerVars = & $env:ComSpec /d /s /c 'call "C:\Program Files\Visual Studio\2022\VC\Auxiliary\Build\vcvars64.bat" >nul && set'
if ($LASTEXITCODE -ne 0) { throw 'Cannot initialize MSVC.' }
foreach ($line in $compilerVars) {
    if ($line -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process') }
}
$cuda = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4'
$nvcc = Join-Path $cuda 'bin/nvcc.exe'
& $nvcc -O3 -std=c++20 --default-stream per-thread --threads 0 --split-compile 0 -Xcompiler /Zc:preprocessor,/MT,/EHsc -gencode arch=compute_120,code=sm_120 -c tools/native-position-audit/verify-prefix.cu -o .cache/native-position-audit/verify-prefix.obj
if ($LASTEXITCODE -ne 0) { throw 'Prefix validation compilation failed.' }
& $nvcc .cache/native-position-audit/verify-prefix.obj build/tensorboard.obj build/revision.obj -L "$cuda\lib\x64" cublas.lib cuda.lib -o .cache/native-position-audit/verify-prefix.exe
if ($LASTEXITCODE -ne 0) { throw 'Prefix validation link failed.' }

$ErrorActionPreference = 'Stop'
$taskRepo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location -LiteralPath $taskRepo
[void][IO.Directory]::CreateDirectory((Join-Path $taskRepo '.cache/native-position-audit'))
$env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'

$vc = 'C:\Program Files\Visual Studio\2022\VC\Auxiliary\Build\vcvars64.bat'
$compilerVars = & $env:ComSpec /d /s /c ('call "' + $vc + '" >nul && set')
if ($LASTEXITCODE -ne 0) { throw 'Cannot initialize MSVC.' }
foreach ($line in $compilerVars) {
    if ($line -match '^([^=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
    }
}
& cl.exe /nologo /O2 /std:c++20 /EHsc /MT tools/native-position-audit/replay.cpp /Fe:.cache/native-position-audit/replay.exe /Fo:.cache/native-position-audit/replay.obj
if ($LASTEXITCODE -ne 0) { throw 'Replay compilation failed.' }
& ./.cache/native-position-audit/replay.exe
if ($LASTEXITCODE -ne 0) { throw 'Sampler replay failed.' }

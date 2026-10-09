# Native CUDA C++ build. Python is used only to prepare/download assets.
[CmdletBinding()]
param(
    [ValidatePattern('^(86|89|120)(,(86|89|120))*$')]
    [string]$CudaArch = '86,120',
    [switch]$Run,
    [string[]]$TrainingArgs = @()
)
$ErrorActionPreference = 'Stop'
if ($TrainingArgs.Count -and -not $Run) { throw 'TrainingArgs requires -Run.' }
$outDir = Join-Path $PSScriptRoot 'build'
[void][IO.Directory]::CreateDirectory($outDir)
# Use a scratch index so untracked source is recorded without changing staging.
$revisionHead = git -C $PSScriptRoot rev-parse HEAD
if ($LASTEXITCODE -ne 0) { throw 'Cannot record the Git revision.' }
$revisionIndex = Join-Path ([IO.Path]::GetTempPath()) "qwen-revision-$PID.index"
$revisionDiff = Join-Path $outDir 'revision.diff'
$priorGitIndex = $env:GIT_INDEX_FILE
try {
    $realIndex = git -C $PSScriptRoot rev-parse --path-format=absolute --git-path index
    Copy-Item -LiteralPath $realIndex -Destination $revisionIndex -Force
    $env:GIT_INDEX_FILE = $revisionIndex
    git -C $PSScriptRoot add -A
    if ($LASTEXITCODE -ne 0) { throw 'Cannot record the working tree.' }
    git -C $PSScriptRoot diff --cached --no-color --no-ext-diff "--output=$revisionDiff" HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Cannot record the experiment diff.' }
} finally {
    $env:GIT_INDEX_FILE = $priorGitIndex
    Remove-Item -LiteralPath $revisionIndex -ErrorAction SilentlyContinue
}
$revisionBytes = [Collections.Generic.List[byte]]::new([Text.Encoding]::UTF8.GetBytes($revisionHead.Trim()))
$patchBytes = [IO.File]::ReadAllBytes($revisionDiff)
if ($patchBytes.Length) {
    $revisionBytes.AddRange([Text.Encoding]::UTF8.GetBytes("`n`n"))
    $revisionBytes.AddRange($patchBytes)
}
[IO.File]::WriteAllBytes((Join-Path $outDir 'revision.txt'), $revisionBytes.ToArray())
$revisionSource = '#include <cstddef>' + "`n" + 'extern const unsigned char qwen_revision[] = {' + ($revisionBytes -join ',') + "};`nextern const size_t qwen_revision_size = " + $revisionBytes.Count + ";`n"
[IO.File]::WriteAllText((Join-Path $outDir 'revision.cpp'), $revisionSource, [Text.UTF8Encoding]::new($false))
$cuda = if ($env:QWEN_CUDA) { $env:QWEN_CUDA } elseif ($env:CUDA_PATH) { $env:CUDA_PATH } else {
    Join-Path $env:ProgramFiles 'NVIDIA GPU Computing Toolkit\CUDA\v13.4'
}
$nvcc = Join-Path $cuda 'bin\nvcc.exe'
if (-not (Test-Path -LiteralPath $nvcc)) { throw 'CUDA not found; set QWEN_CUDA.' }
$vc = 'C:\Program Files\Visual Studio\2022\VC\Auxiliary\Build\vcvars64.bat'
if (-not (Test-Path -LiteralPath $vc)) {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path -LiteralPath $vswhere) {
        $install = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($install) { $vc = Join-Path $install 'VC\Auxiliary\Build\vcvars64.bat' }
    }
}
$previous = @{}
try {
    if (Test-Path -LiteralPath $vc) {
        $compilerVars = & $env:ComSpec /d /s /c ('call "' + $vc + '" >nul && set')
        if ($LASTEXITCODE -ne 0) { throw 'MSVC environment setup failed.' }
        foreach ($line in $compilerVars) {
            if ($line -match '^([^=]+)=(.*)$') {
                $key, $value = $Matches[1], $Matches[2]
                $previous[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
                [Environment]::SetEnvironmentVariable($key, $value, 'Process')
            }
        }
    } elseif (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
        throw 'MSVC not found; run in a Visual Studio developer shell.'
    }
    if (-not $previous.ContainsKey('PATH')) { $previous['PATH'] = $env:PATH }
    $env:PATH = (Join-Path $cuda 'bin') + ';' + (Join-Path $cuda 'bin\x64') + ';' + $env:PATH
    $flags = @('-O3', '-std=c++20', '-lineinfo','--default-stream','per-thread', '--threads', '0', '--split-compile', '0',
        '-Xcompiler', '/Zc:preprocessor,/MT,/EHsc')
    foreach ($target in $CudaArch.Split(',')) {
        $flags += @('-gencode', "arch=compute_$target,code=sm_$target")
    }
    & $nvcc @flags -c (Join-Path $PSScriptRoot 'native\qwen.cu') -o (Join-Path $outDir 'qwen.obj')
    if ($LASTEXITCODE -ne 0) { throw 'CUDA compilation failed.' }
    & cl.exe /nologo /O2 /std:c++20 /EHsc /MT /c (Join-Path $PSScriptRoot 'native\tensorboard.cpp') "/Fo$(Join-Path $outDir 'tensorboard.obj')"
    if ($LASTEXITCODE -ne 0) { throw 'TensorBoard logger compilation failed.' }
    & cl.exe /nologo /O2 /std:c++20 /EHsc /MT /c (Join-Path $outDir 'revision.cpp') "/Fo$(Join-Path $outDir 'revision.obj')"
    if ($LASTEXITCODE -ne 0) { throw 'Revision compilation failed.' }
    $exe = Join-Path $outDir 'qwen-tune.exe'
    & $nvcc (Join-Path $outDir 'qwen.obj') (Join-Path $outDir 'tensorboard.obj') (Join-Path $outDir 'revision.obj') -L (Join-Path $cuda 'lib\x64') cublas.lib cuda.lib -o $exe
    if ($LASTEXITCODE -ne 0) { throw 'Link failed.' }
    Write-Host "Built $exe"
    if ($Run) {
        Push-Location -LiteralPath $PSScriptRoot
        try {
            & $exe @TrainingArgs
            if ($LASTEXITCODE -ne 0) { throw "Training exited with code $LASTEXITCODE." }
        } finally { Pop-Location }
    }
} finally {
    foreach ($key in $previous.Keys) { [Environment]::SetEnvironmentVariable($key, $previous[$key], 'Process') }
}

# Native Windows build; CUDA device functions stay in one translation unit.
[CmdletBinding()]
param(
    [switch]$Rebuild,
    [ValidatePattern('^(86|120)(,(86|120))*$')]
    [string]$Arch = '86,120',
    [ValidateSet(128,256,512)]
    [int]$Threads = 512,
    [ValidateSet(16,32)]
    [int]$Tile = 32,
    [switch]$Pipeline,
    [string]$OutputDirectory = '',
    [switch]$Run,
    [string[]]$TrainingArgs = @()
)
$ErrorActionPreference = 'Stop'
$totalTimer = [Diagnostics.Stopwatch]::StartNew()
if ($TrainingArgs.Count -and -not $Run) { throw 'TrainingArgs requires -Run.' }
$outDir = if ($OutputDirectory) { [IO.Path]::GetFullPath($OutputDirectory) } else {
    Join-Path $PSScriptRoot 'build'
}
[void][IO.Directory]::CreateDirectory($outDir)
$cuda134 = Join-Path $env:ProgramFiles 'NVIDIA GPU Computing Toolkit\CUDA\v13.4'
$cuda = if ($env:TURBOGPT_CUDA) { $env:TURBOGPT_CUDA } elseif (Test-Path -LiteralPath $cuda134) {
    $cuda134
} elseif ($env:CUDA_PATH) {
    $env:CUDA_PATH
} else { 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4' }
$nvcc = Join-Path $cuda 'bin\nvcc.exe'
if (-not (Test-Path -LiteralPath $nvcc)) { throw 'CUDA not found; set TURBOGPT_CUDA.' }
$flags = @('-O3', '-std=c++17', '--use_fast_math', '-lineinfo', '--threads', '0',
    '--split-compile', '0', '-Xcompiler', '/Zc:preprocessor,/MD')
$flags += @("-DTURBOGPT_THREADS=$Threads", "-DTURBOGPT_TILE=$Tile", "-DTURBOGPT_PIPELINE=$([int]$Pipeline.IsPresent)")
foreach ($target in $Arch.Split(',')) {
    $flags += @('-gencode', "arch=compute_$target,code=sm_$target")
}
$signature = $nvcc + "`n" + ($flags -join "`n")

# Follow local includes so a small edit recompiles only affected modules.
function Get-SourceDigest([string]$Source) {
    $pending = [Collections.Generic.Stack[string]]::new()
    $seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    $pending.Push((Join-Path $PSScriptRoot $Source))
    while ($pending.Count) {
        $path = [IO.Path]::GetFullPath($pending.Pop())
        if (-not $seen.Add($path)) { continue }
        $text = [IO.File]::ReadAllText($path)
        foreach ($include in [regex]::Matches($text, '(?m)^\s*#include\s+"([^"]+)"')) {
            $pending.Push((Join-Path (Split-Path $path) $include.Groups[1].Value))
        }
    }
    $bytes = [IO.MemoryStream]::new()
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        # GPU architecture/tile choices do not invalidate the ordinary C++ units.
        $unitSignature = if ($Source -eq 'cuda/runtime.cu') { $signature } else {
            'MSVC /O2 /std:c++17 /EHsc /MD'
        }
        $prefix = [Text.Encoding]::UTF8.GetBytes($unitSignature)
        $bytes.Write($prefix, 0, $prefix.Length)
        foreach ($path in ($seen | Sort-Object)) {
            $sourceBytes = [IO.File]::ReadAllBytes($path)
            $bytes.Write($sourceBytes, 0, $sourceBytes.Length)
        }
        $bytes.Position = 0
        return [BitConverter]::ToString($sha.ComputeHash($bytes)).Replace('-', '').ToLowerInvariant()
    } finally { $sha.Dispose(); $bytes.Dispose() }
}

$sources = [ordered]@{cuda='cuda/runtime.cu'; train='train.cpp'; trainer='trainer.cpp';
    dataset='dataset.cpp'; model='model.cpp'; evaluation='evaluation.cpp'; report='report.cpp'}
$manifest = Join-Path $outDir 'build.json'
$old = if (Test-Path -LiteralPath $manifest) { Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json } else { @{} }
$hashes = @{}
$objects = @()
$phases = [ordered]@{}
$compilerReady = $false
$previousEnvironment = @{}

function Invoke-Compiler([string]$Phase, [string]$Program, [string[]]$Arguments) {
    if (-not $script:compilerReady) {
        $vc = 'C:\Program Files\Visual Studio\2022\VC\Auxiliary\Build\vcvars64.bat'
        if (-not (Test-Path -LiteralPath $vc)) {
            $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
            if (Test-Path -LiteralPath $vswhere) {
                $install = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
                if ($install) { $vc = Join-Path $install 'VC\Auxiliary\Build\vcvars64.bat' }
            }
        }
        if (Test-Path -LiteralPath $vc) {
            $compilerVars = & $env:ComSpec /d /s /c ('call "' + $vc + '" >nul && set')
            if ($LASTEXITCODE -ne 0) { throw 'MSVC environment setup failed.' }
            foreach ($line in $compilerVars) {
                if ($line -match '^([^=]+)=(.*)$') {
                    $key, $value = $Matches[1], $Matches[2]
                    $previousEnvironment[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
                    [Environment]::SetEnvironmentVariable($key, $value, 'Process')
                }
            }
        } elseif (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
            throw 'MSVC not found; use a Visual Studio developer shell.'
        }
        if (-not $previousEnvironment.ContainsKey('PATH')) { $previousEnvironment['PATH'] = $env:PATH }
        $env:PATH = (Join-Path $cuda 'bin') + ';' + $env:PATH
        if (Test-Path -LiteralPath $manifest) { Remove-Item -LiteralPath $manifest }
        $script:compilerReady = $true
    }
    $phaseTimer = [Diagnostics.Stopwatch]::StartNew()
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Phase failed with exit code $LASTEXITCODE." }
    $phases[$Phase] = $phaseTimer.Elapsed.TotalSeconds
}

$exe = Join-Path $outDir 'turbogpt.exe'
$dll = Join-Path $outDir 'turbogpt.dll'
$changed = $false
$cudaChanged = $false
Push-Location -LiteralPath $outDir
try {
    foreach ($unit in $sources.Keys) {
        $hashes[$unit] = Get-SourceDigest $sources[$unit]
        $object = Join-Path $outDir "$unit.obj"
        $objects += $object
        if ($Rebuild -or $old.$unit -ne $hashes[$unit] -or -not (Test-Path -LiteralPath $object)) {
            $source = Join-Path $PSScriptRoot $sources[$unit]
            if ($unit -eq 'cuda') {
                Invoke-Compiler 'cuda_compile_seconds' $nvcc ($flags + @('-c', $source, '-o', $object))
                $cudaChanged = $true
            } else {
                Invoke-Compiler "$($unit)_compile_seconds" 'cl.exe' @(
                    '/nologo', '/O2', '/std:c++17', '/EHsc', '/MD', '/c', $source, "/Fo$object")
            }
            $changed = $true
        }
    }
    if ($changed -or -not (Test-Path -LiteralPath $exe)) {
        Invoke-Compiler 'exe_link_seconds' $nvcc (@('-Xcompiler', '/MD') + $objects + @('-o', $exe))
    }
    # The executable needs no DLL; this exposes the same GPU code to the oracle.
    if ($cudaChanged -or -not (Test-Path -LiteralPath $dll)) {
        Invoke-Compiler 'dll_link_seconds' $nvcc @('-Xcompiler', '/MD', '-shared', $objects[0], '-o', $dll)
    }
    [IO.File]::WriteAllText($manifest, ($hashes | ConvertTo-Json), [Text.UTF8Encoding]::new($false))
    $phases['build_seconds'] = $totalTimer.Elapsed.TotalSeconds
    $phases['recompiled'] = $changed
    @{build=$phases} | ConvertTo-Json -Compress
} finally {
    Pop-Location
    foreach ($key in $previousEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($key, $previousEnvironment[$key], 'Process')
    }
}
if ($Run) {
    & $exe @TrainingArgs
    $trainingExitCode = $LASTEXITCODE
    $phases['build_plus_process_seconds'] = $totalTimer.Elapsed.TotalSeconds
    @{total=$phases} | ConvertTo-Json -Compress
    if ($trainingExitCode -ne 0) { throw "Training failed with exit code $trainingExitCode." }
}

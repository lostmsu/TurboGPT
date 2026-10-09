# Windows build. cuda-oxide is Linux-only, so the crate builds and runs in a WSL 2 distro
# that has CUDA 13, clang/LLVM 21, rustup and cargo-oxide (see README). The source tree is
# copied into ~/turboGPT-rs there; the copy has no .git, so the revision is computed here.
[CmdletBinding()]
param(
    [string]$Distro = 'CudaRust',
    # Default: PTX for sm_80, which the driver JIT-compiles once for newer GPUs.
    [ValidatePattern('^(sm_\d+a?)?$')]
    [string]$Arch = '',
    [switch]$Test,
    [switch]$Run,
    [string[]]$TrainingArgs = @()
)
$ErrorActionPreference = 'Stop'
if ($TrainingArgs.Count -and -not $Run) { throw 'TrainingArgs requires -Run.' }

# HEAD, then the working-tree diff after a blank line. A scratch index includes
# untracked files without touching the real index.
function Get-Revision {
    $ErrorActionPreference = 'Continue'
    $hash = git -C $PSScriptRoot rev-parse HEAD 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $hash) { return [Text.Encoding]::UTF8.GetBytes('unknown') }
    $scratch = Join-Path ([IO.Path]::GetTempPath()) "turbogpt-revision-$PID"
    $index = "$scratch.index"
    $diff = "$scratch.diff"
    $realIndex = git -C $PSScriptRoot rev-parse --path-format=absolute --git-path index 2>$null
    $previousIndex = $env:GIT_INDEX_FILE
    try {
        if ($realIndex -and (Test-Path -LiteralPath $realIndex)) {
            Copy-Item -LiteralPath $realIndex -Destination $index -Force
        }
        $env:GIT_INDEX_FILE = $index
        git -C $PSScriptRoot add -A 2>$null
        if ($LASTEXITCODE -ne 0) { throw 'git add failed for the revision diff.' }
        git -C $PSScriptRoot diff --cached --no-color --no-ext-diff "--output=$diff" HEAD 2>$null
        if ($LASTEXITCODE -ne 0) { throw 'git diff failed for the revision diff.' }
    } finally {
        $env:GIT_INDEX_FILE = $previousIndex
        Remove-Item -LiteralPath $index -ErrorAction SilentlyContinue
    }
    $bytes = [Collections.Generic.List[byte]]::new([Text.Encoding]::UTF8.GetBytes($hash.Trim()))
    $patch = [IO.File]::ReadAllBytes($diff)
    Remove-Item -LiteralPath $diff
    if ($patch.Length) {
        $bytes.AddRange([Text.Encoding]::UTF8.GetBytes("`n`n"))
        $bytes.AddRange($patch)
    }
    return $bytes.ToArray()
}
$revision = Join-Path $PSScriptRoot '.revision'
[IO.File]::WriteAllBytes($revision, (Get-Revision))

function Invoke-Wsl([string]$Command) {
    $environment = 'source ~/.cargo/env && export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda ' +
        '&& export TURBOGPT_REVISION_FILE=$HOME/turboGPT-rs/.revision && cd ~/turboGPT-rs && '
    wsl.exe -d $Distro --exec bash -c ($environment + $Command)
    if ($LASTEXITCODE -ne 0) { throw "WSL command failed with exit code ${LASTEXITCODE}: $Command" }
}

# tar keeps modification times, so cargo rebuilds only what changed.
$excludes = '--exclude=./target --exclude=./build --exclude=./runs --exclude=./.git ' +
    '--exclude=./.venv --exclude=./.cache --exclude=./.pytest_cache --exclude=*.egg-info --exclude=__pycache__'
cmd.exe /d /c "tar -C `"$PSScriptRoot`" $excludes -cf - . | wsl.exe -d $Distro --exec bash -c `"mkdir -p ~/turboGPT-rs && tar -C ~/turboGPT-rs -xf -`""
if ($LASTEXITCODE -ne 0) { throw 'Copying the source tree into WSL failed.' }

$archFlag = if ($Arch) { " --arch $Arch" } else { '' }
Invoke-Wsl "cargo oxide build$archFlag"
if ($Test) {
    # One target at a time: cuda-oxide writes device artifacts next to Cargo.toml.
    Invoke-Wsl "cargo oxide test$archFlag -- --release --lib"
}
if ($Run) {
    $quoted = ($TrainingArgs | ForEach-Object { "'" + ($_ -replace "'", "'\''") + "'" }) -join ' '
    Invoke-Wsl "./target/release/turbogpt $quoted"
}

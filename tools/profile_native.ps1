[CmdletBinding()]
param(
    [ValidateSet('Range','Delta','Gemm','Timeline')][string]$Mode='Range',
    [int]$Device=1,
    [int]$Batch=16,
    [int]$Context=512,
    [double]$VramGiB=12,
    [switch]$CompactCounters,
    [ValidateSet('bf16','bf16x9','tf32','tf32x2','fp32')][string]$Precision='bf16',
    [string]$Output='runs/profile-native',
    [string]$Ncu='C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.3.1\ncu.bat',
    [string]$Nsys='C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.3.2\target-windows-x64\nsys.exe'
)
$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
$cuda=if($env:CUDA_PATH){$env:CUDA_PATH}else{'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4'}
$priorPath=$env:PATH
try {
    $env:PATH=(Join-Path $cuda 'bin')+';'+(Join-Path $cuda 'bin\x64')+';'+$env:PATH
    Push-Location -LiteralPath $taskRoot
    try {
        $trainer=@('build/qwen-tune.exe','--device',"$Device",'--batch',"$Batch",'--context',"$Context",
                   '--precision',$Precision,'--vram-gib',"$VramGiB",
                   '--steps','3','--profile-step','3','--eval-samples','1','--output',"$Output-train")
        [void][IO.Directory]::CreateDirectory((Split-Path -Parent (Join-Path $taskRoot $Output)))
        $provenance=@{
            mode=$Mode; command=$trainer; timestamp_utc=[DateTime]::UtcNow.ToString('o')
            executable_sha256=(Get-FileHash -LiteralPath 'build/qwen-tune.exe' -Algorithm SHA256).Hash.ToLower()
            clock_control='none'
            compact_counters=[bool]$CompactCounters
            gpu_before=(& nvidia-smi -i $Device --query-gpu=name,driver_version,utilization.gpu,memory.used,clocks.current.sm,power.draw,power.limit --format=csv)
        }
        $provenance | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath "$Output.json" -Encoding utf8
        if($Mode -eq 'Timeline') {
            & $Nsys profile --trace cuda --cuda-graph-trace node --sample none --capture-range cudaProfilerApi --capture-range-end stop --force-overwrite true --output $Output @trainer
        } else {
            $flags=@('--clock-control','none','--export',$Output,'--force-overwrite')
            if($CompactCounters) {
                $flags+=@('--metrics','sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,gpc__cycles_elapsed.avg.per_second,gpu__time_duration.sum')
            } else {
                $flags+=@('--section','SpeedOfLight')
            }
            if($Mode -eq 'Range') {
                # Application range replay supports CUDA 13's library API calls
                # and profiles the whole update with its normal kernel ordering.
                $flags+=@('--replay-mode','app-range','--launch-count','1')
            } else {
                $kernel=if($Mode -eq 'Delta'){'regex:delta_tiled_backward'}elseif($Precision -eq 'bf16'){'regex:nvjet_sm120_tss_mma_256x128x64_2_64x64x64_tmaAB_alignCD4_bz_TNNN'}elseif($Precision -in @('fp32','bf16x9')){'regex:.*simt_sgemm.*'}else{'regex:nvjet_sm120_sss_tf32_mma_256x128x32'}
                $flags+=@('--profile-from-start','off','--kernel-name',$kernel,'--launch-count','1','--section','Occupancy','--section','SchedulerStats')
            }
            & $Ncu @flags @trainer
            if($LASTEXITCODE -ne 0){throw 'Nsight Compute profiling failed.'}
            & $Ncu --import "$Output.ncu-repz" --csv --page raw | Set-Content -LiteralPath "$Output.csv" -Encoding utf8
        }
        if($LASTEXITCODE -ne 0){throw 'Profiler exited with a nonzero status.'}
    } finally {Pop-Location}
} finally {$env:PATH=$priorPath}

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [Console]::OutputEncoding
$env:PYTHONUTF8 = "1"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$outputDir = Join-Path $repoRoot "checkpoints\vqhbr\ECG_DSA_all_raw_v1"
$logDir = Join-Path $repoRoot "log\vqhbr\ECG_DSA_all_raw_v1"
$watchdogLog = Join-Path $logDir "watchdog_50.log"
$stdoutLog = Join-Path $logDir "watchdog_50_train_stdout.log"
$stderrLog = Join-Path $logDir "watchdog_50_train_stderr.log"
$restartStatePath = Join-Path $logDir "watchdog_50_restart_count.txt"
$targetEpoch = 50
$activityWindowSeconds = 120

New-Item -ItemType Directory -Force -Path $logDir | Out-Null
Set-Location $repoRoot

function Write-WatchdogLog([string]$message) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $watchdogLog -Encoding UTF8 -Value "[$timestamp] $message"
}

function Get-LatestCompletedEpoch() {
    $ckpts = Get-ChildItem -Path $outputDir -Filter "checkpoint-*.pth" -ErrorAction SilentlyContinue
    $epochs = @()
    foreach ($ckpt in $ckpts) {
        if ($ckpt.BaseName -match "^checkpoint-(\d+)$") {
            $epochs += [int]$Matches[1]
        }
    }
    if ($epochs.Count -eq 0) {
        return 0
    }
    return ($epochs | Measure-Object -Maximum).Maximum
}

function Get-TrainingProcesses() {
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -like "python*" -and
            $_.CommandLine -like "*run_vqhbr_training.py*" -and
            $_.CommandLine -like "*ECG_DSA_all_raw_v1*"
        }
}

function Has-RecentTrainingLogActivity() {
    if (-not (Test-Path $stdoutLog)) {
        return $false
    }
    $ageSeconds = ((Get-Date) - (Get-Item $stdoutLog).LastWriteTime).TotalSeconds
    return ($ageSeconds -le $activityWindowSeconds)
}

function Get-RestartCount() {
    if (Test-Path $restartStatePath) {
        try {
            return [int](Get-Content $restartStatePath -Raw)
        } catch {
            return 0
        }
    }
    return 0
}

function Set-RestartCount([int]$count) {
    Set-Content -Path $restartStatePath -Value $count -Encoding UTF8
}

function Get-BatchSizeForRestart([int]$restartCount) {
    if ($restartCount -ge 2) {
        return 4
    }
    return 6
}

function Start-TrainingProcess() {
    $restartCount = Get-RestartCount
    $batchSize = Get-BatchSizeForRestart $restartCount
    $resumePath = Join-Path $outputDir "checkpoint.pth"

    $args = @(
        "-u",
        "heartlang/cli/run_vqhbr_training.py",
        "--output_dir", "checkpoints/vqhbr/ECG_DSA_all_raw_v1",
        "--log_dir", "log/vqhbr/ECG_DSA_all_raw_v1",
        "--model", "vqhbr",
        "--pretrain_data_dirs", "datasets/ecg_datasets/ECG_DSA/pretrain_qrs/all_raw_v1",
        "--codebook_n_emd", "8192",
        "--codebook_emd_dim", "128",
        "--quantize_kmeans_init",
        "--batch_size", "$batchSize",
        "--num_workers", "0",
        "--opt", "adamw",
        "--opt_betas", "0.9", "0.99",
        "--weight_decay", "1e-4",
        "--warmup_epochs", "3",
        "--epochs", "$targetEpoch",
        "--save_ckpt_freq", "5",
        "--val_freq", "5",
        "--world_size", "1",
        "--lr", "2e-4",
        "--min_lr", "1e-6",
        "--device", "cuda"
    )

    if (Test-Path $resumePath) {
        $args += @("--resume", "checkpoints/vqhbr/ECG_DSA_all_raw_v1/checkpoint.pth", "--no_auto_resume")
    }

    $process = Start-Process `
        -FilePath "python" `
        -ArgumentList $args `
        -WorkingDirectory $repoRoot `
        -RedirectStandardOutput $stdoutLog `
        -RedirectStandardError $stderrLog `
        -WindowStyle Hidden `
        -PassThru

    Set-RestartCount ($restartCount + 1)
    Write-WatchdogLog "Started training PID=$($process.Id) batch_size=$batchSize restart_count=$restartCount"
}

Write-WatchdogLog "Watchdog started. target_epoch=$targetEpoch"

try {
    while ($true) {
        $latestEpoch = Get-LatestCompletedEpoch
        if ($latestEpoch -ge $targetEpoch) {
            Write-WatchdogLog "Target epoch $targetEpoch reached. Watchdog exiting."
            break
        }

        $running = @(Get-TrainingProcesses)
        $recentActivity = Has-RecentTrainingLogActivity
        if ($running.Count -eq 0) {
            Start-TrainingProcess
            Start-Sleep -Seconds 10
        } else {
            $pids = ($running | ForEach-Object { $_.ProcessId }) -join ","
            if ([string]::IsNullOrWhiteSpace($pids)) {
                $pids = "unknown"
            }
            Write-WatchdogLog "Training active. PID=$pids recent_log_activity=$recentActivity latest_epoch=$latestEpoch"
            Start-Sleep -Seconds 30
        }
    }
} catch {
    Write-WatchdogLog "Fatal error: $($_.Exception.Message)"
    throw
}

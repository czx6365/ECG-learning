$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [Console]::OutputEncoding
$env:PYTHONUTF8 = "1"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$outputDir = Join-Path $repoRoot "checkpoints\pretrain\ECG_DSA_vqhbr45_all_raw_v1"
$logDir = Join-Path $repoRoot "log\pretrain\ECG_DSA_vqhbr45_all_raw_v1"
$watchdogLog = Join-Path $logDir "watchdog.log"
$stdoutLog = Join-Path $logDir "train_stdout.log"
$stderrLog = Join-Path $logDir "train_stderr.log"
$restartStatePath = Join-Path $logDir "watchdog_restart_count.txt"
$targetEpoch = 50

New-Item -ItemType Directory -Force -Path $logDir | Out-Null
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
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
            $_.CommandLine -like "*run_heartlang_pretraining.py*" -and
            $_.CommandLine -like "*ECG_DSA_vqhbr45_all_raw_v1*"
        }
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
    return 4
}

function Start-TrainingProcess() {
    $restartCount = Get-RestartCount
    $batchSize = Get-BatchSizeForRestart $restartCount
    $resumePath = Join-Path $outputDir "checkpoint.pth"

    $args = @(
        "-u",
        "heartlang/cli/run_heartlang_pretraining.py",
        "--output_dir", "checkpoints/pretrain/ECG_DSA_vqhbr45_all_raw_v1",
        "--log_dir", "log/pretrain/ECG_DSA_vqhbr45_all_raw_v1",
        "--model", "HeartLang",
        "--tokenizer_model", "vqhbr",
        "--tokenizer_weight", "checkpoints/vqhbr/ECG_DSA_all_raw_v1/checkpoint-45.pth",
        "--pretrain_data_dirs", "datasets/ecg_datasets/ECG_DSA/pretrain_qrs/all_raw_v1",
        "--batch_size", "$batchSize",
        "--num_workers", "0",
        "--lr", "1.5e-5",
        "--min_lr", "1e-6",
        "--warmup_epochs", "5",
        "--clip_grad", "3.0",
        "--layer_scale_init_value", "0.1",
        "--opt", "adamw",
        "--opt_betas", "0.9", "0.98",
        "--opt_eps", "1e-8",
        "--weight_decay", "0.05",
        "--epochs", "$targetEpoch",
        "--save_ckpt_freq", "5",
        "--codebook_size", "8192",
        "--codebook_dim", "128",
        "--gradient_accumulation_steps", "1",
        "--world_size", "1",
        "--device", "cuda"
    )

    if (Test-Path $resumePath) {
        $args += @("--resume", "checkpoints/pretrain/ECG_DSA_vqhbr45_all_raw_v1/checkpoint.pth", "--no_auto_resume")
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
            $running = @(Get-TrainingProcesses)
            foreach ($proc in $running) {
                try {
                    Stop-Process -Id $proc.ProcessId -Force -ErrorAction Stop
                    Write-WatchdogLog "Stopped training PID=$($proc.ProcessId) after reaching checkpoint-$targetEpoch."
                } catch {
                    Write-WatchdogLog "Failed to stop PID=$($proc.ProcessId): $($_.Exception.Message)"
                }
            }
            Write-WatchdogLog "Target epoch $targetEpoch reached. Watchdog exiting."
            break
        }

        $running = @(Get-TrainingProcesses)
        if ($running.Count -eq 0) {
            Start-TrainingProcess
            Start-Sleep -Seconds 10
        } else {
            $pids = ($running | ForEach-Object { $_.ProcessId }) -join ","
            Write-WatchdogLog "Training active. PID=$pids latest_epoch=$latestEpoch"
            Start-Sleep -Seconds 30
        }
    }
} catch {
    Write-WatchdogLog "Fatal error: $($_.Exception.Message)"
    throw
}

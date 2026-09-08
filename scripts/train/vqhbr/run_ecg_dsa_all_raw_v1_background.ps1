$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [Console]::OutputEncoding
$env:PYTHONUTF8 = "1"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$logDir = Join-Path $repoRoot "log\vqhbr\ECG_DSA_all_raw_v1"
$launcherLog = Join-Path $logDir "task_launcher.log"
$stdoutLog = Join-Path $logDir "scheduled_train_stdout.log"
$stderrLog = Join-Path $logDir "scheduled_train_stderr.log"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null
Set-Location $repoRoot

$startTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content -Path $launcherLog -Encoding UTF8 -Value "=== [$startTime] Scheduled task launcher start ==="

$pythonArgs = @(
    "-u",
    "heartlang/cli/run_vqhbr_training.py",
    "--output_dir", "checkpoints/vqhbr/ECG_DSA_all_raw_v1",
    "--log_dir", "log/vqhbr/ECG_DSA_all_raw_v1",
    "--model", "vqhbr",
    "--pretrain_data_dirs", "datasets/ecg_datasets/ECG_DSA/pretrain_qrs/all_raw_v1",
    "--codebook_n_emd", "8192",
    "--codebook_emd_dim", "128",
    "--quantize_kmeans_init",
    "--batch_size", "8",
    "--num_workers", "0",
    "--opt", "adamw",
    "--opt_betas", "0.9", "0.99",
    "--weight_decay", "1e-4",
    "--warmup_epochs", "3",
    "--epochs", "10",
    "--save_ckpt_freq", "1",
    "--val_freq", "5",
    "--world_size", "1",
    "--lr", "2e-4",
    "--min_lr", "1e-6",
    "--device", "cuda"
)

$process = Start-Process `
    -FilePath "python" `
    -ArgumentList $pythonArgs `
    -WorkingDirectory $repoRoot `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -PassThru

$endTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content -Path $launcherLog -Encoding UTF8 -Value "=== [$endTime] Detached python PID: $($process.Id) ==="

exit 0

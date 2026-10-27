# Run a command at BelowNormal priority with capped threads (do not disturb other GPU training).
# Usage: .\scripts\run_lowprio.ps1 -- uv run python scripts\something.py

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Cmd
)

if (-not $Cmd -or $Cmd.Count -eq 0) {
    Write-Host "Usage: .\scripts\run_lowprio.ps1 -- <command> [args...]"
    exit 1
}

# Skip leading "--" if present
if ($Cmd[0] -eq "--") {
    $Cmd = $Cmd[1..($Cmd.Count - 1)]
}

$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
$env:OPENBLAS_NUM_THREADS = "4"
$env:NUMEXPR_NUM_THREADS = "4"
$env:ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS = "4"

Write-Host "Low-priority run (threads=4): $($Cmd -join ' ')"
$p = Start-Process -FilePath $Cmd[0] -ArgumentList ($Cmd[1..($Cmd.Count - 1)]) -NoNewWindow -PassThru -Wait
try {
    $p.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal
} catch {
    # Priority may not be changeable after exit; set via Start if needed
}
exit $p.ExitCode

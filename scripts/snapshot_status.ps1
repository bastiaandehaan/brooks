# scripts/snapshot_status.ps1
# Usage:
#   .\scripts\snapshot_status.ps1
# Optional:
#   .\scripts\snapshot_status.ps1 -Days 10

param(
    [int]$Days = 10
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "============================================================="
Write-Host " SNAPSHOT STATUS (brooks repo) " -ForegroundColor Cyan
Write-Host "============================================================="
Write-Host ""

# 1) Git status
Write-Host "== GIT STATUS ==" -ForegroundColor Yellow
git status
Write-Host ""

# 2) Last commits
Write-Host "== GIT LOG (last 10) ==" -ForegroundColor Yellow
git log -10 --oneline
Write-Host ""

# 3) Pytest
Write-Host "== PYTEST ==" -ForegroundColor Yellow
pytest -q
Write-Host ""

# 4) Backtest quick run
Write-Host "== BACKTEST RUNNER (days=$Days) ==" -ForegroundColor Yellow
python -m backtest.runner --days $Days
Write-Host ""

# 5) Trades/day validation script (if exists)
$scriptPath = ".\scripts\check_trades_per_day.py"
if (Test-Path $scriptPath) {
    Write-Host "== CHECK TRADES PER DAY ==" -ForegroundColor Yellow
    python $scriptPath
    Write-Host ""
} else {
    Write-Host "== CHECK TRADES PER DAY ==" -ForegroundColor Yellow
    Write-Host "Skipped: scripts/check_trades_per_day.py not found"
    Write-Host ""
}

Write-Host "============================================================="
Write-Host " DONE " -ForegroundColor Green
Write-Host "============================================================="
Write-Host ""

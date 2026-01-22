# scripts/dev_check.ps1
# Development workflow script for Brooks Trading Framework

$ErrorActionPreference = "Stop"

Write-Host "🔍 Running Brooks Development Checks..." -ForegroundColor Cyan
Write-Host ""

# 1. Ruff Lint Check
Write-Host "📋 Step 1: Ruff Lint" -ForegroundColor Yellow
ruff check .
if ($LASTEXITCODE -ne 0)
{
    Write-Host "❌ Ruff found issues. Run 'ruff check . --fix' to auto-fix." -ForegroundColor Red
    exit 1
}
Write-Host "✅ Ruff check passed" -ForegroundColor Green
Write-Host ""

# 2. Black Format Check
Write-Host "📋 Step 2: Black Format" -ForegroundColor Yellow
black --check .
if ($LASTEXITCODE -ne 0)
{
    Write-Host "❌ Black formatting issues. Run 'black .' to format." -ForegroundColor Red
    exit 1
}
Write-Host "✅ Black format passed" -ForegroundColor Green
Write-Host ""

# 3. Pytest
Write-Host "📋 Step 3: Running Tests" -ForegroundColor Yellow
pytest -v
if ($LASTEXITCODE -ne 0)
{
    Write-Host "❌ Tests failed" -ForegroundColor Red
    exit 1
}
Write-Host "✅ All tests passed" -ForegroundColor Green
Write-Host ""

Write-Host "🎉 All checks passed! Production ready." -ForegroundColor Green
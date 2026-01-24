param(
    [string]$RepoRoot = (Get-Location).Path,
    [string]$OutDir = "evidence",
    [string]$OutName = ("audit_bundle_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".txt"),
    [switch]$IncludeMainAndToml = $true,
    [switch]$IncludeLatestLog = $true
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Ensure-Dir([string]$p)
{
    New-Item -ItemType Directory -Path $p -Force | Out-Null
}

function Sha256([string]$Path)
{
    (Get-FileHash -Algorithm SHA256 -Path $Path).Hash.ToLower()
}

function Write-SectionHeader([string]$Path, [string]$Label)
{
    "================================================================================" | Out-File $Path -Append -Encoding UTF8
    "FILE: $Label" | Out-File $Path -Append -Encoding UTF8
    "================================================================================" | Out-File $Path -Append -Encoding UTF8
}

Ensure-Dir $OutDir
$outPath = Join-Path $RepoRoot $OutDir
$outFile = Join-Path $outPath $OutName

# --- Baseline Git info ---
$commit = ""
$lastCommit = ""
try
{
    $commit = (git rev-parse HEAD)
}
catch
{
}
try
{
    $lastCommit = (git log -1 --oneline)
}
catch
{
}

# --- File list (must-have) ---
$files = @(
    "scripts\live_auto_trader.py",

    "execution\emergency_stop.py",
    "execution\trade_gate.py",
    "execution\ftmo_state.py",
    "execution\ftmo_guardian.py",
    "execution\guardrails.py",
    "execution\selection.py",
    "execution\risk_manager.py",
    "execution\trade_executor.py",
    "execution\order_manager.py",

    "strategies\config.py",
    "strategies\context.py",
    "strategies\h2l2.py",
    "strategies\regime.py",

    "utils\mt5_client.py",
    "utils\mt5_data.py",
    "utils\symbol_spec.py",

    "config\strategies\us500_sniper.yaml",

    "backtest\runner.py",
    "backtest\visualiser.py"
)

# Optional repo glue
if ($IncludeMainAndToml)
{
    $files += @("main.py", "pyproject.toml", "poetry.lock", "requirements.txt")
}

# Optional latest log
$latestLogPath = $null
if ($IncludeLatestLog)
{
    $logsDir = Join-Path $RepoRoot "logs"
    if (Test-Path $logsDir)
    {
        $latest = Get-ChildItem $logsDir -Filter "auto_trader_*.log" -ErrorAction SilentlyContinue |
                Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($latest)
        {
            $latestLogPath = $latest.FullName
        }
    }
}

# --- Write header ---
"=== AUDIT BUNDLE TXT ===" | Out-File $outFile -Encoding UTF8
"Generated: $( Get-Date -Format s )" | Out-File $outFile -Append -Encoding UTF8
"RepoRoot:   $RepoRoot" | Out-File $outFile -Append -Encoding UTF8
"Commit:     $commit" | Out-File $outFile -Append -Encoding UTF8
"LastCommit: $lastCommit" | Out-File $outFile -Append -Encoding UTF8
"" | Out-File $outFile -Append -Encoding UTF8

# --- Append each file with separators ---
$manifest = New-Object System.Collections.Generic.List[object]
$missing = New-Object System.Collections.Generic.List[string]

foreach ($rel in $files)
{
    $full = Join-Path $RepoRoot $rel
    Write-SectionHeader $outFile $rel
    if (Test-Path $full)
    {
        Get-Content $full | Out-File $outFile -Append -Encoding UTF8
        $fi = Get-Item $full
        $manifest.Add([pscustomobject]@{
            relative_path = $rel
            bytes = $fi.Length
            sha256 = Sha256 $full
            last_write_utc = $fi.LastWriteTimeUtc.ToString("o")
        }) | Out-Null
    }
    else
    {
        "(MISSING)" | Out-File $outFile -Append -Encoding UTF8
        $missing.Add($rel) | Out-Null
    }
    "" | Out-File $outFile -Append -Encoding UTF8
}

# Latest log appended (if found)
if ($latestLogPath)
{
    $relLabel = "logs\" + (Split-Path $latestLogPath -Leaf)
    Write-SectionHeader $outFile $relLabel
    Get-Content $latestLogPath | Out-File $outFile -Append -Encoding UTF8
    $fi = Get-Item $latestLogPath
    $manifest.Add([pscustomobject]@{
        relative_path = $relLabel
        bytes = $fi.Length
        sha256 = Sha256 $latestLogPath
        last_write_utc = $fi.LastWriteTimeUtc.ToString("o")
    }) | Out-Null
    "" | Out-File $outFile -Append -Encoding UTF8
}

# Missing summary
Write-SectionHeader $outFile "MISSING_FILES_SUMMARY"
if ($missing.Count -gt 0)
{
    $missing | Out-File $outFile -Append -Encoding UTF8
}
else
{
    "None" | Out-File $outFile -Append -Encoding UTF8
}
"" | Out-File $outFile -Append -Encoding UTF8

# Manifest at end
Write-SectionHeader $outFile "MANIFEST_SHA256"
$manifest | ConvertTo-Csv -NoTypeInformation | Out-File $outFile -Append -Encoding UTF8

Write-Host ""
Write-Host "DONE"
Write-Host "TXT: $outFile"
Write-Host ""

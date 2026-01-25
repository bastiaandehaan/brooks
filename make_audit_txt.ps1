# tools/make_audit_txt.ps1
# Script: make_audit_txt.ps1
# Module: tools
# Location: tools\make_audit_txt.ps1
# Purpose: Bundle key audit files into a single TXT (code + config + git evidence + requirements + manifest).

[CmdletBinding()]
param(
    [string]$RepoRoot = "",
    [string]$OutDir = "evidence",
    [string]$OutName = ("audit_bundle_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".txt"),
    [string[]]$ExtraFiles = @(),
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

function Get-RepoRootOrCwd()
{
    if ($RepoRoot -and (Test-Path $RepoRoot))
    {
        return (Resolve-Path $RepoRoot).Path
    }
    try
    {
        $top = (git rev-parse --show-toplevel 2> $null)
        if ($top)
        {
            return (Resolve-Path $top).Path
        }
    }
    catch
    {
    }
    return (Get-Location).Path
}

function Get-PythonExe()
{
    # Prefer active venv
    if ($env:VIRTUAL_ENV)
    {
        $venvPy = Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"
        if (Test-Path $venvPy)
        {
            return $venvPy
        }
    }

    # Common local venv folders
    foreach ($cand in @(".venv\Scripts\python.exe", "venv\Scripts\python.exe"))
    {
        $p = Join-Path $global:RepoRootResolved $cand
        if (Test-Path $p)
        {
            return $p
        }
    }

    # Fallbacks
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd)
    {
        return $cmd.Source
    }

    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py)
    {
        return "py"
    }

    return $null
}

function Write-Section([System.IO.StreamWriter]$w, [string]$title)
{
    $w.WriteLine("")
    $w.WriteLine(("=" * 80))
    $w.WriteLine("SECTION: $title")
    $w.WriteLine(("=" * 80))
}

function Write-FileHeader([System.IO.StreamWriter]$w, [string]$rel)
{
    $w.WriteLine("")
    $w.WriteLine(("=" * 80))
    $w.WriteLine("SECTION: $rel")
    $w.WriteLine(("=" * 80))
}

function RelFromFull([string]$full)
{
    $root = $global:RepoRootResolved.TrimEnd('\', '/')
    $f = (Resolve-Path $full).Path
    if ( $f.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase))
    {
        $rel = $f.Substring($root.Length).TrimStart('\', '/')
        return $rel.Replace('/', '\')
    }
    return $full
}

function Add-IfExists([System.Collections.Generic.List[string]]$list, [string]$rel)
{
    if (-not $rel)
    {
        return
    }
    $rel2 = $rel.Replace('/', '\')
    $full = Join-Path $global:RepoRootResolved $rel2
    if (Test-Path $full -PathType Leaf)
    {
        $list.Add($rel2) | Out-Null
    }
}

function Add-Glob([System.Collections.Generic.List[string]]$list, [string]$relDir, [string]$filter, [switch]$Recurse)
{
    $dir = Join-Path $global:RepoRootResolved ($relDir.Replace('/', '\'))
    if (-not (Test-Path $dir -PathType Container))
    {
        return
    }

    $items = Get-ChildItem -Path $dir -File -Filter $filter -Recurse:$Recurse
    foreach ($it in $items)
    {
        $list.Add((RelFromFull $it.FullName)) | Out-Null
    }
}

function Add-PathFlexible([System.Collections.Generic.List[string]]$list, [string]$relPath)
{
    $rel2 = $relPath.Replace('/', '\')
    $full = Join-Path $global:RepoRootResolved $rel2
    if (-not (Test-Path $full))
    {
        return
    }

    $item = Get-Item $full -Force
    if ($item.PSIsContainer)
    {
        # Include relevant source/config artifacts from that directory
        Add-Glob $list $rel2 "*.py" -Recurse
        Add-Glob $list $rel2 "*.yaml" -Recurse
        Add-Glob $list $rel2 "*.yml" -Recurse
        Add-Glob $list $rel2 "*.json" -Recurse
        Add-Glob $list $rel2 "*.txt" -Recurse
    }
    else
    {
        $list.Add($rel2) | Out-Null
    }
}

$global:RepoRootResolved = Get-RepoRootOrCwd

# Output path
Ensure-Dir (Join-Path $global:RepoRootResolved $OutDir)
$outFile = Join-Path (Join-Path $global:RepoRootResolved $OutDir) $OutName

# StreamWriter (UTF8 no BOM)
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$w = New-Object System.IO.StreamWriter($outFile, $false, $utf8NoBom)

# Git info
$commit = ""
$lastCommit = ""
try
{
    $commit = (git rev-parse HEAD 2> $null)
}
catch
{
}
try
{
    $lastCommit = (git log -1 --oneline 2> $null)
}
catch
{
}

# Requirements capture (always create requirements.txt even on failure)
$reqPath = Join-Path $global:RepoRootResolved "requirements.txt"
$pyExe = Get-PythonExe
try
{
    if ($pyExe -eq "py")
    {
        & py -m pip freeze | Out-File -FilePath $reqPath -Encoding utf8
    }
    elseif ($pyExe)
    {
        & $pyExe -m pip freeze | Out-File -FilePath $reqPath -Encoding utf8
    }
    else
    {
        throw "No python executable found (activate your venv or ensure python is on PATH)."
    }
}
catch
{
    @(
        "ERROR: requirements capture failed",
        "Reason: $( $_.Exception.Message )",
        "Hint: activate your venv (e.g. .\.venvs\brooks\Scripts\Activate.ps1) and re-run."
    ) | Out-File -FilePath $reqPath -Encoding utf8
}

# Optional runtime requirements (best-effort)
$reqRuntimePath = Join-Path $global:RepoRootResolved "requirements_runtime.txt"
try
{
    $exclude = @(
        "^pytest", "^pytest-cov", "^coverage",
        "^pandas-stubs", "^types-", "^iniconfig", "^pluggy", "^pygments"
    )
    $lines = Get-Content $reqPath -ErrorAction SilentlyContinue
    $filtered = $lines | Where-Object {
        $line = $_
        -not ($exclude | Where-Object { $line -match $_ })
    }
    $filtered | Out-File -FilePath $reqRuntimePath -Encoding utf8
}
catch
{
    # If it fails, do not block bundling
}

# Build file selection
$list = New-Object System.Collections.Generic.List[string]

# Core must-have files (existing selection)
Add-IfExists $list "scripts\live_auto_trader.py"

Add-IfExists $list "execution\emergency_stop.py"
Add-IfExists $list "execution\trade_gate.py"
Add-IfExists $list "execution\ftmo_state.py"
Add-IfExists $list "execution\ftmo_guardian.py"
Add-IfExists $list "execution\guardrails.py"
Add-IfExists $list "execution\selection.py"
Add-IfExists $list "execution\risk_manager.py"
Add-IfExists $list "execution\trade_executor.py"
Add-IfExists $list "execution\order_manager.py"

Add-IfExists $list "strategies\config.py"
Add-IfExists $list "strategies\context.py"
Add-IfExists $list "strategies\h2l2.py"
Add-IfExists $list "strategies\regime.py"

Add-IfExists $list "utils\mt5_client.py"
Add-IfExists $list "utils\mt5_data.py"
Add-IfExists $list "utils\symbol_spec.py"

# Strategy configs: include ALL YAMLs so new ones are never missed
Add-Glob $list "config\strategies" "*.yaml" -Recurse:$false
Add-Glob $list "config\strategies" "*.yml" -Recurse:$false

# Backtest core + optimizer + analytics/risk reports
Add-IfExists $list "backtest\runner.py"
Add-IfExists $list "backtest\visualiser.py"
Add-IfExists $list "backtest\optimizer.py"
Add-IfExists $list "backtest\ftmo_risk_report.py"
Add-IfExists $list "backtest\ftmo_report_generator.py"

# Analytics layer (auto-include everything under backtest\analytics)
Add-Glob $list "backtest\analytics" "*.py" -Recurse
Add-Glob $list "backtest\analytics" "*.yaml" -Recurse
Add-Glob $list "backtest\analytics" "*.yml" -Recurse

# If you have a risk_optimizer path (file or folder), include it automatically
Add-PathFlexible $list "risk_optimizer"

# Include audit helpers if present
Add-IfExists $list "knobs_and_mismatches.json"
Add-IfExists $list "tools\knobs_and_mismatches.json"

# Entry points (include main.py if it exists; otherwise include legacy marker if present)
Add-IfExists $list "main.py"
if (-not (Test-Path (Join-Path $global:RepoRootResolved "main.py") -PathType Leaf))
{
    Add-IfExists $list "main.py.LEGACY_DO_NOT_USE"
}
Add-IfExists $list "pyproject.toml"

# Always include requirements artifacts
Add-IfExists $list "requirements.txt"
Add-IfExists $list "requirements_runtime.txt"

# Latest log (best-effort)
if ($IncludeLatestLog)
{
    $logDir = Join-Path $global:RepoRootResolved "logs"
    if (Test-Path $logDir -PathType Container)
    {
        $latest = Get-ChildItem $logDir -File -ErrorAction SilentlyContinue |
                Sort-Object LastWriteTimeUtc -Descending |
                Select-Object -First 1
        if ($latest)
        {
            $list.Add((RelFromFull $latest.FullName)) | Out-Null
        }
    }
}

# Extra files from CLI
foreach ($ef in $ExtraFiles)
{
    if (-not $ef)
    {
        continue
    }
    $rel = $ef.Replace('/', '\')
    $full = Join-Path $global:RepoRootResolved $rel
    if (Test-Path $full -PathType Leaf)
    {
        $list.Add($rel) | Out-Null
    }
}

# De-dup + sort
$files = $list.ToArray() |
        Where-Object { $_ -and $_.Trim() -ne "" } |
        ForEach-Object { $_.Replace('/', '\') } |
        Sort-Object -Unique

# Header
$w.WriteLine("=== AUDIT BUNDLE ===")
$w.WriteLine("Generated: " + (Get-Date).ToString("s"))
$w.WriteLine("RepoRoot:  " + $global:RepoRootResolved)
$w.WriteLine("Commit:    " + $commit)
$w.WriteLine("LastCommit:" + $lastCommit)

# Git status/diff evidence
Write-Section $w "GIT_STATUS"
try
{
    (git status --porcelain 2> $null) | ForEach-Object { $w.WriteLine($_) }
}
catch
{
    $w.WriteLine("(git status failed)")
}

Write-Section $w "GIT_DIFF_STAT"
try
{
    (git diff --stat 2> $null) | ForEach-Object { $w.WriteLine($_) }
}
catch
{
    $w.WriteLine("(git diff --stat failed)")
}

# File contents + missing tracking
$missing = New-Object System.Collections.Generic.List[string]
foreach ($rel in $files)
{
    $full = Join-Path $global:RepoRootResolved $rel
    Write-FileHeader $w $rel

    if (-not (Test-Path $full -PathType Leaf))
    {
        $w.WriteLine("(MISSING)")
        $missing.Add($rel) | Out-Null
        continue
    }

    try
    {
        Get-Content -Path $full -ErrorAction Stop | ForEach-Object { $w.WriteLine($_) }
    }
    catch
    {
        $w.WriteLine("(FAILED TO READ: $( $_.Exception.Message ))")
    }
}

# Missing list summary
Write-Section $w "MISSING_FILES"
foreach ($m in $missing)
{
    $w.WriteLine($m)
}

# Manifest
Write-Section $w "MANIFEST_SHA256"
$w.WriteLine('"relative_path","bytes","sha256","last_write_utc"')
foreach ($rel in $files)
{
    $full = Join-Path $global:RepoRootResolved $rel
    if (-not (Test-Path $full -PathType Leaf))
    {
        continue
    }

    $fi = Get-Item $full
    $bytes = $fi.Length
    $hash = Sha256 $full
    $ts = $fi.LastWriteTimeUtc.ToString("o")
    $w.WriteLine(('"{0}","{1}","{2}","{3}"' -f $rel, $bytes, $hash, $ts))
}

$w.Flush()
$w.Close()

Write-Host "DONE / TXT: $outFile"

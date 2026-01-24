# make_audit_txt.ps1
# Create a single audit TXT bundle with key source files, git evidence and SHA256 manifest.

[CmdletBinding()]
param(
    [string]$RepoRoot = "",
    [string]$OutDir = "evidence",
    [string]$OutName = ("audit_bundle_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".txt")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---------- helpers ----------

function Ensure-Dir([string]$Path)
{
    if (-not (Test-Path $Path))
    {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
}

function Sha256([string]$Path)
{
    (Get-FileHash -Algorithm SHA256 -Path $Path).Hash.ToLower()
}

function Write-Header([string]$File, [string]$Title)
{
    "================================================================================" | Out-File $File -Append -Encoding UTF8
    "SECTION: $Title" | Out-File $File -Append -Encoding UTF8
    "================================================================================" | Out-File $File -Append -Encoding UTF8
}

function Try-Run([scriptblock]$Cmd)
{
    try
    {
        & $Cmd
    }
    catch
    {
        $null
    }
}

function Safe-Trim([object]$v)
{
    if ($null -eq $v)
    {
        return ""
    }
    return (($v | Out-String).Trim())
}

function Resolve-RepoRoot([string]$Given)
{
    if ($Given -and (Test-Path $Given))
    {
        return (Resolve-Path $Given).Path
    }
    $gitRoot = Safe-Trim (Try-Run { git rev-parse --show-toplevel })
    if (-not [string]::IsNullOrWhiteSpace($gitRoot))
    {
        return $gitRoot
    }
    return (Get-Location).Path
}

# ---------- resolve paths ----------

$RepoRoot = Resolve-RepoRoot $RepoRoot
Ensure-Dir (Join-Path $RepoRoot $OutDir)
$outFile = Join-Path (Join-Path $RepoRoot $OutDir) $OutName

# ---------- git evidence ----------

$gitCommit = Safe-Trim (Try-Run { git -C $RepoRoot rev-parse HEAD })
$gitLog = Safe-Trim (Try-Run { git -C $RepoRoot log -1 --oneline })
$gitStatus = Safe-Trim (Try-Run { git -C $RepoRoot status --porcelain })
$gitDiff = Safe-Trim (Try-Run { git -C $RepoRoot diff --stat })

if ( [string]::IsNullOrWhiteSpace($gitCommit))
{
    $gitCommit = "(unknown)"
}
if ( [string]::IsNullOrWhiteSpace($gitLog))
{
    $gitLog = "(unknown)"
}
if ( [string]::IsNullOrWhiteSpace($gitStatus))
{
    $gitStatus = "clean"
}
if ( [string]::IsNullOrWhiteSpace($gitDiff))
{
    $gitDiff = "none"
}

# ---------- file list (LIVE CORE) ----------

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
    "main.py",
    "requirements.txt"
)

# ---------- write bundle ----------

"=== AUDIT BUNDLE ==="                 | Out-File $outFile -Encoding UTF8
"Generated: $( Get-Date -Format s )"     | Out-File $outFile -Append -Encoding UTF8
"RepoRoot:  $RepoRoot"                 | Out-File $outFile -Append -Encoding UTF8
"Commit:    $gitCommit"                | Out-File $outFile -Append -Encoding UTF8
"LastCommit:$gitLog"                   | Out-File $outFile -Append -Encoding UTF8
""                                     | Out-File $outFile -Append -Encoding UTF8

Write-Header $outFile "GIT_STATUS"
$gitStatus | Out-File $outFile -Append -Encoding UTF8
"" | Out-File $outFile -Append -Encoding UTF8

Write-Header $outFile "GIT_DIFF_STAT"
$gitDiff | Out-File $outFile -Append -Encoding UTF8
"" | Out-File $outFile -Append -Encoding UTF8

$manifest = @()
$missing = @()

foreach ($rel in $files)
{
    $full = Join-Path $RepoRoot $rel
    Write-Header $outFile $rel

    if (Test-Path $full)
    {
        Get-Content $full | Out-File $outFile -Append -Encoding UTF8
        $fi = Get-Item $full
        $manifest += [pscustomobject]@{
            relative_path = $rel
            bytes = $fi.Length
            sha256 = Sha256 $full
            last_write_utc = $fi.LastWriteTimeUtc.ToString("o")
        }
    }
    else
    {
        "(MISSING)" | Out-File $outFile -Append -Encoding UTF8
        $missing += $rel
    }

    "" | Out-File $outFile -Append -Encoding UTF8
}

Write-Header $outFile "MISSING_FILES"
if ($missing.Count -gt 0)
{
    $missing
}
else
{
    "None"
}
Out-File $outFile -Append -Encoding UTF8
"" | Out-File $outFile -Append -Encoding UTF8

Write-Header $outFile "MANIFEST_SHA256"
$manifest | ConvertTo-Csv -NoTypeInformation |
        Out-File $outFile -Append -Encoding UTF8

Write-Host ""
Write-Host "DONE"
Write-Host "RepoRoot: $RepoRoot"
Write-Host "TXT:      $outFile"
Write-Host ""

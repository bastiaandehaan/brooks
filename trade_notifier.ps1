# Save as: trade_notifier.ps1
$LogPath = (Get-ChildItem logs\auto_trader_*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
$LastLine = 0

Write-Host "Trade Notifier gestart - watching $LogPath" -ForegroundColor Cyan

while ($true)
{
    $Lines = Get-Content $LogPath
    $CurrentCount = $Lines.Count

    if ($CurrentCount -gt $LastLine)
    {
        # Check nieuwe regels
        $NewLines = $Lines[$LastLine..($CurrentCount - 1)]

        # Zoek naar trade execution
        $NewLines | ForEach-Object {
            if ($_ -match "TRADE EXECUTED SUCCESSFULLY")
            {
                Write-Host "`n========================================" -ForegroundColor Green
                Write-Host "TRADE EXECUTED!" -ForegroundColor Green
                Write-Host "========================================`n" -ForegroundColor Green

                # Geluid afspelen (Windows beep)
                [console]::beep(800, 300)
                [console]::beep(1000, 300)
            }

            if ($_ -match "SETUP DETECTED")
            {
                Write-Host "`nSETUP FOUND - Trade wordt nu geevalueerd..." -ForegroundColor Yellow
            }

            if ($_ -match "Side: (LONG|SHORT)")
            {
                Write-Host $_ -ForegroundColor Cyan
            }

            if ($_ -match "Entry:|Stop:|Target:")
            {
                Write-Host $_ -ForegroundColor White
            }

            if ($_ -match "Ticket:")
            {
                Write-Host $_ -ForegroundColor Green
            }

            if ($_ -match "FTMO GATE BLOCKED")
            {
                Write-Host "`nTRADE BLOCKED door FTMO limiet" -ForegroundColor Red
            }

            if ($_ -match "Guardrails rejected")
            {
                Write-Host "TRADE BLOCKED door Guardrails (session/limits)" -ForegroundColor Red
            }
        }

        $LastLine = $CurrentCount
    }

    Start-Sleep -Seconds 2
}
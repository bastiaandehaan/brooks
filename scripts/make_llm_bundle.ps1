cd C:\Users\basti\PycharmProjects\brooks

$exclude = "\\.venv\\|\\__pycache__\\|\\.pytest_cache\\|\\.mypy_cache\\|\\.git\\|\\_export\\|\\_llm_export\\"
$out = "llm_dump.txt"

Remove-Item $out -Force -ErrorAction SilentlyContinue

Get-ChildItem -Recurse -File -Filter *.py |
  Where-Object { $_.FullName -notmatch $exclude } |
  Sort-Object FullName |
  ForEach-Object {
    Add-Content -Encoding UTF8 -Path $out -Value ""
    Add-Content -Encoding UTF8 -Path $out -Value ("### FILE: " + $_.FullName)
    Add-Content -Encoding UTF8 -Path $out -Value (Get-Content $_.FullName -Raw -Encoding UTF8)
  }

Write-Host "Wrote $out"

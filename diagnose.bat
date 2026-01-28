@echo off
REM ============================================================================
REM DIAGNOSE SCRIPT - Check welke versie je hebt
REM ============================================================================

echo.
echo ============================================================================
echo   BROOKS GRID SEARCH - DIAGNOSE
echo ============================================================================
echo.

cd /d C:\Users\basti\PycharmProjects\brooks

echo Checking scripts folder...
echo.

if exist "scripts\fast_grid_search.py" (
    echo [OK] fast_grid_search.py GEVONDEN!
    echo      Locatie: scripts\fast_grid_search.py

    REM Check file size (nieuwe versie is groter)
    for %%A in (scripts\fast_grid_search.py) do (
        echo      Grootte: %%~zA bytes

        if %%~zA GTR 15000 (
            echo      Status: NIEUWE VERSIE (geoptimaliseerd)
        ) else (
            echo      Status: ONBEKEND - mogelijk oude versie
        )
    )
) else (
    echo [FOUT] fast_grid_search.py NIET GEVONDEN!
    echo        Download deze van Claude!
)

echo.

if exist "scripts\full_grid_search.py" (
    echo [OK] full_grid_search.py gevonden (oude versie)
    for %%A in (scripts\full_grid_search.py) do (
        echo      Grootte: %%~zA bytes
    )
) else (
    echo [INFO] full_grid_search.py niet gevonden (normaal)
)

echo.
echo ============================================================================
echo   CONTROLE BATCH FILES
echo ============================================================================
echo.

if exist "run_grid_search_v2_fixed.bat" (
    echo [OK] run_grid_search_v2_fixed.bat GEVONDEN! (GEBRUIK DEZE!)
) else (
    echo [WAARSCHUWING] run_grid_search_v2_fixed.bat NIET GEVONDEN
    echo                Download deze van Claude!
)

echo.

if exist "run_grid_search_v2.bat" (
    echo [INFO] run_grid_search_v2.bat gevonden (heeft bug, gebruik fixed versie)
)

if exist "run_grid_search.bat" (
    echo [INFO] run_grid_search.bat gevonden (oude versie)
)

echo.
echo ============================================================================
echo   AANBEVELING
echo ============================================================================
echo.
echo 1. Zorg dat fast_grid_search.py in scripts\ staat (GROTER dan 15KB)
echo 2. Gebruik run_grid_search_v2_fixed.bat om te starten
echo 3. Kies optie 1 (Fast mode geoptimaliseerd)
echo.
echo Druk op een toets om af te sluiten...
pause >nul
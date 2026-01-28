@echo off
REM ============================================================================
REM BROOKS GRID SEARCH V2 - GEOPTIMALISEERD (10x SNELLER!)
REM ============================================================================
REM
REM NIEUWE VERSIE: Data wordt 1x opgehaald, niet 384x!
REM - Fast mode: ~5-10 minuten (was 40 min)
REM - Full mode: ~2-3 uur (was 68 uur!)
REM
REM ============================================================================

echo.
echo ============================================================================
echo   BROOKS GRID SEARCH V2 - GEOPTIMALISEERDE VERSIE
echo ============================================================================
echo.
echo Stap 1: Controleren of MT5 draait...
echo.

REM Check of MT5 terminal draait
tasklist /FI "IMAGENAME eq terminal64.exe" 2>NUL | find /I /N "terminal64.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo [OK] MT5 terminal draait!
) else (
    echo [WAARSCHUWING] MT5 terminal niet gevonden
    echo.
    echo Start alstublieft MetaTrader 5 EERST voordat je verder gaat!
    echo.
    pause
    exit /b 1
)

echo.
echo Stap 2: Navigeren naar project directory...
cd /d C:\Users\basti\PycharmProjects\brooks

echo.
echo Stap 3: UTF-8 encoding instellen...
chcp 65001 >nul

echo.
echo Stap 4: Python environment activeren...
set PYTHONUTF8=1

echo.
echo ============================================================================
echo   KIES JE VERSIE:
echo ============================================================================
echo.
echo   1. FAST MODE (GEOPTIMALISEERD)  - 384 tests   (~5-10 min)   [AANGERADEN!]
echo      Data wordt 1x opgehaald, 10x sneller!
echo.
echo   2. FULL MODE (GEOPTIMALISEERD)  - 40,960 tests (~2-3 uur)
echo      Voor absolute zekerheid, 's nachts runnen
echo.
echo   3. OLD MODE (LANGZAAM)          - 384 tests   (~40 min)
echo      Oude methode, elke test haalt data opnieuw op
echo.
set /p MODE_CHOICE="Typ 1, 2 of 3 en druk op ENTER: "

REM ============================================================================
REM FIXED: Proper IF/ELSE logic with GOTO
REM ============================================================================

if "%MODE_CHOICE%"=="1" goto MODE_FAST_OPTIMIZED
if "%MODE_CHOICE%"=="2" goto MODE_FULL_OPTIMIZED
if "%MODE_CHOICE%"=="3" goto MODE_OLD_SLOW
goto MODE_ERROR

:MODE_FAST_OPTIMIZED
set MODE=fast
set SCRIPT=fast_grid_search.py
echo.
echo [GEKOZEN] Fast mode GEOPTIMALISEERD
echo   - 384 configuraties
echo   - Geschatte tijd: 5-10 minuten
echo   - Data wordt 1x opgehaald (SNEL!)
goto START_GRID_SEARCH

:MODE_FULL_OPTIMIZED
set MODE=full
set SCRIPT=fast_grid_search.py
echo.
echo [GEKOZEN] Full mode GEOPTIMALISEERD
echo   - 40,960 configuraties
echo   - Geschatte tijd: 2-3 uur
echo.
echo ============================================================================
echo   !! LET OP !!
echo ============================================================================
echo   Full mode duurt UREN om te runnen!
echo   - Zorg dat laptop NIET in slaapstand gaat
echo   - Houd laptop aangesloten op stroom
echo   - Laat dit venster OPEN
echo   - Run dit 's nachts of in weekend
echo ============================================================================
echo.
pause
goto START_GRID_SEARCH

:MODE_OLD_SLOW
set MODE=fast
set SCRIPT=full_grid_search.py
echo.
echo [GEKOZEN] Old mode (langzaam - NIET AANGERADEN)
echo   - 384 configuraties
echo   - Geschatte tijd: 40 minuten
echo   - Waarschuwing: haalt data 384x opnieuw op!
echo.
echo OPMERKING: Kies liever optie 1 (10x sneller, zelfde resultaat!)
echo.
pause
goto START_GRID_SEARCH

:MODE_ERROR
echo.
echo [FOUT] Ongeldige keuze! Alleen 1, 2 of 3 is toegestaan.
pause
exit /b 1

:START_GRID_SEARCH
echo.
echo ============================================================================
echo   START GRID SEARCH - %MODE% MODE
echo ============================================================================
echo.
echo Script: %SCRIPT%
echo Starttijd: %date% %time%
echo.
echo Dit kan een tijdje duren...
echo NIET dit venster sluiten!
echo.

REM Start de grid search
python scripts\%SCRIPT% --mode %MODE% --train-days 180 --val-days 340

REM Check of het gelukt is
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================================
    echo   KLAAR! Grid search succesvol afgerond
    echo ============================================================================
    echo.
    echo Eindtijd: %date% %time%
    echo.
    echo Resultaten staan in:
    echo   - grid_search_train_%MODE%_*.csv
    echo   - grid_search_val_%MODE%_*.csv
    echo   - optimal_config_%MODE%_*.json
    echo.
    echo Open deze bestanden in Excel of een text editor!
    echo.
) else (
    echo.
    echo ============================================================================
    echo   FOUT! Er is iets misgegaan
    echo ============================================================================
    echo.
    echo Check de error messages hierboven.
    echo.
    echo Mogelijke oorzaken:
    echo   - MT5 is niet verbonden
    echo   - Python packages ontbreken
    echo   - Geen historische data beschikbaar
    echo   - Script niet gevonden (fast_grid_search.py)
    echo.
    echo Als fast_grid_search.py niet bestaat:
    echo   Download hem van Claude en zet in scripts\ folder
    echo.
)

echo.
echo Druk op een toets om dit venster te sluiten...
pause >nul
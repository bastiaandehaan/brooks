@echo off
echo ================================================================================
echo   BROOKS BACKTEST - OPTIMAL CONFIGURATION
echo ================================================================================
echo.
echo Configuration:
echo   - Days: 340
echo   - Regime Filter: ENABLED (threshold=2.0)
echo   - Stop Buffer: 1.0
echo   - Cooldown: 0
echo   - Costs: 0.04R per trade
echo.
echo Expected: Daily Sharpe ~1.426, Net R +92.12R, 722 trades
echo This will take ~30 seconds...
echo ================================================================================
echo.

.venv\Scripts\python.exe backtest\runner.py ^
    --days 340 ^
    --stop-buffer 1.0 ^
    --cooldown 0 ^
    --regime-filter ^
    --chop-threshold 2.0 ^
    --costs 0.04

echo.
echo ================================================================================
echo   BACKTEST COMPLETE - Check Daily Sharpe above!
echo ================================================================================
pause
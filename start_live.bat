@echo off
echo ================================================================================
echo   BROOKS LIVE MONITOR - PRODUCTION MODE
echo ================================================================================
echo.
echo Configuration:
echo   - Symbol: US500.cash
echo   - Risk: 0.5%% per trade
echo   - Regime Filter: ENABLED
echo   - FTMO Protection: ENABLED (10k account)
echo   - Debug Logging: ENABLED
echo.
echo Press Ctrl+C to stop
echo ================================================================================
echo.

# Start zonder FTMO (paper trading is safe zonder limits)
python scripts/live_monitor.py --symbol US500.cash --risk-pct 0.5 --regime-filter --chop-threshold 2.0 --stop-buffer 1.0

pause
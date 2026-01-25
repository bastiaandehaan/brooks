# scripts/check_backtest_period.py
import MetaTrader5 as mt5
import pandas as pd

mt5.initialize()

symbol = "US500.cash"
days = 340
count_m15 = days * 96

rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, count_m15)

if rates is not None:
    df = pd.DataFrame(rates)
    df["datetime"] = pd.to_datetime(df["time"], unit="s")

    print("=" * 70)
    print("BACKTEST PERIOD USED IN LAST 340 DAYS")
    print("=" * 70)
    print(f"Start: {df['datetime'].iloc[0]}")
    print(f"End:   {df['datetime'].iloc[-1]}")
    print(f"Bars:  {len(df)}")
    print("=" * 70)
else:
    print("Failed to fetch data")

mt5.shutdown()

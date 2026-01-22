# backtest/optimizer.py
"""
Simple parameter optimizer for Brooks strategy
"""

import os
import sys

# FIX: Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import itertools

import pandas as pd

# FIX: Import from correct location
from backtest.runner import run_backtest


def start_optimization():
    """Run basic parameter grid search"""
    # --- CONFIGURATIE ---
    SYMBOL = "US500.cash"
    DAYS = 60
    COSTS = 0.04

    # --- DE PARAMETER GRID ---
    grid = {
        "ema_period": [15, 20],
        "pullback_bars": [3, 4],
        "signal_close_frac": [0.25, 0.30],
        "chop_threshold": [2.0, 2.5],
        "regime_filter": [True],
        "stop_buffer": [1.5, 2.0],
    }

    # Maak alle combinaties
    keys, values = zip(*grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    all_results = []
    total = len(combinations)

    print(f"🚀 Start optimalisatie voor {SYMBOL}")
    print(f"📊 Totaal aantal combinaties: {total}")
    print("-" * 50)

    for i, params in enumerate(combinations, 1):
        print(f"[{i}/{total}] Testing...", end="", flush=True)

        # Voer de backtest uit
        res = run_backtest(
            symbol=SYMBOL,
            days=DAYS,
            ema_period=params["ema_period"],
            pullback_bars=params["pullback_bars"],
            signal_close_frac=params["signal_close_frac"],
            regime_filter=params["regime_filter"],
            chop_threshold=params["chop_threshold"],
            stop_buffer=params["stop_buffer"],
            costs_per_trade_r=COSTS,
            min_slope=0.15,  # Fixed
            min_risk_price_units=2.0,  # Fixed
            cooldown_bars=10,  # Fixed
            max_trades_day=2,  # Fixed
        )

        # Check if successful
        if "error" not in res and res.get("trades", 0) > 0:
            row = {**params, **res}
            all_results.append(row)
            print(f" ✓ Sharpe={res.get('daily_sharpe_r', 0):.3f}, Net={res.get('net_r', 0):.1f}R")
        else:
            print(" ✗ Failed")

    # --- ANALYSE ---
    if not all_results:
        print("❌ Geen resultaten gevonden")
        return

    df = pd.DataFrame(all_results)

    # Sorteer op Daily Sharpe
    top_sharpe = df.sort_values("daily_sharpe_r", ascending=False).head(5)

    print("\n" + "=" * 80)
    print("🏆 TOP 5 CONFIGURATIES (Geselecteerd op Daily Sharpe)")
    print("=" * 80)

    cols_to_show = [
        "ema_period",
        "pullback_bars",
        "signal_close_frac",
        "chop_threshold",
        "stop_buffer",
        "net_r",
        "daily_sharpe_r",
        "trades",
        "winrate",
    ]
    print(top_sharpe[cols_to_show].to_string(index=False))

    # Opslaan
    df.to_csv("optimization_results.csv", index=False)
    print(f"\n✅ Alle resultaten opgeslagen in 'optimization_results.csv'")

    # Best config
    best = top_sharpe.iloc[0]
    print("\n" + "=" * 80)
    print("🎯 BESTE CONFIGURATIE:")
    print("=" * 80)
    print(f"  EMA Period       : {best['ema_period']}")
    print(f"  Pullback Bars    : {best['pullback_bars']}")
    print(f"  Signal Close Frac: {best['signal_close_frac']:.2f}")
    print(f"  Chop Threshold   : {best['chop_threshold']:.1f}")
    print(f"  Stop Buffer      : {best['stop_buffer']:.1f}")
    print(f"\n  📊 PERFORMANCE:")
    print(f"  Daily Sharpe     : {best['daily_sharpe_r']:.3f}")
    print(f"  Net R            : {best['net_r']:+.1f}R")
    print(f"  Trades           : {best['trades']}")
    print(f"  Winrate          : {best['winrate'] * 100:.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    start_optimization()

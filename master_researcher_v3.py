#!/usr/bin/env python3
"""
MASTER RESEARCHER V3.3 - UTF-8 COMPLIANT
-----------------------------------------
FIX: Emojis verwijderd en encoding=utf-8 toegevoegd om crashes op Windows te voorkomen.
Nu wordt de CSV gegarandeerd gevuld.
"""

import csv
import os
import sys
from datetime import datetime

# Zorg dat Python de backtest module kan vinden
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtest.runner import run_backtest

# =====================================================
# CONFIGURATIE
# =====================================================
INSTRUMENTS = ["US500.cash", "US100.cash", "XAUUSD"]
TRAIN_DAYS = 250
TEST_DAYS = 90
COSTS = 0.04
CSV_FILENAME = f"research_logbook_STABLE_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"


def init_csv():
    headers = [
        "Symbol",
        "EMA",
        "Slope",
        "Filter_On",
        "Chop_Thresh",
        "Stop_Buff",
        "Max_Trades",
        "Train_Net_R",
        "Train_Sharpe",
        "Train_DD_Daily",
        "Train_Winrate",
        "Train_Trades",
        "Test_Net_R",
        "Test_Sharpe",
        "Test_DD_Daily",
        "Test_Winrate",
        "Test_Trades",
        "Combined_Score",
        "FTMO_Status",
    ]
    # Gebruik encoding='utf-8' voor Windows compatibiliteit
    with open(CSV_FILENAME, mode="w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(headers)
    print(f"Logboek aangemaakt: {CSV_FILENAME}")


def determine_ftmo_status(train_dd, test_dd):
    max_dd = max(train_dd, test_dd)
    if max_dd < 3.0:
        return "PASS (Perfect)"
    if max_dd < 5.0:
        return "PASS (Risky)"
    return "FAIL"


def run_research():
    print("\n" + "=" * 40)
    print("  MASTER RESEARCHER V3.3 - STABLE MODE")
    print("=" * 40 + "\n")

    init_csv()

    configs = []
    for ema in [15, 20]:
        for slope in [0.10, 0.15]:
            for stop in [1.0, 1.5]:
                for trades in [1, 2, 3]:
                    for filt_sett in [
                        {"regime_filter": False, "chop": 0},
                        {"regime_filter": True, "chop": 2.5},
                    ]:
                        configs.append(
                            {
                                "ema_period": ema,
                                "min_slope": slope,
                                "stop_buffer": stop,
                                "max_trades_day": trades,
                                "regime_filter": filt_sett["regime_filter"],
                                "chop_threshold": filt_sett["chop"],
                                "pullback_bars": 3,
                                "signal_close_frac": 0.20,
                                "min_risk_price_units": 1.5,
                            }
                        )

    total_runs = len(INSTRUMENTS) * len(configs)
    count = 0

    for symbol in INSTRUMENTS:
        print(f"\nSTART INSTRUMENT: {symbol}")
        for p in configs:
            count += 1
            print(
                f"[{count}/{total_runs}] {symbol} | EMA:{p['ema_period']} | Trades:{p['max_trades_day']}...",
                end="",
                flush=True,
            )

            try:
                # RUN TRAIN
                train = run_backtest(symbol=symbol, days=TRAIN_DAYS, costs_per_trade_r=COSTS, **p)
                if "error" in train or train.get("trades", 0) == 0:
                    print(" - Geen Trades")
                    continue

                # RUN TEST
                test = run_backtest(
                    symbol=symbol, days=TEST_DAYS, costs_per_trade_r=COSTS + 0.02, **p
                )

                # DATA VERWERKEN
                t_net, t_sharpe, t_dd = (
                    train.get("net_r", 0),
                    train.get("daily_sharpe_r", 0),
                    abs(train.get("max_dd_r_daily", 0)),
                )
                v_net, v_sharpe, v_dd = (
                    test.get("net_r", 0),
                    test.get("daily_sharpe_r", 0),
                    abs(test.get("max_dd_r_daily", 0)),
                )

                score = (0.6 * t_sharpe) + (0.4 * v_sharpe)
                status = determine_ftmo_status(t_dd, v_dd)

                # Print resultaat zonder emojis
                print(f" OK | R:{t_net:.1f} | DD:{t_dd:.1f}R | {status}")

                # OPSLAAN (Met encoding='utf-8')
                row = [
                    symbol,
                    p["ema_period"],
                    p["min_slope"],
                    p["regime_filter"],
                    p["chop_threshold"],
                    p["stop_buffer"],
                    p["max_trades_day"],
                    round(t_net, 2),
                    round(t_sharpe, 3),
                    round(t_dd, 2),
                    round(train.get("winrate", 0), 3),
                    train.get("trades", 0),
                    round(v_net, 2),
                    round(v_sharpe, 3),
                    round(v_dd, 2),
                    round(test.get("winrate", 0), 3),
                    test.get("trades", 0),
                    round(score, 3),
                    status,
                ]
                with open(CSV_FILENAME, mode="a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(row)

            except Exception as e:
                print(f" Fout bij opslaan: {e}")

    print(f"\nKLAAR! Resultaten in: {CSV_FILENAME}")


if __name__ == "__main__":
    run_research()

import sys
import os
import numpy as np  # Zorg dat je pip install numpy hebt gedaan

# Pad fix zodat imports werken
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)

import argparse
import logging
import pandas as pd
import MetaTrader5 as mt5

from utils.mt5_client import Mt5Client
from utils.mt5_data import fetch_rates, RatesRequest
from strategies.context import infer_trend_m15, TrendParams, Trend
from strategies.h2l2 import plan_h2l2_trades, H2L2Params, Side
from execution.guardrails import Guardrails, apply_guardrails

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("Backtest")


def run_backtest(symbol: str, days: int):
    # 1. Connectie & Data
    client = Mt5Client(mt5_module=mt5)
    if not client.initialize():
        logger.error("Kon niet verbinden met MT5.")
        return

    spec = client.get_symbol_specification(symbol)
    if not spec:
        logger.error(f"Geen specificaties voor {symbol}")
        return

    logger.info(f"--- STARTING EXTENDED BACKTEST: {symbol} ({days} days) ---")

    count_m5 = days * 288
    count_m15 = days * 96

    m15_data = fetch_rates(mt5, RatesRequest(symbol, mt5.TIMEFRAME_M15, count_m15))
    m5_data = fetch_rates(mt5, RatesRequest(symbol, mt5.TIMEFRAME_M5, count_m5))

    if m15_data is None or m5_data is None or m5_data.empty:
        logger.error("Geen data ontvangen.")
        client.shutdown()
        return

    # 2. Trend (MVP: Global Trend)
    trend_res, _ = infer_trend_m15(m15_data, TrendParams())

    if trend_res not in [Trend.BULL, Trend.BEAR]:
        logger.warning(f"Geen duidelijke trend ({trend_res}).")
        client.shutdown()
        return

    side = Side.LONG if trend_res == Trend.BULL else Side.SHORT

    # 3. Genereer Trades
    params = H2L2Params(min_risk_price_units=2.0, signal_close_frac=0.30)
    raw_trades = plan_h2l2_trades(m5_data, side, spec, params)

    # 4. Filter (Guardrails)
    g = Guardrails(max_trades_per_day=99, session_start="09:30", session_end="16:00")
    accepted_trades, _ = apply_guardrails(raw_trades, g)

    if not accepted_trades:
        logger.warning("Geen trades na filtering.")
        client.shutdown()
        return

    logger.info(f"Analyseren van {len(accepted_trades)} trades...")

    # 5. Simulatie Loop
    results_r = []

    for t in accepted_trades:
        future = m5_data.loc[t.execute_ts:]
        trade_outcome = 0.0

        # Simuleer verloop trade
        for ts, bar in future.iterrows():
            if t.side == Side.LONG:
                if bar["low"] <= t.stop:
                    trade_outcome = -1.0
                    break
                if bar["high"] >= t.tp:
                    trade_outcome = 2.0
                    break
            else:
                if bar["high"] >= t.stop:
                    trade_outcome = -1.0
                    break
                if bar["low"] <= t.tp:
                    trade_outcome = 2.0
                    break

        results_r.append(trade_outcome)

    # 6. Bereken Geavanceerde Metrics
    equity_curve = pd.Series(results_r).cumsum()

    # A. Max Drawdown (Diepte)
    running_max = equity_curve.cummax()
    drawdown = equity_curve - running_max
    max_dd_depth = drawdown.min()

    # B. Max Drawdown Period (Duur in trades)
    # We zoeken de langste reeks waarin we onder de 'running max' zitten
    is_underwater = drawdown < 0
    # Groepeer opeenvolgende True/False blokken
    dd_groups = (is_underwater != is_underwater.shift()).cumsum()
    # Tel lengte van blokken waar we underwater zijn
    underwater_lengths = is_underwater.groupby(dd_groups).sum()
    # Pak de langste
    max_dd_duration_trades = int(underwater_lengths.max()) if not underwater_lengths.empty else 0

    # C. Sharpe Ratio
    avg_return = np.mean(results_r)
    std_return = np.std(results_r)
    sharpe = (avg_return / std_return) * np.sqrt(len(results_r)) if std_return != 0 else 0

    # D. Profit Factor
    wins = [r for r in results_r if r > 0]
    losses = [r for r in results_r if r < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    pf = gross_profit / gross_loss if gross_loss != 0 else float('inf')

    winrate = (len(wins) / len(results_r)) * 100

    # 7. Print Rapport
    print("\n" + "=" * 45)
    print(f" UITGEBREID BACKTEST RAPPORT: {symbol}")
    print("=" * 45)
    print(f"Periode              : {days} dagen")
    print(f"Totaal Trades        : {len(results_r)}")
    print(f"Winrate              : {winrate:.1f}%")
    print("-" * 45)
    print(f"Netto Resultaat      : {equity_curve.iloc[-1]:.2f} R")
    print(f"Profit Factor        : {pf:.2f}")
    print(f"Sharpe Ratio         : {sharpe:.2f}")
    print("-" * 45)
    print(f"Max Drawdown (Diepte): {max_dd_depth:.2f} R")
    print(f"Max Drawdown (Duur)  : {max_dd_duration_trades} trades")
    print("=" * 45)

    client.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--symbol", type=str, default="US500.cash")
    args = parser.parse_args()

    # Gebruik -m backtest.runner om dit correct aan te roepen
    run_backtest(args.symbol, args.days)
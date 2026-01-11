import sys
import os
import numpy as np
import argparse
import logging
import pandas as pd
import MetaTrader5 as mt5

# Pad fix
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)

from utils.mt5_client import Mt5Client
from utils.mt5_data import fetch_rates, RatesRequest
from strategies.context import infer_trend_m15, TrendParams, Trend
from strategies.h2l2 import plan_next_open_trade, H2L2Params, Side
from execution.guardrails import Guardrails, apply_guardrails
from backtest.visualiser import generate_performance_report

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("Backtest")


def precalculate_trends(m15_df, params):
    """Berekent trends vectorized voor maximale snelheid."""
    logger.info("Trends pre-calculeren...")
    trends = []
    for i in range(len(m15_df)):
        slice_df = m15_df.iloc[:i + 1]
        trend, _ = infer_trend_m15(slice_df, params)
        trends.append(trend)
    m15_df['trend'] = trends
    return m15_df[['trend']]


def run_backtest(symbol: str, days: int):
    client = Mt5Client(mt5_module=mt5)
    if not client.initialize(): return

    spec = client.get_symbol_specification(symbol)
    if not spec:
        client.shutdown()
        return

    logger.info(f"--- OPTIMIZED BACKTEST: {symbol} ({days} days) ---")

    # Exacte counts voor 46R resultaat
    count_m5 = days * 288
    count_m15 = days * 96 * 2

    m15_data = fetch_rates(mt5, RatesRequest(symbol, mt5.TIMEFRAME_M15, count_m15))
    m5_data = fetch_rates(mt5, RatesRequest(symbol, mt5.TIMEFRAME_M5, count_m5))

    if m15_data is None or m5_data is None:
        logger.error("Data error.");
        client.shutdown();
        return

    trend_data = precalculate_trends(m15_data, TrendParams())

    m5_with_trend = pd.merge_asof(
        m5_data.sort_index(),
        trend_data.sort_index(),
        left_index=True,
        right_index=True,
        direction='backward'
    )

    strat_params = H2L2Params(use_atr_risk=True, signal_close_frac=0.30)
    raw_trades = []

    start_idx = 200
    logger.info(f"Simuleren van {len(m5_with_trend)} bars...")

    for i in range(start_idx, len(m5_with_trend)):
        current_trend = m5_with_trend.iloc[i]['trend']
        if not current_trend: continue

        side = Side.LONG if current_trend == Trend.BULL else Side.SHORT
        m5_slice = m5_with_trend.iloc[i - 50: i + 1]

        trade = plan_next_open_trade(m5_slice, side, spec, strat_params, 5)
        if trade: raw_trades.append(trade)

    g = Guardrails(max_trades_per_day=5, session_start="09:30", session_end="16:00")
    accepted_trades, _ = apply_guardrails(raw_trades, g)

    if not accepted_trades:
        logger.warning("Geen trades.");
        client.shutdown();
        return

    results_r = []
    trade_details = []
    for t in accepted_trades:
        future = m5_data.loc[t.execute_ts:].iloc[1:]
        outcome = 0.0
        for ts, bar in future.iterrows():
            if t.side == Side.LONG:
                if bar["low"] <= t.stop: outcome = -1.0; break
                if bar["high"] >= t.tp: outcome = 2.0; break
            else:
                if bar["high"] >= t.stop: outcome = -1.0; break
                if bar["low"] <= t.tp: outcome = 2.0; break
        results_r.append(outcome)
        trade_details.append({'ts': t.execute_ts, 'result': outcome, 'side': t.side})

    # --- Uitgebreide Metrics ---
    res_series = pd.Series(results_r)
    equity_curve = res_series.cumsum()
    running_max = equity_curve.cummax()
    drawdown = equity_curve - running_max

    max_dd_depth = drawdown.min()
    is_underwater = drawdown < 0
    dd_groups = (is_underwater != is_underwater.shift()).cumsum()
    max_dd_duration = int(is_underwater.groupby(dd_groups).sum().max())

    sharpe = (res_series.mean() / res_series.std()) * np.sqrt(len(results_r)) if res_series.std() != 0 else 0
    pf = sum([r for r in results_r if r > 0]) / abs(sum([r for r in results_r if r < 0]))
    winrate = (res_series > 0).mean() * 100

    # Dashboard genereren
    generate_performance_report(results_r, equity_curve, drawdown, symbol, days)

    # Volledige terminal output
    print("\n" + "=" * 45)
    print(f" FINAL OPTIMIZED REPORT: {symbol}")
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
    print(f"Max Drawdown (Duur)  : {max_dd_duration} trades")
    print("=" * 45)

    client.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--symbol", type=str, default="US500.cash")
    args = parser.parse_args()
    run_backtest(args.symbol, args.days)
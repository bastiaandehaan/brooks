# backtest/runner.py
"""
Brooks Backtest Runner - INTELLIGENT OUTPUT
Clean metrics, multi-timeframe support, strategy focus
"""
import sys
import os
import time
import numpy as np
import argparse
import logging
import pandas as pd
import MetaTrader5 as mt5
from typing import Dict, Any

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)

from utils.mt5_client import Mt5Client
from utils.mt5_data import fetch_rates, RatesRequest
from strategies.context import TrendParams, Trend, infer_trend_m15_series
from strategies.h2l2 import plan_h2l2_trades, H2L2Params, Side, PlannedTrade
from execution.guardrails import Guardrails, apply_guardrails

# Suppress guardrail spam logging
import logging as _log

_log.getLogger("execution.guardrails").setLevel(_log.WARNING)
from execution.selection import select_top_per_ny_day
from backtest.visualiser import generate_performance_report

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("Backtest")

NY_TZ = "America/New_York"


def _normalize_ohlc(df: pd.DataFrame, *, name: str) -> pd.DataFrame:
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name}: missing columns: {sorted(missing)}")

    out = df.sort_index()
    out = out.loc[~out.index.duplicated(keep="first")]
    return out


def precalculate_trends(m15_df: pd.DataFrame, params: TrendParams) -> pd.DataFrame:
    logger.info("→ Calculating trends (vectorized)...")
    t0 = time.perf_counter()
    trend_series = infer_trend_m15_series(m15_df, params)
    out = pd.DataFrame({"trend": trend_series}, index=m15_df.index)
    logger.info("  Trend calc: %.2fs", time.perf_counter() - t0)
    return out


def _trend_to_side(trend: Trend) -> Side | None:
    if trend == Trend.BULL:
        return Side.LONG
    if trend == Trend.BEAR:
        return Side.SHORT
    return None


def _simulate_trade_outcome(m5_data: pd.DataFrame, t: PlannedTrade) -> float:
    future = m5_data.loc[t.execute_ts:]
    for _, bar in future.iterrows():
        high = float(bar["high"])
        low = float(bar["low"])

        if t.side == Side.LONG:
            hit_sl = low <= t.stop
            hit_tp = high >= t.tp
            if hit_sl and hit_tp:
                return -1.0
            if hit_sl:
                return -1.0
            if hit_tp:
                return 2.0
        else:
            hit_sl = high >= t.stop
            hit_tp = low <= t.tp
            if hit_sl and hit_tp:
                return -1.0
            if hit_sl:
                return -1.0
            if hit_tp:
                return 2.0
    return 0.0


def run_backtest(
        symbol: str,
        days: int,
        max_trades_day: int = 2,
        # STRATEGY PARAMS (expose for grid search)
        min_slope: float = 0.15,
        ema_period: int = 20,
        pullback_bars: int = 3,
        signal_close_frac: float = 0.30,
        stop_buffer: float = 2.0,
        min_risk_price_units: float = 2.0,
        cooldown_bars: int = 10,
) -> Dict[str, Any]:
    """
    Returns metrics dict for grid search compatibility.
    """
    client = Mt5Client(mt5_module=mt5)
    if not client.initialize():
        return {"error": "MT5 init failed"}

    spec = client.get_symbol_specification(symbol)
    if not spec:
        client.shutdown()
        return {"error": "Symbol not found"}

    print("\n" + "=" * 80)
    print(f"  BROOKS BACKTEST: {symbol} ({days} days)")
    print("=" * 80)
    print(f"\n📊 STRATEGY PARAMETERS:")
    print(f"  Context    : min_slope={min_slope:.2f}, ema_period={ema_period}")
    print(f"  H2/L2      : pullback={pullback_bars}, close_frac={signal_close_frac:.2f}")
    print(f"  Risk       : stop_buffer={stop_buffer:.1f}, min_risk={min_risk_price_units:.1f}")
    print(f"  Execution  : cooldown={cooldown_bars}bars, max_trades_day={max_trades_day}")
    print()

    count_m5 = days * 288
    count_m15 = days * 96 * 2

    logger.info("→ Fetching data...")
    m15_data = fetch_rates(mt5, RatesRequest(symbol, mt5.TIMEFRAME_M15, count_m15))
    m5_data = fetch_rates(mt5, RatesRequest(symbol, mt5.TIMEFRAME_M5, count_m5))

    if m15_data.empty or m5_data.empty:
        logger.warning("Empty data!")
        client.shutdown()
        return {"error": "Empty data"}

    m15_data = _normalize_ohlc(m15_data, name="M15")
    m5_data = _normalize_ohlc(m5_data, name="M5")

    logger.info("  M15: %d bars, M5: %d bars", len(m15_data), len(m5_data))

    # Trends
    m15_trends = precalculate_trends(
        m15_data,
        TrendParams(min_slope=min_slope, ema_period=ema_period)
    )

    # Merge
    trend_series = m15_trends.copy()
    trend_series = trend_series.reset_index().rename(columns={"index": "ts"})
    m5_ts = m5_data.reset_index().rename(columns={"index": "ts"})
    merged = pd.merge_asof(
        m5_ts.sort_values("ts"),
        trend_series.sort_values("ts"),
        on="ts",
        direction="backward",
    )
    m5_data = m5_data.copy()
    m5_data["trend"] = merged["trend"].values

    # Trend distribution
    trend_counts = m5_data["trend"].value_counts()
    bull_bars = trend_counts.get(Trend.BULL, 0)
    bear_bars = trend_counts.get(Trend.BEAR, 0)
    none_bars = m5_data["trend"].isna().sum()

    print(f"📈 TREND DISTRIBUTION:")
    print(f"  Bull  : {bull_bars:5d} bars ({bull_bars / len(m5_data) * 100:5.1f}%)")
    print(f"  Bear  : {bear_bars:5d} bars ({bear_bars / len(m5_data) * 100:5.1f}%)")
    print(f"  Range : {none_bars:5d} bars ({none_bars / len(m5_data) * 100:5.1f}%)")
    print()

    # Strategy params
    strat_params = H2L2Params(
        pullback_bars=pullback_bars,
        signal_close_frac=signal_close_frac,
        min_risk_price_units=min_risk_price_units,
        stop_buffer=stop_buffer,
        cooldown_bars=cooldown_bars,
    )

    planned_trades: list[PlannedTrade] = []

    logger.info("→ Planning trades (segment-based)...")
    t_plan0 = time.perf_counter()

    total_bars = len(m5_data)

    # Segment-based
    segments = []
    current_trend = None
    segment_start = 50

    for i in range(50, total_bars):
        trend_val = m5_data.iloc[i]["trend"]
        side = _trend_to_side(trend_val) if not pd.isna(trend_val) else None

        if side != current_trend:
            if current_trend is not None and segment_start < i:
                segments.append((segment_start, i, current_trend))
            current_trend = side
            segment_start = i

    if current_trend is not None and segment_start < total_bars:
        segments.append((segment_start, total_bars, current_trend))

    for seg_idx, (start_idx, end_idx, trend_side) in enumerate(segments):
        lookback_start = max(0, start_idx - 10)
        segment_data = m5_data.iloc[lookback_start:end_idx]

        trades = plan_h2l2_trades(segment_data, trend_side, spec, strat_params)

        segment_start_ts = m5_data.index[start_idx]
        for t in trades:
            if t.execute_ts >= segment_start_ts:
                planned_trades.append(t)

    planning_time = time.perf_counter() - t_plan0
    logger.info("  Planning: %.2fs, %d segments → %d candidates",
                planning_time, len(segments), len(planned_trades))

    # Guardrails (silent mode)
    logger.info("→ Applying guardrails (session + daily limit)...")
    g_session = Guardrails(
        session_tz=NY_TZ,
        day_tz=NY_TZ,
        session_start="09:30",
        session_end="15:00",
        max_trades_per_day=10_000,
    )
    in_session, rejected1 = apply_guardrails(planned_trades, g_session)

    selected, sel_stats = select_top_per_ny_day(
        in_session,
        max_trades_day=max_trades_day,
        tick_size=float(spec.tick_size),
    )

    g_all = Guardrails(
        session_tz=NY_TZ,
        day_tz=NY_TZ,
        session_start="09:30",
        session_end="15:00",
        max_trades_per_day=max_trades_day,
    )
    final_trades, rejected2 = apply_guardrails(selected, g_all)

    print(f"🔍 TRADE PIPELINE:")
    print(f"  Candidates    : {len(planned_trades):4d}")
    print(f"  In session    : {len(in_session):4d} (rejected: {len(rejected1)})")
    print(f"  After select  : {len(selected):4d} (days: {len(sel_stats)})")
    print(f"  Final         : {len(final_trades):4d}")
    print()

    # Simulate
    results_r: list[float] = []
    for t in final_trades:
        results_r.append(_simulate_trade_outcome(m5_data, t))

    if not results_r:
        logger.info("⚠️  No trades executed.")
        client.shutdown()
        return {"error": "No trades"}

    res = pd.Series(results_r, dtype="float64")
    equity_curve = res.cumsum()
    running_max = equity_curve.cummax()
    drawdown = equity_curve - running_max
    max_dd = float(drawdown.min())

    # DD duration
    dd_bars = 0
    max_dd_bars = 0
    for dd_val in drawdown:
        if dd_val < 0:
            dd_bars += 1
            max_dd_bars = max(max_dd_bars, dd_bars)
        else:
            dd_bars = 0

    wins = res[res > 0].sum()
    losses = -res[res < 0].sum()
    profit_factor = float(wins / losses) if losses > 0 else float("inf")

    mean_r = float(res.mean())
    std_r = float(res.std(ddof=1)) if len(res) > 1 else 0.0
    sharpe = float(mean_r / std_r) if std_r > 0 else 0.0

    winrate = float((res > 0).sum() / len(res)) if len(res) else 0.0

    # Side breakdown
    long_trades = [t for t in final_trades if t.side == Side.LONG]
    short_trades = [t for t in final_trades if t.side == Side.SHORT]

    long_results = [r for t, r in zip(final_trades, results_r) if t.side == Side.LONG]
    short_results = [r for t, r in zip(final_trades, results_r) if t.side == Side.SHORT]

    long_wr = sum(1 for r in long_results if r > 0) / len(long_results) if long_results else 0
    short_wr = sum(1 for r in short_results if r > 0) / len(short_results) if short_results else 0

    # Output
    print("=" * 80)
    print("  📊 RESULTS")
    print("=" * 80)
    print(f"\n💰 PERFORMANCE:")
    print(f"  Trades        : {len(res):4d}")
    print(f"  Net R         : {float(equity_curve.iloc[-1]):+7.2f}R")
    print(f"  Avg R/trade   : {mean_r:+7.4f}R")
    print(f"  Sharpe Ratio  : {sharpe:7.3f}")
    print(f"  Profit Factor : {profit_factor:7.2f}")

    print(f"\n📈 WIN/LOSS:")
    print(f"  Winrate       : {winrate * 100:6.2f}%")
    print(f"  Winners       : {int((res > 0).sum()):4d} ({int((res > 0).sum()) / len(res) * 100:5.1f}%)")
    print(f"  Losers        : {int((res < 0).sum()):4d} ({int((res < 0).sum()) / len(res) * 100:5.1f}%)")
    print(f"  Breakeven     : {int((res == 0).sum()):4d}")

    print(f"\n📉 DRAWDOWN:")
    print(f"  Max DD        : {max_dd:7.2f}R")
    print(f"  Max DD bars   : {max_dd_bars:4d} trades")
    print(f"  DD % of equity: {abs(max_dd) / float(equity_curve.iloc[-1]) * 100:6.2f}%")

    print(f"\n⚖️  SIDE BREAKDOWN:")
    print(f"  Long trades   : {len(long_trades):4d} (WR: {long_wr * 100:5.1f}%)")
    print(f"  Short trades  : {len(short_trades):4d} (WR: {short_wr * 100:5.1f}%)")

    print(f"\n⏱️  PERFORMANCE:")
    print(f"  Planning      : {planning_time:6.2f}s")
    print(f"  Bars/second   : {total_bars / planning_time:8.1f}")

    print("\n" + "=" * 80 + "\n")

    generate_performance_report(results_r, equity_curve, drawdown, symbol=symbol, days=days)
    client.shutdown()

    return {
        "days": days,
        "trades": len(res),
        "net_r": float(equity_curve.iloc[-1]),
        "winrate": winrate,
        "sharpe": sharpe,
        "profit_factor": profit_factor,
        "max_dd": max_dd,
        "avg_r": mean_r,
        "long_wr": long_wr,
        "short_wr": short_wr,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="US500.cash")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--max-trades-day", type=int, default=2)

    # Strategy params (for grid search)
    parser.add_argument("--min-slope", type=float, default=0.15)
    parser.add_argument("--ema-period", type=int, default=20)
    parser.add_argument("--pullback-bars", type=int, default=3)
    parser.add_argument("--signal-close-frac", type=float, default=0.30)
    parser.add_argument("--stop-buffer", type=float, default=2.0)
    parser.add_argument("--min-risk", type=float, default=2.0)
    parser.add_argument("--cooldown", type=int, default=10)

    args = parser.parse_args()

    run_backtest(
        args.symbol,
        args.days,
        max_trades_day=args.max_trades_day,
        min_slope=args.min_slope,
        ema_period=args.ema_period,
        pullback_bars=args.pullback_bars,
        signal_close_frac=args.signal_close_frac,
        stop_buffer=args.stop_buffer,
        min_risk_price_units=args.min_risk,
        cooldown_bars=args.cooldown,
    )
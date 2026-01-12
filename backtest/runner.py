# backtest/runner.py
import sys
import os
import time
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
from strategies.context import TrendParams, Trend, infer_trend_m15_series
from strategies.h2l2 import plan_next_open_trade, H2L2Params, Side, PlannedTrade
from execution.guardrails import Guardrails, apply_guardrails
from execution.selection import select_top_per_ny_day
from backtest.visualiser import generate_performance_report

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("Backtest")

NY_TZ = "America/New_York"


def _normalize_ohlc(df: pd.DataFrame, *, name: str) -> pd.DataFrame:
    """Sort index, drop duplicate timestamps, validate OHLC schema."""
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name}: missing columns: {sorted(missing)}")

    out = df.sort_index()
    out = out.loc[~out.index.duplicated(keep="first")]
    return out


def _format_eta(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "?"
    s = int(seconds)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def precalculate_trends(m15_df: pd.DataFrame, params: TrendParams) -> pd.DataFrame:
    """
    Fast trend precalc (O(n)): uses infer_trend_m15_series once.
    No look-ahead: ewm/rolling are causal.
    """
    logger.info("Trends pre-calculeren (vectorized)...")
    t0 = time.perf_counter()
    trend_series = infer_trend_m15_series(m15_df, params)
    out = pd.DataFrame({"trend": trend_series}, index=m15_df.index)
    logger.info("Trend precalc done: bars=%d elapsed=%.3fs", len(out), time.perf_counter() - t0)
    return out


def _trend_to_side(trend: Trend) -> Side | None:
    if trend == Trend.BULL:
        return Side.LONG
    if trend == Trend.BEAR:
        return Side.SHORT
    return None


def _simulate_trade_outcome(m5_data: pd.DataFrame, t: PlannedTrade) -> float:
    """
    Simuleer trade outcome in R.
    Policy:
      - execute bar telt mee
      - SL en TP in dezelfde bar => worst-case SL (-1R)
    """
    future = m5_data.loc[t.execute_ts:]  # INCLUDE execute bar
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


def run_backtest(symbol: str, days: int, max_trades_day: int = 2) -> None:
    if max_trades_day < 0:
        raise ValueError("max_trades_day must be >= 0")

    client = Mt5Client(mt5_module=mt5)
    if not client.initialize():
        return

    spec = client.get_symbol_specification(symbol)
    if not spec:
        client.shutdown()
        return

    logger.info(f"--- OPTIMIZED BACKTEST: {symbol} ({days} days) ---")

    count_m5 = days * 288
    count_m15 = days * 96 * 2  # extra trend-history

    m15_data = fetch_rates(mt5, RatesRequest(symbol, mt5.TIMEFRAME_M15, count_m15))
    m5_data = fetch_rates(mt5, RatesRequest(symbol, mt5.TIMEFRAME_M5, count_m5))

    if m15_data.empty or m5_data.empty:
        logger.warning("Geen data opgehaald (m15=%d, m5=%d).", len(m15_data), len(m5_data))
        client.shutdown()
        return

    m15_data = _normalize_ohlc(m15_data, name="M15")
    m5_data = _normalize_ohlc(m5_data, name="M5")

    # Trends pre-calc (fast, no look-ahead)
    m15_trends = precalculate_trends(m15_data, TrendParams())

    # Merge trend into m5 timeline (asof backward)
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

    strat_params = H2L2Params()
    planned_trades: list[PlannedTrade] = []

    # >>> MVP PERFORMANCE FIX: cap history passed into planner <<<
    lookback_bars = 400  # tune later if needed; must cover your H2/L2 pattern needs

    logger.info("Planning trades over M5 bars...")
    t_plan0 = time.perf_counter()

    total_bars = len(m5_data)

    # Start after some warmup
    for i in range(50, total_bars - 1):
        # Progress update every 250 bars
        if i % 250 == 0 and i > 0:
            elapsed = time.perf_counter() - t_plan0
            bars_per_sec = i / elapsed if elapsed > 0 else 0.0
            remaining = (total_bars - i)
            eta_sec = (remaining / bars_per_sec) if bars_per_sec > 0 else float("inf")
            pct = (i / total_bars) * 100.0

            logger.info(
                "planning: %5.1f%% i=%d/%d planned=%d speed=%.1f bars/s ETA=%s",
                pct,
                i,
                total_bars,
                len(planned_trades),
                bars_per_sec,
                _format_eta(eta_sec),
            )

        trend_val = m5_data.iloc[i]["trend"]
        if pd.isna(trend_val):
            continue

        side = _trend_to_side(trend_val)
        if side is None:
            continue

        # Only bars up to (signal bar + next open bar), BUT cap lookback for speed
        end = i + 2
        start = max(0, end - lookback_bars)
        m5_slice = m5_data.iloc[start:end]

        # simulated "now" for backtest semantics (NEXT_OPEN)
        simulated_now_utc = m5_slice.index[-1]

        trade = plan_next_open_trade(
            m5=m5_slice,
            trend=side,
            spec=spec,
            p=strat_params,
            timeframe_minutes=5,
            now_utc=simulated_now_utc,
        )
        if trade is not None:
            planned_trades.append(trade)

    logger.info(
        "Planning done: bars=%d elapsed=%.2fs planned=%d",
        total_bars,
        time.perf_counter() - t_plan0,
        len(planned_trades),
    )

    logger.info("Planned trades: %d", len(planned_trades))

    # Guardrails: session-only filter first (no max/day here)
    g_session = Guardrails(
        session_tz=NY_TZ,
        day_tz=NY_TZ,
        session_start="09:30",
        session_end="15:00",
        max_trades_per_day=10_000,
    )
    in_session, rejected1 = apply_guardrails(planned_trades, g_session)
    logger.info("After session guardrails: %d (rejected: %d)", len(in_session), len(rejected1))

    # Deterministic daily selection (tick-quantized risk ordering)
    selected, sel_stats = select_top_per_ny_day(
        in_session,
        max_trades_day=max_trades_day,
        tick_size=float(spec.tick_size),
    )
    logger.info("After daily selection: %d (days logged: %d)", len(selected), len(sel_stats))

    # Final guardrails: enforce per-day + one-trade-per-timestamp
    g_all = Guardrails(
        session_tz=NY_TZ,
        day_tz=NY_TZ,
        session_start="09:30",
        session_end="15:00",
        max_trades_per_day=max_trades_day,
    )
    final_trades, rejected2 = apply_guardrails(selected, g_all)
    logger.info("After final guardrails: %d (rejected: %d)", len(final_trades), len(rejected2))

    logger.info(
        "runner: candidates=%d after_session=%d after_selection=%d after_final=%d max_trades_day=%d",
        len(planned_trades),
        len(in_session),
        len(selected),
        len(final_trades),
        max_trades_day,
    )

    # Simulate outcomes in R
    results_r: list[float] = []
    for t in final_trades:
        results_r.append(_simulate_trade_outcome(m5_data, t))

    if not results_r:
        logger.info("Geen trades uitgevoerd na filters/selection.")
        client.shutdown()
        return

    res = pd.Series(results_r, dtype="float64")
    equity_curve = res.cumsum()
    running_max = equity_curve.cummax()
    drawdown = equity_curve - running_max
    max_dd = float(drawdown.min())

    wins = res[res > 0].sum()
    losses = -res[res < 0].sum()
    profit_factor = float(wins / losses) if losses > 0 else float("inf")

    mean_r = float(res.mean())
    std_r = float(res.std(ddof=1)) if len(res) > 1 else 0.0
    sharpe = float(mean_r / std_r) if std_r > 0 else 0.0

    logger.info(
        "RESULTS: trades=%d netR=%.2f PF=%.2f maxDD=%.2f sharpe=%.2f",
        len(res),
        float(equity_curve.iloc[-1]),
        profit_factor,
        max_dd,
        sharpe,
    )

    generate_performance_report(results_r, equity_curve, drawdown)
    client.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="US500.cash")
    parser.add_argument("--days", type=int, default=10)
    parser.add_argument("--max-trades-day", type=int, default=2)
    args = parser.parse_args()

    run_backtest(args.symbol, args.days, max_trades_day=args.max_trades_day)

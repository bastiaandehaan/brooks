# backtest/runner.py
"""
Brooks Backtest Runner - WITH REGIME FILTER & COSTS & R-BASED DAILY METRICS

Strategy is unchanged. Only reporting/metrics:
- Real exit timestamps from simulation (no +2h placeholder)
- Daily metrics computed on daily PnL in R-units (option 1)
- PNG dashboard saved to backtest/backtest_png via visualiser
"""
from __future__ import annotations

import sys
import os
import time
import numpy as np
import argparse
import logging
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime
from typing import Dict, Any, List, Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)

from utils.mt5_client import Mt5Client
from utils.mt5_data import fetch_rates, RatesRequest
from strategies.context import TrendParams, Trend, infer_trend_m15_series
from strategies.h2l2 import plan_h2l2_trades, H2L2Params, Side, PlannedTrade
from strategies.regime import RegimeParams, detect_regime_series, MarketRegime
from execution.guardrails import Guardrails, apply_guardrails
from strategies.config import StrategyConfig
config = StrategyConfig.from_yaml("config/production.yaml")

# Suppress guardrail spam logging
import logging as _log
_log.getLogger("execution.guardrails").setLevel(_log.WARNING)

from execution.selection import select_top_per_ny_day
_log.getLogger("execution.selection").setLevel(_log.WARNING)

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


def _simulate_trade_outcome(m5_data: pd.DataFrame, t: PlannedTrade) -> tuple[float, pd.Timestamp]:
    """
    Simulate trade outcome with worst-case both-hit policy.
    Returns (R-value BEFORE costs, exit_ts).

    If no hit, exit_ts is last available bar time.
    """
    future = m5_data.loc[t.execute_ts:]
    last_ts = pd.to_datetime(future.index[-1]) if len(future) else pd.to_datetime(t.execute_ts)

    for ts, bar in future.iterrows():
        high = float(bar["high"])
        low = float(bar["low"])
        ts = pd.to_datetime(ts)

        if t.side == Side.LONG:
            hit_sl = low <= t.stop
            hit_tp = high >= t.tp
            if hit_sl and hit_tp:
                return -1.0, ts
            if hit_sl:
                return -1.0, ts
            if hit_tp:
                return 2.0, ts
        else:
            hit_sl = high >= t.stop
            hit_tp = low <= t.tp
            if hit_sl and hit_tp:
                return -1.0, ts
            if hit_sl:
                return -1.0, ts
            if hit_tp:
                return 2.0, ts

    return 0.0, last_ts


def _apply_costs(result_r: float, costs_r: float) -> float:
    return result_r - costs_r


def _build_trades_dataframe(
    final_trades: List[PlannedTrade],
    results_r: List[float],
    exit_ts_list: List[pd.Timestamp],
    *,
    m5_data: pd.DataFrame,
    ny_tz: str,
) -> pd.DataFrame:
    rows = []
    for trade, result, exit_ts in zip(final_trades, results_r, exit_ts_list):
        entry_ts = pd.to_datetime(trade.execute_ts)
        exit_ts = pd.to_datetime(exit_ts)

        regime_val = None
        chop_ratio = None
        try:
            row = m5_data.loc[trade.execute_ts]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            regime_val = row.get("regime", None)
            chop_ratio = row.get("chop_ratio", None)
        except Exception:
            pass

        rows.append(
            {
                "entry_time": entry_ts,
                "exit_time": exit_ts,
                "side": trade.side.value,
                "entry": float(trade.entry),
                "stop": float(trade.stop),
                "tp": float(trade.tp),
                "net_r": float(result),
                "reason": str(trade.reason),
                "regime_at_entry": str(regime_val) if regime_val is not None else None,
                "chop_ratio_at_entry": float(chop_ratio) if chop_ratio is not None else None,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    df = df.sort_values("entry_time").reset_index(drop=True)

    # Group by NY calendar day using exit_time.
    # Assumption: naive timestamps are UTC (MT5 Python API times are typically UTC).
    if df["exit_time"].dt.tz is None:
        df["ny_day"] = df["exit_time"].dt.tz_localize("UTC").dt.tz_convert(ny_tz).dt.date
    else:
        df["ny_day"] = df["exit_time"].dt.tz_convert(ny_tz).dt.date

    return df


def _max_consecutive_losses(trade_pnl: List[float]) -> int:
    max_run = 0
    run = 0
    for r in trade_pnl:
        if r < 0:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return int(max_run)


def _daily_series_from_trades(
    trades_df: pd.DataFrame,
    *,
    start_dt: Optional[pd.Timestamp],
    end_dt: Optional[pd.Timestamp],
    ny_tz: str,
) -> pd.Series:
    """
    Daily PnL in R units indexed by NY calendar day (date objects).
    Ensures all calendar days in [start, end] exist (0R on no-trade days).
    """
    if trades_df.empty:
        return pd.Series(dtype="float64")

    daily = trades_df.groupby("ny_day")["net_r"].sum().astype("float64")

    if start_dt is not None and end_dt is not None:
        s = pd.to_datetime(start_dt)
        e = pd.to_datetime(end_dt)

        # interpret naive timestamps as UTC then convert to NY for day range
        if s.tzinfo is None:
            s = s.tz_localize("UTC").tz_convert(ny_tz)
        else:
            s = s.tz_convert(ny_tz)

        if e.tzinfo is None:
            e = e.tz_localize("UTC").tz_convert(ny_tz)
        else:
            e = e.tz_convert(ny_tz)

        full_days = pd.date_range(s.date(), e.date(), freq="D").date
        daily = daily.reindex(full_days, fill_value=0.0)

    return daily


def _compute_manager_metrics_r_based(
    trades_df: pd.DataFrame,
    *,
    daily_pnl_r: pd.Series,          # R per NY-day
    trading_days_per_year: int,
    initial_capital: float,          # only for "% of initial" reporting
) -> Dict[str, Any]:
    """
    Option 1: all daily risk/return metrics computed on daily PnL in R-units (no pct_change).
    """
    out: Dict[str, Any] = {}
    if daily_pnl_r.empty:
        return out

    dp = pd.Series(daily_pnl_r, dtype="float64").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    mu = float(dp.mean())
    sig = float(dp.std(ddof=1)) if len(dp) > 1 else 0.0
    daily_sharpe_r = (mu / sig) * np.sqrt(trading_days_per_year) if sig > 0 else 0.0

    downside = dp[dp < 0]
    dsig = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    daily_sortino_r = (mu / dsig) * np.sqrt(trading_days_per_year) if dsig > 0 else 0.0

    # Daily equity in R-units
    eq_r = dp.cumsum()
    run_max = eq_r.cummax()
    dd_r = eq_r - run_max
    max_dd_r_daily = float(dd_r.min())

    # Explicitly: percent of initial capital (NOT a portfolio return)
    max_dd_pct_initial = (max_dd_r_daily / initial_capital) * 100.0 if initial_capital else 0.0

    best_day_r = float(dp.max())
    worst_day_r = float(dp.min())

    var_95_r = float(np.nanpercentile(dp.values, 5)) if len(dp) else 0.0
    cvar_95_r = float(dp[dp <= var_95_r].mean()) if len(dp) else 0.0

    skew_r = float(dp.skew()) if len(dp) > 2 else 0.0
    kurtosis_r = float(dp.kurtosis()) if len(dp) > 3 else 0.0

    pct_pos_days = float((dp > 0).mean() * 100.0) if len(dp) else 0.0

    underwater = dd_r < 0
    max_underwater_days = 0
    run = 0
    for v in underwater.astype(int).values:
        if v == 1:
            run += 1
            max_underwater_days = max(max_underwater_days, run)
        else:
            run = 0

    total_calendar_days = int(len(dp))
    days_with_trades = int((dp != 0).sum())
    trades_total = int(len(trades_df))
    trades_per_active_day = (trades_total / days_with_trades) if days_with_trades else 0.0
    trades_per_calendar_day = (trades_total / total_calendar_days) if total_calendar_days else 0.0

    out.update(
        {
            "daily_sharpe_r": daily_sharpe_r,
            "daily_sortino_r": daily_sortino_r,
            "max_dd_r_daily": max_dd_r_daily,
            "max_dd_pct_initial": max_dd_pct_initial,
            "best_day_r": best_day_r,
            "worst_day_r": worst_day_r,
            "var_95_r": var_95_r,
            "cvar_95_r": cvar_95_r,
            "skew_r": skew_r,
            "kurtosis_r": kurtosis_r,
            "pct_pos_days": pct_pos_days,
            "max_underwater_days": int(max_underwater_days),
            "calendar_days": total_calendar_days,
            "days_with_trades": days_with_trades,
            "trades_per_active_day": trades_per_active_day,
            "trades_per_calendar_day": trades_per_calendar_day,
        }
    )

    return out


def run_backtest(
    symbol: str,
    days: int,
    max_trades_day: int = 2,
    # STRATEGY PARAMS
    min_slope: float = 0.15,
    ema_period: int = 20,
    pullback_bars: int = 3,
    signal_close_frac: float = 0.30,
    stop_buffer: float = 2.0,
    min_risk_price_units: float = 2.0,
    cooldown_bars: int = 10,
    # REGIME FILTER
    regime_filter: bool = False,
    chop_threshold: float = 2.5,
    # COSTS
    costs_per_trade_r: float = 0.0,
    # METRICS
    initial_capital: float = 10000.0,
    trading_days_per_year: int = 252,
) -> Dict[str, Any]:
    client = Mt5Client(mt5_module=mt5)
    if not client.initialize():
        return {"error": "MT5 init failed"}

    spec = client.get_symbol_specification(symbol)
    if not spec:
        client.shutdown()
        return {"error": "Symbol not found"}

    print("\n" + "=" * 80)
    print(f"  BROOKS BACKTEST: {symbol} ({days} days)")
    if regime_filter:
        print(f"  🔎 REGIME FILTER: ENABLED (chop_threshold={chop_threshold})")
    else:
        print("  ⚠️  REGIME FILTER: DISABLED")
    if costs_per_trade_r > 0:
        print(f"  💸 COSTS: {costs_per_trade_r:.4f}R per trade")
    print("=" * 80)
    print("\n📊 STRATEGY PARAMETERS:")
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

    # Regime (optional)
    regime_data = None
    if regime_filter:
        logger.info("→ Calculating market regime (vectorized)...")
        t0 = time.perf_counter()

        regime_params = RegimeParams(chop_threshold=chop_threshold)
        regime_series = detect_regime_series(m15_data, regime_params)

        high = m15_data["high"].astype(float)
        low = m15_data["low"].astype(float)
        close = m15_data["close"].astype(float)

        bar_range = high - low
        atr = bar_range.rolling(regime_params.atr_period, min_periods=regime_params.atr_period).mean()
        avg_atr = atr.rolling(regime_params.range_period, min_periods=regime_params.range_period).mean()

        range_high = close.rolling(regime_params.range_period, min_periods=regime_params.range_period).max()
        range_low = close.rolling(regime_params.range_period, min_periods=regime_params.range_period).min()
        price_range = range_high - range_low

        threshold_range = regime_params.chop_threshold * avg_atr
        chop_ratio = (price_range / threshold_range).fillna(0.0)

        regime_data = pd.DataFrame({"regime": regime_series, "chop_ratio": chop_ratio}, index=m15_data.index)

        logger.info("  Regime calc: %.2fs", time.perf_counter() - t0)

    # Trends
    m15_trends = precalculate_trends(m15_data, TrendParams(min_slope=min_slope, ema_period=ema_period))

    # Merge trends to M5
    trend_series = m15_trends.reset_index().rename(columns={"index": "ts"})
    m5_ts = m5_data.reset_index().rename(columns={"index": "ts"})
    merged = pd.merge_asof(
        m5_ts.sort_values("ts"),
        trend_series.sort_values("ts"),
        on="ts",
        direction="backward",
    )
    m5_data = m5_data.copy()
    m5_data["trend"] = merged["trend"].values

    # Merge regime to M5 (if enabled)
    if regime_filter and regime_data is not None:
        regime_series_df = regime_data.reset_index().rename(columns={"index": "ts"})
        merged_regime = pd.merge_asof(
            m5_ts.sort_values("ts"),
            regime_series_df.sort_values("ts"),
            on="ts",
            direction="backward",
        )
        m5_data["regime"] = merged_regime["regime"].values
        m5_data["chop_ratio"] = merged_regime["chop_ratio"].values

    # Distributions
    trend_counts = m5_data["trend"].value_counts()
    bull_bars = int(trend_counts.get(Trend.BULL, 0))
    bear_bars = int(trend_counts.get(Trend.BEAR, 0))
    none_bars = int(m5_data["trend"].isna().sum())

    print("📈 TREND DISTRIBUTION:")
    print(f"  Bull  : {bull_bars:5d} bars ({bull_bars / len(m5_data) * 100:5.1f}%)")
    print(f"  Bear  : {bear_bars:5d} bars ({bear_bars / len(m5_data) * 100:5.1f}%)")
    print(f"  Range : {none_bars:5d} bars ({none_bars / len(m5_data) * 100:5.1f}%)")

    if regime_filter:
        regime_counts = m5_data["regime"].value_counts()
        trending_bars = int(regime_counts.get(MarketRegime.TRENDING, 0))
        choppy_bars = int(regime_counts.get(MarketRegime.CHOPPY, 0))
        unknown_bars = int(regime_counts.get(MarketRegime.UNKNOWN, 0))
        print("\n🔎 REGIME DISTRIBUTION:")
        print(f"  Trending : {trending_bars:5d} bars ({trending_bars / len(m5_data) * 100:5.1f}%)")
        print(f"  Choppy   : {choppy_bars:5d} bars ({choppy_bars / len(m5_data) * 100:5.1f}%)")
        print(f"  Unknown  : {unknown_bars:5d} bars ({unknown_bars / len(m5_data) * 100:5.1f}%)")
    print()

    # Strategy params (unchanged)
    strat_params = H2L2Params(
        pullback_bars=pullback_bars,
        signal_close_frac=signal_close_frac,
        min_risk_price_units=min_risk_price_units,
        stop_buffer=stop_buffer,
        cooldown_bars=cooldown_bars,
    )

    planned_trades: list[PlannedTrade] = []

    logger.info("→ Planning trades (segment-based with regime filter)...")
    t_plan0 = time.perf_counter()

    total_bars = len(m5_data)

    segments = []
    current_trend = None
    current_regime = None
    segment_start = 50

    for i in range(50, total_bars):
        trend_val = m5_data.iloc[i]["trend"]
        side = _trend_to_side(trend_val) if not pd.isna(trend_val) else None

        regime_val = None
        if regime_filter:
            regime_val = m5_data.iloc[i].get("regime", MarketRegime.UNKNOWN)

        should_break = (side != current_trend)
        if regime_filter:
            should_break = should_break or (regime_val != current_regime)

        if should_break:
            if current_trend is not None and segment_start < i:
                segments.append((segment_start, i, current_trend, current_regime))
            current_trend = side
            current_regime = regime_val
            segment_start = i

    if current_trend is not None and segment_start < total_bars:
        segments.append((segment_start, total_bars, current_trend, current_regime))

    skipped_choppy = 0
    processed_segments = 0

    for start_idx, end_idx, trend_side, regime_val in segments:
        if regime_filter and regime_val == MarketRegime.CHOPPY:
            skipped_choppy += 1
            continue
        if trend_side is None:
            continue

        processed_segments += 1
        lookback_start = max(0, start_idx - 10)
        segment_data = m5_data.iloc[lookback_start:end_idx]

        trades = plan_h2l2_trades(segment_data, trend_side, spec, strat_params)

        segment_start_ts = m5_data.index[start_idx]
        for t in trades:
            if t.execute_ts >= segment_start_ts:
                planned_trades.append(t)

    planning_time = time.perf_counter() - t_plan0

    if regime_filter:
        logger.info(
            "  Planning: %.2fs, %d segments (%d choppy skipped, %d processed) → %d candidates",
            planning_time, len(segments), skipped_choppy, processed_segments, len(planned_trades)
        )
    else:
        logger.info("  Planning: %.2fs, %d segments → %d candidates", planning_time, len(segments), len(planned_trades))

    # Guardrails + selection (unchanged)
    logger.info("→ Applying guardrails (session + daily limit)...")
    g_session = Guardrails(session_tz=NY_TZ, day_tz=NY_TZ, session_start="09:30", session_end="15:00", max_trades_per_day=10_000)
    in_session, rejected1 = apply_guardrails(planned_trades, g_session)

    selected, sel_stats = select_top_per_ny_day(in_session, max_trades_day=max_trades_day, tick_size=float(spec.tick_size))

    g_all = Guardrails(session_tz=NY_TZ, day_tz=NY_TZ, session_start="09:30", session_end="15:00", max_trades_per_day=max_trades_day)
    final_trades, rejected2 = apply_guardrails(selected, g_all)

    # Ensure time order for time-based charts (does not change which trades exist)
    final_trades = sorted(final_trades, key=lambda t: pd.to_datetime(t.execute_ts))

    print("🔎 TRADE PIPELINE:")
    print(f"  Candidates    : {len(planned_trades):4d}")
    if regime_filter:
        print(f"  Choppy skipped: {skipped_choppy:4d} segments")
    print(f"  In session    : {len(in_session):4d} (rejected: {len(rejected1)})")
    print(f"  After select  : {len(selected):4d} (days: {len(sel_stats)})")
    print(f"  Final         : {len(final_trades):4d}")
    print()

    # Simulate (unchanged outcome), but capture real exit timestamp
    results_r: list[float] = []
    exit_ts_list: list[pd.Timestamp] = []

    for t in final_trades:
        raw_r, exit_ts = _simulate_trade_outcome(m5_data, t)
        net_r = _apply_costs(raw_r, costs_per_trade_r)
        results_r.append(net_r)
        exit_ts_list.append(pd.to_datetime(exit_ts))

    if not results_r:
        logger.info("⚠️  No trades executed.")
        client.shutdown()
        return {"error": "No trades"}

    # Trade-level series on entry timestamp (for equity/drawdown/winrate)
    trade_ts = pd.to_datetime([t.execute_ts for t in final_trades])
    res = pd.Series(results_r, index=trade_ts, dtype="float64").sort_index()

    equity_curve = res.cumsum()
    running_max = equity_curve.cummax()
    drawdown = equity_curve - running_max
    max_dd_r = float(drawdown.min())

    # Drawdown duration in trades
    dd_bars = 0
    max_dd_bars = 0
    for dd_val in drawdown.values:
        if dd_val < 0:
            dd_bars += 1
            max_dd_bars = max(max_dd_bars, dd_bars)
        else:
            dd_bars = 0

    wins = float(res[res > 0].sum())
    losses = float(-res[res < 0].sum())
    profit_factor = float(wins / losses) if losses > 0 else float("inf")

    mean_r = float(res.mean())
    std_r = float(res.std(ddof=1)) if len(res) > 1 else 0.0
    sharpe_trade = float(mean_r / std_r) if std_r > 0 else 0.0

    recovery_factor = float(equity_curve.iloc[-1] / abs(max_dd_r)) if max_dd_r < 0 else float("inf")
    annual_r = float(equity_curve.iloc[-1]) * (252.0 / float(days))
    mar_ratio = annual_r / abs(max_dd_r) if max_dd_r < 0 else float("inf")

    winrate = float((res > 0).sum() / len(res)) if len(res) else 0.0

    avg_win_r = float(res[res > 0].mean()) if (res > 0).any() else 0.0
    avg_loss_r = float(res[res < 0].mean()) if (res < 0).any() else 0.0  # negative
    payoff_ratio = (avg_win_r / abs(avg_loss_r)) if avg_loss_r < 0 else float("inf")
    max_consec_losses = _max_consecutive_losses(list(res.values))

    # Side breakdown (keep consistent with final_trades order)
    long_results = [r for t, r in zip(final_trades, results_r) if t.side == Side.LONG]
    short_results = [r for t, r in zip(final_trades, results_r) if t.side == Side.SHORT]
    long_wr = (sum(1 for r in long_results if r > 0) / len(long_results)) if long_results else 0.0
    short_wr = (sum(1 for r in short_results if r > 0) / len(short_results)) if short_results else 0.0

    # Period bounds
    period_start = m5_data.index[0] if len(m5_data) else None
    period_end = m5_data.index[-1] if len(m5_data) else None

    # Build trades df with real exits
    trades_df = _build_trades_dataframe(
        final_trades,
        results_r,
        exit_ts_list,
        m5_data=m5_data,
        ny_tz=NY_TZ,
    )

    # Daily PnL (R/day) indexed by NY date
    daily_pnl_r = _daily_series_from_trades(trades_df, start_dt=period_start, end_dt=period_end, ny_tz=NY_TZ)

    # Manager metrics (R-based)
    mgr = _compute_manager_metrics_r_based(
        trades_df,
        daily_pnl_r=daily_pnl_r,
        trading_days_per_year=trading_days_per_year,
        initial_capital=initial_capital,
    )

    # Output (stdout)
    print("=" * 80)
    print("  📊 RESULTS")
    print("=" * 80)

    if costs_per_trade_r > 0:
        total_cost = costs_per_trade_r * len(res)
        gross_profit = float(equity_curve.iloc[-1]) + total_cost
        print("\n💸 COSTS IMPACT:")
        print(f"  Cost per trade : {costs_per_trade_r:.4f}R")
        print(f"  Total cost     : {total_cost:.2f}R over {len(res)} trades")
        print(f"  Gross profit   : {gross_profit:+.2f}R (before costs)")
        print(f"  Net profit     : {float(equity_curve.iloc[-1]):+.2f}R (after costs)")
        if gross_profit != 0:
            print(f"  Impact         : {-total_cost:.2f}R ({-total_cost / gross_profit * 100:.1f}% reduction)")
        else:
            print(f"  Impact         : {-total_cost:.2f}R")

    print("\n💰 PERFORMANCE:")
    print(f"  Trades        : {len(res):4d}")
    print(f"  Net R         : {float(equity_curve.iloc[-1]):+7.2f}R")
    print(f"  Avg R/trade   : {mean_r:+7.4f}R")
    print(f"  Trade Sharpe  : {sharpe_trade:7.3f}  (trade-level, legacy)")
    print(f"  Profit Factor : {profit_factor:7.2f}")

    print("\n🎯 BROOKS METRICS:")
    print(f"  Expectancy     : {mean_r:+7.4f}R per trade")
    print(f"  Recovery Factor: {recovery_factor:7.2f} (Net/MaxDD)")
    print(f"  MAR Ratio      : {mar_ratio:7.2f} (Annual/MaxDD)")
    print(f"  Annual R est.  : {annual_r:+7.2f}R ({days} days → 252 days)")

    if mgr:
        print("\n📌 MANAGER METRICS (R/day):")
        print(f"  Daily Sharpe (R/day)   : {mgr.get('daily_sharpe_r', 0.0):7.3f}")
        print(f"  Daily Sortino (R/day)  : {mgr.get('daily_sortino_r', 0.0):7.3f}")
        print(f"  Max DD (daily, R)      : {mgr.get('max_dd_r_daily', 0.0):7.2f}R")
        print(f"  Max DD (% of initial)  : {mgr.get('max_dd_pct_initial', 0.0):7.3f}%")
        print(f"  VaR 95% (R/day)        : {mgr.get('var_95_r', 0.0):7.3f}R")
        print(f"  CVaR 95% (R/day)       : {mgr.get('cvar_95_r', 0.0):7.3f}R")
        print(f"  Best / Worst day (R)   : {mgr.get('best_day_r', 0.0):+7.2f} / {mgr.get('worst_day_r', 0.0):+7.2f}")
        print(f"  % positive days        : {mgr.get('pct_pos_days', 0.0):7.2f}%")
        print(f"  Max underwater (days)  : {mgr.get('max_underwater_days', 0)}")
        print(f"  Trades/day active      : {mgr.get('trades_per_active_day', 0.0):.2f}")
        print(f"  Trades/day calendar    : {mgr.get('trades_per_calendar_day', 0.0):.2f}")
        print(f"  Skew / Kurtosis (R/day): {mgr.get('skew_r', 0.0):+.3f} / {mgr.get('kurtosis_r', 0.0):+.3f}")

    print("\n📈 WIN/LOSS:")
    print(f"  Winrate       : {winrate * 100:6.2f}%")
    print(f"  Winners       : {int((res > 0).sum()):4d} ({int((res > 0).sum()) / len(res) * 100:5.1f}%)")
    print(f"  Losers        : {int((res < 0).sum()):4d} ({int((res < 0).sum()) / len(res) * 100:5.1f}%)")
    print(f"  Breakeven     : {int((res == 0).sum()):4d}")

    print("\n🧪 TRADE QUALITY:")
    print(f"  Avg Win (R)   : {avg_win_r:+7.4f}")
    print(f"  Avg Loss (R)  : {avg_loss_r:+7.4f}")
    print(f"  Payoff Ratio  : {payoff_ratio:7.3f}")
    print(f"  Max consec Ls : {max_consec_losses:4d}")

    print("\n📉 DRAWDOWN (trade-level):")
    print(f"  Max DD        : {max_dd_r:7.2f}R")
    print(f"  Max DD bars   : {max_dd_bars:4d} trades")

    print("\n⚖️ SIDE BREAKDOWN:")
    print(f"  Long trades   : {len(long_results):4d} (WR: {long_wr * 100:5.1f}%)")
    print(f"  Short trades  : {len(short_results):4d} (WR: {short_wr * 100:5.1f}%)")

    print("\n⏱️ PERFORMANCE:")
    print(f"  Planning      : {planning_time:6.2f}s")
    print(f"  Bars/second   : {total_bars / planning_time:8.1f}")

    print("\n" + "=" * 80 + "\n")

    # Price series (M15 close for chart)
    price_series = m15_data["close"] if "close" in m15_data.columns else None

    # Unique run id
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{symbol}_{days}d_{timestamp}"

    command_args = (
        f"--days {days}"
        + (" --regime-filter" if regime_filter else "")
        + (f" --chop-threshold {chop_threshold}" if regime_filter else "")
        + f" --stop-buffer {stop_buffer}"
        + f" --cooldown {cooldown_bars}"
        + f" --costs {costs_per_trade_r}"
    )

    # Stats dict for PNG
    stats: Dict[str, Any] = {
        # core trade-level
        "trades": int(len(res)),
        "net_r": float(equity_curve.iloc[-1]),
        "avg_r": float(mean_r),
        "winrate_pct": float(winrate * 100.0),
        "profit_factor": float(profit_factor),
        "max_dd_r_trade": float(max_dd_r),
        "trade_sharpe": float(sharpe_trade),
        "recovery_factor": float(recovery_factor),
        "mar_ratio": float(mar_ratio),
        "annual_r": float(annual_r),
        # costs
        "costs_per_trade_r": float(costs_per_trade_r),
        "total_cost_r": float(costs_per_trade_r * len(res)),
        # trade quality
        "avg_win_r": float(avg_win_r),
        "avg_loss_r": float(avg_loss_r),
        "payoff_ratio": float(payoff_ratio),
        "max_consec_losses": int(max_consec_losses),
        # side
        "long_trades": int(len(long_results)),
        "short_trades": int(len(short_results)),
        "long_wr_pct": float(long_wr * 100.0),
        "short_wr_pct": float(short_wr * 100.0),
        # regime
        "regime_filter": bool(regime_filter),
        "choppy_segments_skipped": int(skipped_choppy if regime_filter else 0),
    }

    # add manager metrics (R/day)
    stats.update(mgr)

    # Add long/short PF + net R
    def _pf_and_net(values: List[float]) -> tuple[float, float]:
        if not values:
            return 0.0, 0.0
        s = pd.Series(values, dtype="float64")
        w = float(s[s > 0].sum())
        l = float(-s[s < 0].sum())
        pf = (w / l) if l > 0 else float("inf")
        return pf, float(s.sum())

    pf_long, net_long = _pf_and_net(long_results)
    pf_short, net_short = _pf_and_net(short_results)
    stats["pf_long"] = pf_long
    stats["net_r_long"] = net_long
    stats["pf_short"] = pf_short
    stats["net_r_short"] = net_short

    # Regime breakdown (if present)
    if "regime_at_entry" in trades_df.columns and trades_df["regime_at_entry"].notna().any():
        for reg in sorted(trades_df["regime_at_entry"].dropna().unique().tolist()):
            rdf = trades_df[trades_df["regime_at_entry"] == reg]
            if rdf.empty:
                continue
            stats[f"trades_reg_{reg}"] = int(len(rdf))
            stats[f"net_r_reg_{reg}"] = float(rdf["net_r"].sum())

    # Generate report
    generate_performance_report(
        results_r=res,                 # time-indexed Series (entry_time)
        equity_curve=equity_curve,     # time-indexed Series
        drawdown=drawdown,             # time-indexed Series
        symbol=symbol,
        days=days,
        run_id=run_id,
        command=command_args,
        period_start=period_start,
        period_end=period_end,
        stats=stats,
        price_series=price_series,
        daily_pnl=daily_pnl_r,         # NY date-indexed series
    )

    client.shutdown()

    return {
        "days": days,
        "trades": int(len(res)),
        "net_r": float(equity_curve.iloc[-1]),
        "winrate": float(winrate),
        "profit_factor": float(profit_factor),
        "max_dd_r_trade": float(max_dd_r),
        "avg_r": float(mean_r),
        "trade_sharpe": float(sharpe_trade),
        **mgr,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="US500.cash")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--max-trades-day", type=int, default=2)

    # Strategy params
    parser.add_argument("--min-slope", type=float, default=0.15)
    parser.add_argument("--ema-period", type=int, default=20)
    parser.add_argument("--pullback-bars", type=int, default=3)
    parser.add_argument("--signal-close-frac", type=float, default=0.30)
    parser.add_argument("--stop-buffer", type=float, default=2.0)
    parser.add_argument("--min-risk", type=float, default=2.0)
    parser.add_argument("--cooldown", type=int, default=10)

    # Regime filter
    parser.add_argument("--regime-filter", action="store_true", help="Enable regime filter (skip choppy markets)")
    parser.add_argument("--chop-threshold", type=float, default=2.5, help="Chop threshold (higher = stricter)")

    # Costs
    parser.add_argument("--costs", type=float, default=0.0, help="Trading costs per trade in R (e.g., 0.04)")

    # Metrics (only used for labeling % of initial in R-based reporting)
    parser.add_argument("--initial-capital", type=float, default=10000.0)
    parser.add_argument("--trading-days-year", type=int, default=252)

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
        regime_filter=args.regime_filter,
        chop_threshold=args.chop_threshold,
        costs_per_trade_r=args.costs,
        initial_capital=args.initial_capital,
        trading_days_per_year=args.trading_days_year,
    )

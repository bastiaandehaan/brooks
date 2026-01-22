# Script: runner.py
# Module: backtest.runner
# Location: backtest/runner.py
"""
BROOKS BACKTEST RUNNER (US500.cash)

Audit controls (optional):
- --strict-m15-close: shift M15-derived features (trend/regime) to become available at M15 bar close (ts + 15min)
  before aligning onto M5. This removes a common MTF lookahead leak when MT5 timestamps represent bar open.
- --strict-entry-bar: do NOT allow TP/SL hits on the entry bar (simulation starts from the first M5 bar strictly AFTER
  execute_ts). This removes intrabar lookahead in bar-based backtests.

This file is designed to be a drop-in replacement that matches your existing CLI flags and your current framework API.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Iterable, Tuple

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

# Repo root
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)

from backtest.visualiser import generate_performance_report
from execution.guardrails import Guardrails, apply_guardrails
from execution.selection import select_top_per_ny_day
from strategies.context import Trend, TrendParams, infer_trend_m15_series
from strategies.h2l2 import H2L2Params, PlannedTrade, Side, plan_h2l2_trades
from strategies.regime import MarketRegime, RegimeParams, detect_regime_series
from utils.mt5_client import Mt5Client
from utils.mt5_data import RatesRequest, fetch_rates

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("Backtest")

NY_TZ = "America/New_York"


# ----------------------------
# Helpers: robust field access
# ----------------------------
def _get_trade_entry(t: PlannedTrade) -> float:
    for k in ("entry", "entry_price"):
        if hasattr(t, k):
            return float(getattr(t, k))
    raise AttributeError("PlannedTrade missing entry field (entry/entry_price).")


def _get_trade_stop(t: PlannedTrade) -> float:
    for k in ("stop", "stop_price"):
        if hasattr(t, k):
            return float(getattr(t, k))
    raise AttributeError("PlannedTrade missing stop field (stop/stop_price).")


def _get_trade_target(t: PlannedTrade) -> float:
    for k in ("tp", "target", "target_price"):
        if hasattr(t, k):
            return float(getattr(t, k))
    raise AttributeError("PlannedTrade missing target field (tp/target/target_price).")


def _get_trade_ts(t: PlannedTrade) -> pd.Timestamp:
    if hasattr(t, "execute_ts"):
        return pd.to_datetime(getattr(t, "execute_ts"))
    raise AttributeError("PlannedTrade missing execute_ts.")


def _normalize_ohlc(df: pd.DataFrame, *, name: str) -> pd.DataFrame:
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name}: missing columns: {sorted(missing)}")
    out = df.sort_index()
    out = out.loc[~out.index.duplicated(keep="first")]
    return out


def _trend_to_side(trend: Any) -> Side | None:
    if trend == Trend.BULL:
        return Side.LONG
    if trend == Trend.BEAR:
        return Side.SHORT
    return None


def _safe_unpack_regime(output: Any) -> Tuple[pd.Series, pd.Series | None]:
    """
    Your detect_regime_series may return:
      - Series (regime)
      - tuple(regime_series, chop_ratio_series)
    We normalize into (regime_series, chop_ratio_series_or_None).
    """
    if isinstance(output, tuple) and len(output) >= 2:
        reg = output[0]
        chop = output[1]
        return pd.Series(reg, index=getattr(reg, "index", None)), pd.Series(
            chop, index=getattr(chop, "index", None)
        )
    # Series-only
    reg = pd.Series(output)
    return reg, None


def _merge_m15_features_to_m5(
    m5: pd.DataFrame,
    m15_features: pd.DataFrame,
    *,
    strict_m15_close: bool,
    m15_bar_minutes: int = 15,
) -> pd.DataFrame:
    """
    Align M15 features onto M5 bars using merge_asof(backward).
    If strict_m15_close=True, shift M15 feature timestamps by +15 minutes to represent availability at bar close.
    """
    m5_ts = m5.reset_index().rename(columns={"index": "ts"})
    m5_ts["ts"] = pd.to_datetime(m5_ts["ts"])

    m15_ts = m15_features.reset_index().rename(columns={"index": "ts"})
    m15_ts["ts"] = pd.to_datetime(m15_ts["ts"])

    if strict_m15_close:
        m15_ts["ts_available"] = m15_ts["ts"] + pd.Timedelta(minutes=m15_bar_minutes)
        right_on = "ts_available"
        m15_sort_col = "ts_available"
    else:
        right_on = "ts"
        m15_sort_col = "ts"

    merged = pd.merge_asof(
        m5_ts.sort_values("ts"),
        m15_ts.sort_values(m15_sort_col),
        left_on="ts",
        right_on=right_on,
        direction="backward",
    )

    merged = merged.set_index("ts")
    # Restore original M5 columns + appended feature columns
    for col in m5.columns:
        if col not in merged.columns:
            merged[col] = m5[col].values
    merged = merged[m5.columns.tolist() + [c for c in merged.columns if c not in m5.columns]]
    return merged


def _slice_future_bars(
    m5_data: pd.DataFrame, exec_ts: pd.Timestamp, *, strict_entry_bar: bool
) -> pd.DataFrame:
    """
    strict_entry_bar=True -> start strictly AFTER exec_ts (exclude entry bar).
    strict_entry_bar=False -> include entry bar (legacy).
    """
    if m5_data.empty:
        return m5_data
    idx = m5_data.index
    ts = pd.to_datetime(exec_ts)
    pos = idx.searchsorted(ts, side="right" if strict_entry_bar else "left")
    if pos >= len(m5_data):
        return m5_data.iloc[0:0]
    return m5_data.iloc[pos:]


def _simulate_trade_outcome(
    m5_data: pd.DataFrame,
    t: PlannedTrade,
    *,
    strict_entry_bar: bool,
) -> tuple[float, pd.Timestamp]:
    """
    Worst-case both-hit policy.
    Returns (raw_r, exit_ts) BEFORE costs.

    Uses fixed payoff convention consistent with your outputs:
      win: +2R, loss: -1R, no hit by end: 0R.
    """
    exec_ts = _get_trade_ts(t)
    entry = _get_trade_entry(t)
    stop = _get_trade_stop(t)
    target = _get_trade_target(t)

    future = _slice_future_bars(m5_data, exec_ts, strict_entry_bar=strict_entry_bar)
    if future.empty:
        return 0.0, exec_ts

    last_ts = pd.to_datetime(future.index[-1])

    for ts, row in future.iterrows():
        high = float(row["high"])
        low = float(row["low"])

        if t.side == Side.LONG:
            stop_hit = low <= stop
            target_hit = high >= target
            if stop_hit and target_hit:
                return -1.0, pd.to_datetime(ts)  # worst-case
            if stop_hit:
                return -1.0, pd.to_datetime(ts)
            if target_hit:
                return 2.0, pd.to_datetime(ts)

        else:  # SHORT
            stop_hit = high >= stop
            target_hit = low <= target
            if stop_hit and target_hit:
                return -1.0, pd.to_datetime(ts)  # worst-case
            if stop_hit:
                return -1.0, pd.to_datetime(ts)
            if target_hit:
                return 2.0, pd.to_datetime(ts)

    return 0.0, last_ts


def _apply_costs(raw_r: float, costs_per_trade_r: float) -> float:
    return float(raw_r) - float(costs_per_trade_r)


def _ny_day_from_ts(ts: pd.Timestamp) -> pd.Timestamp:
    return (
        pd.to_datetime(ts)
        .tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT")
        .tz_convert(NY_TZ)
        .normalize()
        .tz_localize(None)
    )


def _compute_daily_metrics_rday(
    daily_pnl_r: pd.Series,
    *,
    initial_capital: float,
    trading_days_per_year: int,
) -> dict[str, Any]:
    s = daily_pnl_r.astype("float64")
    if s.empty:
        return {}

    mean = float(s.mean())
    std = float(s.std(ddof=1)) if len(s) > 1 else 0.0
    daily_sharpe = (mean / std) * np.sqrt(trading_days_per_year) if std > 0 else 0.0

    downside = s[s < 0]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    daily_sortino = (
        (mean / downside_std) * np.sqrt(trading_days_per_year) if downside_std > 0 else 0.0
    )

    eq = s.cumsum()
    dd = eq - eq.cummax()
    max_dd = float(dd.min()) if len(dd) else 0.0

    # Underwater duration
    underwater = dd < 0
    max_underwater = 0
    cur = 0
    for v in underwater.values:
        if v:
            cur += 1
            max_underwater = max(max_underwater, cur)
        else:
            cur = 0

    # VaR / CVaR 95%
    var95 = float(np.quantile(s.values, 0.05)) if len(s) else 0.0
    tail = s[s <= var95]
    cvar95 = float(tail.mean()) if len(tail) else var95

    best = float(s.max())
    worst = float(s.min())
    pct_pos = float((s > 0).mean() * 100.0)

    return {
        "daily_sharpe_rday": float(daily_sharpe),
        "daily_sortino_rday": float(daily_sortino),
        "max_dd_r_day": float(max_dd),
        "max_dd_pct_initial": float((max_dd / initial_capital) * 100.0) if initial_capital else 0.0,
        "var95_rday": float(var95),
        "cvar95_rday": float(cvar95),
        "best_day_r": float(best),
        "worst_day_r": float(worst),
        "pct_positive_days": float(pct_pos),
        "max_underwater_days": int(max_underwater),
        "skew_rday": float(s.skew()) if len(s) > 2 else 0.0,
        "kurtosis_rday": float(s.kurtosis()) if len(s) > 3 else 0.0,
        "downside_std_rday": float(downside_std),
        "downside_days": int((s < 0).sum()),
    }


def run_backtest(
    symbol: str,
    days: int,
    *,
    max_trades_day: int,
    min_slope: float,
    ema_period: int,
    pullback_bars: int,
    signal_close_frac: float,
    stop_buffer: float,
    min_risk_price_units: float,
    cooldown_bars: int,
    regime_filter: bool,
    chop_threshold: float,
    costs_per_trade_r: float,
    initial_capital: float,
    trading_days_per_year: int,
    strict_m15_close: bool,
    strict_entry_bar: bool,
) -> dict[str, Any]:
    print()
    print("=" * 80)
    print(f"  BROOKS BACKTEST: {symbol} ({days} days)")
    if regime_filter:
        print(f"  🔎 REGIME FILTER: ENABLED (chop_threshold={chop_threshold})")
    else:
        print("  🔎 REGIME FILTER: DISABLED")
    print(
        f"  💸 COSTS: {costs_per_trade_r:.4f}R per trade"
        if costs_per_trade_r > 0
        else "  💸 COSTS: disabled"
    )
    if strict_m15_close:
        print("  🧪 AUDIT: strict_m15_close=ON (M15 features available at bar close)")
    if strict_entry_bar:
        print("  🧪 AUDIT: strict_entry_bar=ON (entry bar excluded from TP/SL hits)")
    print("=" * 80)
    print()
    print("📊 STRATEGY PARAMETERS:")
    print(f"  Context    : min_slope={min_slope:.2f}, ema_period={ema_period}")
    print(f"  H2/L2      : pullback={pullback_bars}, close_frac={signal_close_frac:.2f}")
    print(f"  Risk       : stop_buffer={stop_buffer:.1f}, min_risk={min_risk_price_units:.1f}")
    print(f"  Execution  : cooldown={cooldown_bars}bars, max_trades_day={max_trades_day}")
    print()

    print("Initializing MT5 connection...")
    client = Mt5Client()
    client.initialize()
    acc = mt5.account_info()
    term = mt5.terminal_info()
    print(
        f"MT5 connected. Terminal={term.name if term else 'unknown'}, Account={acc.login if acc else 'unknown'}"
    )
    print()

    logger.info("→ Fetching data...")
    req_m15 = RatesRequest(symbol=symbol, timeframe=mt5.TIMEFRAME_M15, count=int(days * 24 * 4 * 4))
    req_m5 = RatesRequest(symbol=symbol, timeframe=mt5.TIMEFRAME_M5, count=int(days * 24 * 12 * 4))

    m15_data = fetch_rates(req_m15)
    m5_data = fetch_rates(req_m5)

    m15_data = _normalize_ohlc(m15_data, name="m15_data")
    m5_data = _normalize_ohlc(m5_data, name="m5_data")

    m15_data.index = pd.to_datetime(m15_data.index)
    m5_data.index = pd.to_datetime(m5_data.index)

    logger.info("  M15: %d bars, M5: %d bars", len(m15_data), len(m5_data))

    spec = client.get_symbol_spec(symbol)

    # ----------------------------
    # Regime (optional) + Trend
    # ----------------------------
    regime_data: pd.DataFrame | None = None
    if regime_filter:
        logger.info("→ Calculating market regime (vectorized)...")
        t0 = time.perf_counter()

        # IMPORTANT: RegimeParams in your codebase does NOT accept regime_filter kwarg.
        params = RegimeParams(chop_threshold=chop_threshold)
        reg_out = detect_regime_series(m15_data, params)
        reg_series, chop_series = _safe_unpack_regime(reg_out)

        # Ensure index alignment to M15 data
        reg_series = pd.Series(reg_series.values, index=m15_data.index)
        if chop_series is not None:
            chop_series = pd.Series(chop_series.values, index=m15_data.index)

        regime_data = pd.DataFrame(
            {
                "regime": reg_series,
                "chop_ratio": chop_series if chop_series is not None else np.nan,
            },
            index=m15_data.index,
        )
        logger.info("  Regime calc: %.2fs", time.perf_counter() - t0)

    logger.info("→ Calculating trends (vectorized)...")
    t0 = time.perf_counter()
    trend_series = infer_trend_m15_series(
        m15_data, TrendParams(min_slope=min_slope, ema_period=ema_period)
    )
    m15_trends = pd.DataFrame({"trend": trend_series}, index=m15_data.index)
    logger.info("  Trend calc: %.2fs", time.perf_counter() - t0)

    # Build M15 feature frame for merge
    m15_features = m15_trends.copy()
    if regime_filter and regime_data is not None:
        m15_features["regime"] = regime_data["regime"]
        m15_features["chop_ratio"] = regime_data["chop_ratio"]

    # Align onto M5
    m5_data = m5_data.copy()
    m5_merged = _merge_m15_features_to_m5(m5_data, m15_features, strict_m15_close=strict_m15_close)
    m5_data["trend"] = m5_merged["trend"].values
    if regime_filter and "regime" in m5_merged.columns:
        m5_data["regime"] = m5_merged["regime"].values
        if "chop_ratio" in m5_merged.columns:
            m5_data["chop_ratio"] = m5_merged["chop_ratio"].values

    # ----------------------------
    # Distributions
    # ----------------------------
    trend_counts = m5_data["trend"].value_counts()
    bull_bars = int(trend_counts.get(Trend.BULL, 0))
    bear_bars = int(trend_counts.get(Trend.BEAR, 0))
    range_bars = (
        int(trend_counts.get(Trend.RANGE, 0))
        if hasattr(Trend, "RANGE")
        else int(m5_data["trend"].isna().sum())
    )

    print("📈 TREND DISTRIBUTION:")
    total = max(len(m5_data), 1)
    print(f"  Bull  : {bull_bars:5d} bars ({bull_bars / total * 100:5.1f}%)")
    print(f"  Bear  : {bear_bars:5d} bars ({bear_bars / total * 100:5.1f}%)")
    print(f"  Range : {range_bars:5d} bars ({range_bars / total * 100:5.1f}%)")

    if regime_filter and "regime" in m5_data.columns:
        reg_counts = m5_data["regime"].value_counts()
        trending_bars = int(reg_counts.get(MarketRegime.TRENDING, 0))
        choppy_bars = int(reg_counts.get(MarketRegime.CHOPPY, 0))
        unknown_bars = int(reg_counts.get(MarketRegime.UNKNOWN, 0))
        print()
        print("🔎 REGIME DISTRIBUTION:")
        print(f"  Trending : {trending_bars:5d} bars ({trending_bars / total * 100:5.1f}%)")
        print(f"  Choppy   : {choppy_bars:5d} bars ({choppy_bars / total * 100:5.1f}%)")
        print(f"  Unknown  : {unknown_bars:5d} bars ({unknown_bars / total * 100:5.1f}%)")
    print()

    # ----------------------------
    # Strategy parameters
    # ----------------------------
    strat_params = H2L2Params(
        pullback_bars=pullback_bars,
        signal_close_frac=signal_close_frac,
        min_risk_price_units=min_risk_price_units,
        stop_buffer=stop_buffer,
        cooldown_bars=cooldown_bars,
    )

    # ----------------------------
    # Planning
    # ----------------------------
    logger.info("→ Planning trades (segment-based with regime filter)...")
    t_plan0 = time.perf_counter()

    total_bars = len(m5_data)
    segments: list[tuple[int, int, Side | None, MarketRegime | None]] = []
    current_trend: Side | None = None
    current_regime: MarketRegime | None = None
    segment_start = 50

    for idx in range(50, total_bars):
        trend_val = m5_data.iloc[idx]["trend"]
        side = _trend_to_side(trend_val) if not pd.isna(trend_val) else None

        regime_val: MarketRegime | None = None
        if regime_filter and "regime" in m5_data.columns:
            regime_val = m5_data.iloc[idx].get("regime", MarketRegime.UNKNOWN)

        should_break = side != current_trend
        if regime_filter:
            should_break = should_break or (regime_val != current_regime)

        if should_break:
            if current_trend is not None and segment_start < idx:
                segments.append((segment_start, idx, current_trend, current_regime))
            current_trend = side
            current_regime = regime_val
            segment_start = idx

    if current_trend is not None and segment_start < total_bars:
        segments.append((segment_start, total_bars, current_trend, current_regime))

    skipped_choppy = 0
    processed_segments = 0
    planned_trades: list[PlannedTrade] = []

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
        for tr in trades:
            if _get_trade_ts(tr) >= pd.to_datetime(segment_start_ts):
                planned_trades.append(tr)

    planning_time = time.perf_counter() - t_plan0

    if regime_filter:
        logger.info(
            "  Planning: %.2fs, %d segments (%d choppy skipped, %d processed) → %d candidates",
            planning_time,
            len(segments),
            skipped_choppy,
            processed_segments,
            len(planned_trades),
        )
    else:
        logger.info(
            "  Planning: %.2fs, %d segments → %d candidates",
            planning_time,
            len(segments),
            len(planned_trades),
        )

    # ----------------------------
    # Guardrails + selection
    # ----------------------------
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
        tick_size=float(getattr(spec, "tick_size", 1.0)),
    )

    g_all = Guardrails(
        session_tz=NY_TZ,
        day_tz=NY_TZ,
        session_start="09:30",
        session_end="15:00",
        max_trades_per_day=max_trades_day,
    )
    final_trades, rejected2 = apply_guardrails(selected, g_all)
    final_trades = sorted(final_trades, key=lambda tr: _get_trade_ts(tr))

    print("🔎 TRADE PIPELINE:")
    print(f"  Candidates    : {len(planned_trades):4d}")
    if regime_filter:
        print(f"  Choppy skipped: {skipped_choppy:4d} segments")
    print(f"  In session    : {len(in_session):4d} (rejected: {len(rejected1)})")
    print(f"  After select  : {len(selected):4d} (days: {len(sel_stats)})")
    print(f"  Final         : {len(final_trades):4d}")
    print()

    # ----------------------------
    # Simulate outcomes
    # ----------------------------
    results_r: list[float] = []
    exit_ts_list: list[pd.Timestamp] = []

    for tr in final_trades:
        raw_r, exit_ts = _simulate_trade_outcome(m5_data, tr, strict_entry_bar=strict_entry_bar)
        net_r = _apply_costs(raw_r, costs_per_trade_r)
        results_r.append(net_r)
        exit_ts_list.append(pd.to_datetime(exit_ts))

    if not results_r:
        logger.info("No trades executed.")
        client.shutdown()
        return {"error": "No trades"}

    trade_ts = pd.to_datetime([_get_trade_ts(tr) for tr in final_trades])
    res = pd.Series(results_r, index=trade_ts, dtype="float64").sort_index()

    equity_curve = res.cumsum()
    running_max = equity_curve.cummax()
    drawdown = equity_curve - running_max
    max_dd_r = float(drawdown.min())

    wins = float(res[res > 0].sum())
    losses = float(-res[res < 0].sum())
    profit_factor = float(wins / losses) if losses > 0 else float("inf")

    mean_r = float(res.mean())
    std_r = float(res.std(ddof=1)) if len(res) > 1 else 0.0
    sharpe_trade = float(mean_r / std_r) if std_r > 0 else 0.0

    winrate = float((res > 0).sum() / len(res)) if len(res) else 0.0

    # Daily PnL in R/day using NY day buckets (entry timestamps)
    ny_days = pd.Index([_ny_day_from_ts(ts) for ts in res.index], name="ny_day")
    daily_pnl_r = (
        pd.Series(res.values, index=ny_days, dtype="float64").groupby(level=0).sum().sort_index()
    )

    mgr = _compute_daily_metrics_rday(
        daily_pnl_r,
        initial_capital=initial_capital,
        trading_days_per_year=trading_days_per_year,
    )

    recovery_factor = float(res.sum() / abs(max_dd_r)) if max_dd_r < 0 else float("inf")
    annual_r_est = float(res.sum()) * (252.0 / float(days))
    mar_ratio = float(annual_r_est / abs(max_dd_r)) if max_dd_r < 0 else float("inf")

    gross_profit = (
        float((res + costs_per_trade_r).sum()) if costs_per_trade_r > 0 else float(res.sum())
    )
    total_cost = float(costs_per_trade_r) * float(len(res)) if costs_per_trade_r > 0 else 0.0

    print("=" * 80)
    print("  📊 RESULTS")
    print("=" * 80)
    print()
    if costs_per_trade_r > 0:
        print("💸 COSTS IMPACT:")
        print(f"  Cost per trade : {costs_per_trade_r:.4f}R")
        print(f"  Total cost     : {total_cost:.2f}R over {len(res)} trades")
        print(f"  Gross profit   : +{gross_profit:.2f}R (before costs)")
        print(f"  Net profit     : +{res.sum():.2f}R (after costs)")
        impact_pct = (total_cost / gross_profit * 100.0) if gross_profit != 0 else 0.0
        print(f"  Impact         : -{total_cost:.2f}R (-{impact_pct:.1f}% reduction)")
        print()

    print("💰 PERFORMANCE:")
    print(f"  Trades        : {len(res):4d}")
    print(f"  Net R         : {res.sum():+8.2f}R")
    print(f"  Avg R/trade   : {mean_r:+0.4f}R")
    print(f"  Trade Sharpe  : {sharpe_trade:8.3f}  (trade-level, legacy)")
    print(f"  Profit Factor : {profit_factor:8.2f}")
    print()
    print("🎯 BROOKS METRICS:")
    print(f"  Expectancy     : {mean_r:+.4f}R per trade")
    print(f"  Recovery Factor: {recovery_factor:8.2f} (Net/MaxDD)")
    print(f"  MAR Ratio      : {mar_ratio:8.2f} (Annual/MaxDD)")
    print(f"  Annual R est.  : {annual_r_est:+.2f}R ({days} days → 252 days)")
    print()

    if mgr:
        print("📌 MANAGER METRICS (R/day):")
        print(f"  Daily Sharpe (R/day)   : {mgr.get('daily_sharpe_rday', 0.0):7.3f}")
        print(f"  Daily Sortino (R/day)  : {mgr.get('daily_sortino_rday', 0.0):7.3f}")
        print(f"  Max DD (daily, R)      : {mgr.get('max_dd_r_day', 0.0):7.2f}R")
        print(f"  Max DD (% of initial)  : {mgr.get('max_dd_pct_initial', 0.0):7.3f}%")
        print(f"  VaR 95% (R/day)        : {mgr.get('var95_rday', 0.0):7.3f}R")
        print(f"  CVaR 95% (R/day)       : {mgr.get('cvar95_rday', 0.0):7.3f}R")
        print(
            f"  Best / Worst day (R)   : {mgr.get('best_day_r', 0.0):7.2f} / {mgr.get('worst_day_r', 0.0):7.2f}"
        )
        print(f"  % positive days        : {mgr.get('pct_positive_days', 0.0):7.2f}%")
        print(f"  Max underwater (days)  : {mgr.get('max_underwater_days', 0):4d}")
        print(
            f"  Downside std (R/day)   : {mgr.get('downside_std_rday', 0.0):7.3f} (days: {mgr.get('downside_days', 0)})"
        )
        print(
            f"  Skew / Kurtosis (R/day): {mgr.get('skew_rday', 0.0):+0.3f} / {mgr.get('kurtosis_rday', 0.0):+0.3f}"
        )
        print()

    print("📈 WIN/LOSS:")
    print(f"  Winrate       : {winrate * 100:7.2f}%")
    print(f"  Winners       : {(res > 0).sum():4d} ({winrate * 100:5.1f}%)")
    print(f"  Losers        : {(res < 0).sum():4d} ({(1 - winrate) * 100:5.1f}%)")
    print()

    # Visual report
    price_series = m15_data["close"].copy()
    period_start = pd.to_datetime(m15_data.index.min())
    period_end = pd.to_datetime(m15_data.index.max())
    run_id = f"{symbol}_{days}d_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    command_args = " ".join(sys.argv)

    stats: dict[str, Any] = {
        "trades": int(len(res)),
        "net_r": float(res.sum()),
        "winrate": float(winrate),
        "profit_factor": float(profit_factor),
        "trade_sharpe_legacy": float(sharpe_trade),
        "max_dd_r": float(max_dd_r),
        "recovery_factor": float(recovery_factor),
        "mar_ratio": float(mar_ratio),
        "annual_r_est": float(annual_r_est),
        "strict_m15_close": bool(strict_m15_close),
        "strict_entry_bar": bool(strict_entry_bar),
        "costs_per_trade_r": float(costs_per_trade_r),
    }
    stats.update(mgr)

    generate_performance_report(
        results_r=res,
        equity_curve=equity_curve,
        drawdown=drawdown,
        symbol=symbol,
        days=days,
        run_id=run_id,
        command=command_args,
        period_start=period_start,
        period_end=period_end,
        stats=stats,
        price_series=price_series,
        daily_pnl=daily_pnl_r,
    )

    client.shutdown()

    return {
        "days": days,
        "trades": int(len(res)),
        "net_r": float(res.sum()),
        "winrate": float(winrate),
        "profit_factor": float(profit_factor),
        "max_dd_r": float(max_dd_r),
        "avg_r": float(mean_r),
        "trade_sharpe": float(sharpe_trade),
        **mgr,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--symbol", type=str, default="US500.cash")
    parser.add_argument("--days", type=int, default=60)

    # Execution / selection
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
    parser.add_argument("--regime-filter", action="store_true")
    parser.add_argument("--chop-threshold", type=float, default=2.5)

    # Costs
    parser.add_argument(
        "--costs", type=float, default=0.0, help="Trading costs per trade in R (e.g. 0.04)"
    )

    # Metrics
    parser.add_argument("--initial-capital", type=float, default=10000.0)
    parser.add_argument("--trading-days-year", type=int, default=252)

    # Audit switches (optional, default OFF to preserve baseline behavior)
    parser.add_argument(
        "--strict-m15-close",
        action="store_true",
        help="Shift M15 features to bar close before M5 merge",
    )
    parser.add_argument(
        "--strict-entry-bar",
        action="store_true",
        help="Exclude entry bar from TP/SL hit simulation",
    )

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
        strict_m15_close=args.strict_m15_close,
        strict_entry_bar=args.strict_entry_bar,
    )

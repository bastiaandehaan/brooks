# backtest/runner.py
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
from strategies.h2l2 import plan_next_open_trade, H2L2Params, Side, PlannedTrade
from execution.guardrails import Guardrails, apply_guardrails
from backtest.visualiser import generate_performance_report

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("Backtest")


NY_TZ = "America/New_York"


def precalculate_trends(m15_df: pd.DataFrame, params: TrendParams) -> pd.DataFrame:
    """Pre-calc trend series without look-ahead (uses only bars <= i)."""
    logger.info("Trends pre-calculeren...")
    trends = []
    for i in range(len(m15_df)):
        slice_df = m15_df.iloc[: i + 1]
        trend, _ = infer_trend_m15(slice_df, params)
        trends.append(trend)
    m15_df = m15_df.copy()
    m15_df["trend"] = trends
    return m15_df[["trend"]]


def trade_risk_price_units(t: PlannedTrade) -> float:
    """Absolute risk in price units (entry-stop)."""
    return float(abs(t.entry - t.stop))


def score_trade(t: PlannedTrade) -> float:
    """
    MVP score (no extra features available):
    - prefer smaller risk (tighter stop) => higher score
    """
    r = trade_risk_price_units(t)
    if not np.isfinite(r) or r <= 0:
        return -1e18
    return -r  # smaller risk => higher score (less negative)


def select_top_trades_per_day(
    trades: list[PlannedTrade],
    *,
    max_per_day: int,
    day_tz: str = NY_TZ,
) -> list[PlannedTrade]:
    """
    Deterministic selection:
    - group by execute_ts NY day
    - sort by score desc, then execute_ts asc (tie-breaker), then signal_ts asc
    - keep top max_per_day per day
    """
    if not trades:
        return []

    # Ensure deterministic input order doesn't matter
    s_exec = pd.Series([t.execute_ts for t in trades], dtype="datetime64[ns, UTC]")
    days = s_exec.dt.tz_convert(day_tz).dt.date

    grouped: dict[object, list[PlannedTrade]] = {}
    for d, t in zip(days.tolist(), trades):
        grouped.setdefault(d, []).append(t)

    selected: list[PlannedTrade] = []
    for d in sorted(grouped.keys()):
        cand = grouped[d]
        cand_sorted = sorted(
            cand,
            key=lambda t: (
                -score_trade(t),                 # score desc (note: score is negative risk)
                t.execute_ts.value,              # earliest execute
                t.signal_ts.value,               # earliest signal
            ),
        )
        pick = cand_sorted[:max_per_day]
        logger.info(f"Selection {d}: candidates={len(cand)} selected={len(pick)}")
        selected.extend(pick)

    # Keep global chronological order
    selected = sorted(selected, key=lambda t: (t.execute_ts.value, t.signal_ts.value))
    return selected


def _simulate_trade_outcome(m5_data: pd.DataFrame, t: PlannedTrade) -> float:
    """
    Simuleer trade outcome in R.
    Policy:
      - execute bar telt mee (geen .iloc[1:] bias)
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


def run_backtest(symbol: str, days: int) -> None:
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

    if m15_data is None or m5_data is None or m15_data.empty or m5_data.empty:
        logger.error("Data error.")
        client.shutdown()
        return

    trend_data = precalculate_trends(m15_data, TrendParams())

    m5_with_trend = pd.merge_asof(
        m5_data.sort_index(),
        trend_data.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
    )

    strat_params = H2L2Params(
        min_risk_price_units=1.0,
        signal_close_frac=0.30,
        pullback_bars=2,
        cooldown_bars=0,
    )

    raw_trades: list[PlannedTrade] = []
    start_idx = 200
    logger.info(f"Simuleren van {len(m5_with_trend)} bars...")

    for i in range(start_idx, len(m5_with_trend)):
        current_trend = m5_with_trend.iloc[i]["trend"]
        if current_trend not in (Trend.BULL, Trend.BEAR):
            continue

        side = Side.LONG if current_trend == Trend.BULL else Side.SHORT
        m5_slice = m5_with_trend.iloc[i - 50 : i + 1]

        trade = plan_next_open_trade(m5_slice, side, spec, strat_params, timeframe_minutes=5)
        if trade:
            raw_trades.append(trade)

    logger.info(f"Ruwe signalen: {len(raw_trades)}")

    # 1) Session filter first (but don't cap here)
    g_session_only = Guardrails(
        max_trades_per_day=10_000,  # effectively off for this step
        session_start="09:30",
        session_end="15:00",
        day_tz=NY_TZ,
        session_tz=NY_TZ,
    )
    in_session, rejected_session = apply_guardrails(raw_trades, g_session_only)

    logger.info(f"Na session filter: {len(in_session)} (rejected={len(rejected_session)})")

    # 2) Deterministic selection: pick top-2 per NY day
    selected = select_top_trades_per_day(in_session, max_per_day=2, day_tz=NY_TZ)

    # 3) Apply final guardrails (cap still enforced; also catches any edge cases)
    g_final = Guardrails(
        max_trades_per_day=2,
        session_start="09:30",
        session_end="15:00",
        day_tz=NY_TZ,
        session_tz=NY_TZ,
    )
    accepted_trades, rejected_final = apply_guardrails(selected, g_final)

    logger.info(f"Trades na Selection: {len(selected)}")
    logger.info(f"Trades na Guardrails: {len(accepted_trades)} (rejected={len(rejected_final)})")

    if not accepted_trades:
        logger.warning("Geen trades na guardrails.")
        client.shutdown()
        return

    results_r = []
    for t in accepted_trades:
        results_r.append(_simulate_trade_outcome(m5_data, t))

    res_series = pd.Series(results_r)
    equity_curve = res_series.cumsum()
    running_max = equity_curve.cummax()
    drawdown = equity_curve - running_max

    max_dd_depth = float(drawdown.min())
    is_underwater = drawdown < 0
    dd_groups = (is_underwater != is_underwater.shift()).cumsum()
    max_dd_duration = int(is_underwater.groupby(dd_groups).sum().max()) if len(is_underwater) else 0

    sharpe = (res_series.mean() / res_series.std()) * np.sqrt(len(results_r)) if res_series.std() != 0 else 0.0
    neg_sum = abs(sum([r for r in results_r if r < 0]))
    pf = (sum([r for r in results_r if r > 0]) / neg_sum) if neg_sum > 0 else float("inf")
    winrate = (res_series > 0).mean() * 100 if len(res_series) else 0.0

    generate_performance_report(results_r, equity_curve, drawdown, symbol, days)

    print("\n" + "=" * 45)
    print(f" FINAL OPTIMIZED REPORT: {symbol}")
    print("=" * 45)
    print(f"Periode              : {days} dagen")
    print(f"Totaal Trades        : {len(results_r)}")
    print(f"Winrate              : {winrate:.1f}%")
    print("-" * 45)
    print(f"Netto Resultaat      : {equity_curve.iloc[-1]:.2f} R")
    print("Profit Factor        : inf" if pf == float("inf") else f"Profit Factor        : {pf:.2f}")
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

# main.py
"""
Brooks MVP - FIXED versie
Simpele parameters, geen over-engineering
"""
from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd
import MetaTrader5 as mt5

from execution.guardrails import Guardrails, apply_guardrails
from strategies.context import Trend, TrendParams, infer_trend_m15
from strategies.h2l2 import H2L2Params, Side, plan_next_open_trade
from utils.mt5_client import Mt5Client
from utils.mt5_data import fetch_rates, RatesRequest
from execution.risk_manager import RiskManager

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="US500.cash")
    p.add_argument("--m15-bars", type=int, default=300)
    p.add_argument("--m5-bars", type=int, default=500)
    p.add_argument("--timeframe-minutes", type=int, default=5)

    # Trend (SIMPEL - geen excessive filters)
    p.add_argument("--ema", type=int, default=20)

    # Strategy (SIMPEL - Brooks defaults)
    p.add_argument("--pullback-bars", type=int, default=3)
    p.add_argument("--signal-close-frac", type=float, default=0.30)
    p.add_argument("--min-risk-price-units", type=float, default=2.0)
    p.add_argument("--stop-buffer", type=float, default=1.0)

    # Risk
    p.add_argument("--risk-pct", type=float, default=1.0)

    # Guardrails
    p.add_argument("--session-tz", default="America/New_York")
    p.add_argument("--day-tz", default="America/New_York")
    p.add_argument("--session-start", default="09:30")
    p.add_argument("--session-end", default="15:00")
    p.add_argument("--max-trades-day", type=int, default=2)

    return p


def main() -> int:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args()

    logger.info("Starting Brooks MVP (FIXED) for %s...", args.symbol)

    # 1. Connect to MT5
    client = Mt5Client(mt5_module=mt5)
    if not client.initialize():
        logger.error("Failed to connect to MT5.")
        return 1

    # 2. Check symbol
    spec = client.get_symbol_specification(args.symbol)
    if spec is None:
        logger.error("Symbol %s not found.", args.symbol)
        client.shutdown()
        return 1

    # 3. Fetch Data
    logger.info("Fetching data...")
    req_m15 = RatesRequest(args.symbol, mt5.TIMEFRAME_M15, args.m15_bars)
    req_m5 = RatesRequest(args.symbol, mt5.TIMEFRAME_M5, args.m5_bars)

    try:
        m15 = fetch_rates(mt5, req_m15)
        m5 = fetch_rates(mt5, req_m5)
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        client.shutdown()
        return 1

    if m15.empty or m5.empty:
        logger.error("Fetched empty dataframes.")
        client.shutdown()
        return 1

    # 4. Infer Trend (M15) - SIMPEL
    t_params = TrendParams(ema_period=args.ema)
    trend_res, metrics = infer_trend_m15(m15, t_params)

    logger.info(
        "Trend M15: %s (close=%.2f ema=%.2f slope=%.2f)",
        trend_res, metrics.last_close, metrics.last_ema, metrics.ema_slope
    )

    if trend_res not in [Trend.BULL, Trend.BEAR]:
        logger.info("No clear trend. No trade.")
        client.shutdown()
        return 0

    side = Side.LONG if trend_res == Trend.BULL else Side.SHORT

    # 5. Plan Trade (M5) - SIMPEL
    strat_params = H2L2Params(
        pullback_bars=args.pullback_bars,
        signal_close_frac=args.signal_close_frac,
        min_risk_price_units=args.min_risk_price_units,
        stop_buffer=args.stop_buffer,
        cooldown_bars=0,
    )

    planned_trade = plan_next_open_trade(
        m5,
        trend=side,
        spec=spec,
        p=strat_params,
        timeframe_minutes=args.timeframe_minutes,
        now_utc=pd.Timestamp.now(tz="UTC"),
    )

    if planned_trade is None:
        logger.info("Planner: no NEXT_OPEN candidate.")
        client.shutdown()
        return 0

    # 6. Risk Management
    acc_info = mt5.account_info()
    if acc_info is None:
        logger.error("Could not fetch account info.")
        client.shutdown()
        return 1

    rm = RiskManager(risk_per_trade_pct=args.risk_pct)
    lots = rm.calculate_lot_size(
        balance=acc_info.balance,
        spec=spec,
        entry=planned_trade.entry,
        stop=planned_trade.stop
    )

    # 7. Guardrails
    g = Guardrails(
        session_tz=args.session_tz,
        day_tz=args.day_tz,
        session_start=args.session_start,
        session_end=args.session_end,
        max_trades_per_day=args.max_trades_day,
    )

    accepted, rejected = apply_guardrails([planned_trade], g)

    if rejected:
        for _, reason in rejected:
            logger.info("Guardrail reject: %s", reason)

    if not accepted:
        logger.info("Planner: candidate rejected by guardrails.")
        client.shutdown()
        return 0

    pick = accepted[0]

    # TRADE OUTPUT
    print("\n" + "!" * 60)
    print(f" BROOKS TRADE SIGNAL (FIXED): {args.symbol}")
    print(f" RICHTING    : {pick.side}")
    print(f" ENTRY (OPEN): {pick.entry:.2f}")
    print(f" STOP LOSS   : {pick.stop:.2f}")
    print(f" TAKE PROFIT : {pick.tp:.2f}")
    print(f" VOLUME      : {lots} LOTS")
    print(f" RISICO      : {args.risk_pct}% (${acc_info.balance * args.risk_pct / 100:.2f})")
    print(f" REDEN       : {pick.reason}")
    print("!" * 60 + "\n")

    logger.info("LIVE MODE: Order-executie is handmatig. Kopieer bovenstaande data.")

    client.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
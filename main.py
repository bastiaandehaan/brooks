from __future__ import annotations

import argparse
import logging
import sys
from zoneinfo import ZoneInfo

import MetaTrader5 as mt5  # We hebben de module nodig voor constanten
import pandas as pd

from execution.guardrails import Guardrails, apply_guardrails
from strategies.context import Trend, TrendParams, infer_trend_m15
from strategies.h2l2 import H2L2Params, Side, plan_next_open_trade
from utils.mt5_client import Mt5Client
# Let op: RatesRequest import toegevoegd
from utils.mt5_data import fetch_rates, RatesRequest
from utils.symbol_spec import SymbolSpec

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
    p.add_argument("--ema", type=int, default=20)
    p.add_argument("--min-slope", type=float, default=0.20)
    p.add_argument("--min-risk-price-units", type=float, default=1.0)
    p.add_argument("--signal-close-frac", type=float, default=0.25)
    p.add_argument("--cooldown-bars", type=int, default=0)
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

    logger.info("Starting Brooks MVP for %s...", args.symbol)

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

    # 3. Fetch Data (AANGEPAST voor nieuwe mt5_data.py)
    logger.info("Fetching data...")

    # Maak Requests aan met MT5 constanten (geen strings meer)
    req_m15 = RatesRequest(
        symbol=args.symbol,
        timeframe=mt5.TIMEFRAME_M15,
        count=args.m15_bars
    )

    req_m5 = RatesRequest(
        symbol=args.symbol,
        timeframe=mt5.TIMEFRAME_M5,
        count=args.m5_bars
    )

    try:
        # We geven 'mt5' (de module) mee, niet 'client'
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

    # 4. Infer Trend (M15)
    t_params = TrendParams(ema_period=args.ema, min_slope=args.min_slope)
    trend_res, metrics = infer_trend_m15(m15, t_params)
    logger.info("Trend M15: %s (Metrics: slope=%.2f, dist=%.2f)", trend_res, metrics.ema_slope, metrics.close_ema_dist)

    if trend_res not in [Trend.BULL, Trend.BEAR]:
        logger.info("No clear trend (RANGE or NONE). No trade.")
        client.shutdown()
        return 0

    side = Side.LONG if trend_res == Trend.BULL else Side.SHORT

    # 5. Plan Trade (M5)
    strat_params = H2L2Params(
        min_risk_price_units=args.min_risk_price_units,
        signal_close_frac=args.signal_close_frac,
        cooldown_bars=args.cooldown_bars,
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

    # 6. Guardrails
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
    logger.info(
        "CANDIDATE APPROVED: side=%s signal=%s exec=%s entry=%.2f stop=%.2f tp=%.2f",
        pick.side,
        pick.signal_ts,
        pick.execute_ts,
        pick.entry,
        pick.stop,
        pick.tp
    )

    client.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
from __future__ import annotations

import argparse
import logging

import MetaTrader5 as mt5

from execution.guardrails import Guardrails, apply_guardrails
from strategies.context import Trend, TrendParams, infer_trend_m15
from strategies.h2l2 import Side, H2L2Params, plan_h2l2_trades
from utils.mt5_client import Mt5Client, Mt5ConnectionParams
from utils.mt5_data import RatesRequest, fetch_rates
from utils.symbol_spec import SymbolSpec

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger("brooks")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="US500.cash")
    p.add_argument("--m5-bars", type=int, default=500)
    p.add_argument("--m15-bars", type=int, default=300)
    p.add_argument("--max-trades-day", type=int, default=2)
    args = p.parse_args()

    c = Mt5Client(mt5, Mt5ConnectionParams())
    c.initialize()
    try:
        c.ensure_selected(args.symbol)
        info = c.symbol_info(args.symbol)
        spec = SymbolSpec.from_symbol_info(info)

        m15 = fetch_rates(mt5, RatesRequest(symbol=args.symbol, timeframe=mt5.TIMEFRAME_M15, count=args.m15_bars))
        trend = infer_trend_m15(m15, TrendParams())

        if trend is None:
            logger.info("Trend filter: NONE (skip)")
            return 0

        side = Side.LONG if trend == Trend.BULL else Side.SHORT
        logger.info("Trend filter: %s => %s", trend, side)

        m5 = fetch_rates(mt5, RatesRequest(symbol=args.symbol, timeframe=mt5.TIMEFRAME_M5, count=args.m5_bars))
        plans = plan_h2l2_trades(m5, side, spec, H2L2Params(min_risk_price_units=1.0, cooldown_bars=3))

        g = Guardrails(max_trades_per_day=args.max_trades_day)
        acc, rej = apply_guardrails(plans, g)

        logger.info("Planned=%d Accepted=%d Rejected=%d", len(plans), len(acc), len(rej))
        for t in acc[-5:]:
            logger.info("ACCEPT Signal=%s Exec=%s Side=%s Stop=%.2f %s", t.signal_ts, t.execute_ts, t.side, t.stop, t.reason)

        return 0
    finally:
        c.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())

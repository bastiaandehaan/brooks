# examples/mt5_list_us500.py
from __future__ import annotations

import json
import logging
from typing import Any

import MetaTrader5 as mt5

from utils.mt5_client import Mt5Client, Mt5ConnectionParams

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("mt5_list_us500")


KEY_FIELDS = [
    "name",
    "description",
    "path",
    "currency_base",
    "currency_profit",
    "currency_margin",
    "digits",
    "point",
    "trade_mode",
    "trade_calc_mode",
    "volume_min",
    "volume_max",
    "volume_step",
    "trade_tick_size",
    "trade_tick_value",
    "spread",
    "stops_level",
    "swap_long",
    "swap_short",
    "margin_initial",
    "margin_maintenance",
]


def pick_fields(d: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {k: d.get(k) for k in keys if k in d}


def main() -> None:
    c = Mt5Client(mt5, Mt5ConnectionParams())
    c.initialize()

    try:
        matches = c.symbols_search("US500")
        if not matches:
            logger.error("No symbols containing 'US500' found. We'll print a hint: top 30 symbols.")
            syms = c.symbols_list()[:30]
            for s in syms:
                logger.info("Symbol: %s", getattr(s, "name", s))
            return

        logger.info("US500 candidates:")
        for s in matches:
            logger.info(" - %s", s.name)

        # Neem eerste kandidaat als start; jij kunt daarna kiezen.
        symbol = matches[0].name
        c.ensure_selected(symbol)

        info = c.symbol_info(symbol)

        logger.info(
            "Key fields for %s:\n%s",
            symbol,
            json.dumps(pick_fields(info, KEY_FIELDS), indent=2, default=str),
        )
        logger.info("Full SymbolInfo for %s:\n%s", symbol, json.dumps(info, indent=2, default=str))

    finally:
        c.shutdown()


if __name__ == "__main__":
    main()

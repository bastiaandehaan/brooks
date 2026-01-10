# utils/mt5_data.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RatesRequest:
    symbol: str
    timeframe: int
    count: int


def rates_to_df(rates: Any) -> pd.DataFrame:
    """
    MT5 copy_rates_* returns a numpy structured array.
    Typical fields: time, open, high, low, close, tick_volume, spread, real_volume
    """
    if rates is None:
        return pd.DataFrame()

    # Robust conversion: for numpy structured array, pd.DataFrame(rates) works best
    df = pd.DataFrame(rates)
    if df.empty:
        return df

    cols = set(df.columns.astype(str))
    if "time" in cols:
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(columns={"time": "ts"}).set_index("ts")
    elif "datetime" in cols:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.rename(columns={"datetime": "ts"}).set_index("ts")
    else:
        # Give actionable error
        logger.error("MT5 rates dataframe missing time column. Columns=%s", list(df.columns))
        raise KeyError("MT5 rates missing 'time' column (or 'datetime'). See logs for columns.")

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error("Missing OHLC columns: %s. Columns=%s", missing, list(df.columns))
        raise ValueError(f"Missing OHLC columns: {missing}")

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def fetch_rates(mt5, req: RatesRequest) -> pd.DataFrame:
    logger.info("Fetching rates: %s tf=%s count=%s", req.symbol, req.timeframe, req.count)
    rates = mt5.copy_rates_from_pos(req.symbol, req.timeframe, 0, req.count)
    if rates is None:
        code, msg = mt5.last_error()
        raise RuntimeError(f"copy_rates_from_pos failed: {code} {msg}")

    df = rates_to_df(rates)
    logger.info("Fetched %d bars for %s", len(df), req.symbol)
    return df

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RatesRequest:
    """Request for MT5 rates via copy_rates_from_pos.

    Notes:
      - `pos` is the starting offset.
      - `count` is the number of bars to fetch.
    """

    symbol: str
    timeframe: int
    count: int
    pos: int = 0


def _validate_rates_request(req: RatesRequest) -> None:
    if not req.symbol or not isinstance(req.symbol, str):
        raise ValueError("symbol must be a non-empty string")
    if not isinstance(req.timeframe, int):
        raise ValueError("timeframe must be int")
    if not isinstance(req.count, int) or req.count <= 0:
        raise ValueError("count must be a positive int")
    if not isinstance(req.pos, int) or req.pos < 0:
        raise ValueError("pos must be a non-negative int")


def rates_to_df(rates: Any, *, require_ohlc: bool = True) -> pd.DataFrame:
    """Convert MT5 copy_rates_* output to a DataFrame.

    MT5 returns a numpy structured array with typical fields:
      time, open, high, low, close, tick_volume, spread, real_volume

    Behaviour:
      - Always keeps the raw integer `time` column if present.
      - Creates a UTC datetime index named `ts` from `time` (seconds) or `datetime`.
      - If OHLC is missing:
          * require_ohlc=True  -> raise ValueError
          * require_ohlc=False -> WARNING + return whatever columns exist (time-indexed)
    """
    if rates is None:
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    if df.empty:
        return df

    cols = set(map(str, df.columns))

    # Build datetime index (ts) but do NOT drop the raw integer column(s).
    if "time" in cols:
        ts = pd.to_datetime(df["time"], unit="s", utc=True)
        df.insert(0, "ts", ts)
        df = df.set_index("ts")
    elif "datetime" in cols:
        ts = pd.to_datetime(df["datetime"], utc=True)
        df.insert(0, "ts", ts)
        df = df.set_index("ts")
    else:
        logger.error("MT5 rates dataframe missing time column. Columns=%s", list(df.columns))
        raise KeyError("MT5 rates missing 'time' column (or 'datetime'). See logs for columns.")

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        if require_ohlc:
            logger.error("Missing OHLC columns: %s. Columns=%s", missing, list(df.columns))
            raise ValueError(f"Missing OHLC columns: {missing}")
        logger.warning(
            "OHLC columns not present in MT5 rates. Proceeding with time-index only. Columns=%s",
            list(df.columns),
        )

    # Data hygiene
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def fetch_rates(mt5, req: RatesRequest, *, require_ohlc: bool = True) -> pd.DataFrame:
    """Fetch rates in a single MT5 call."""
    _validate_rates_request(req)
    logger.info("Fetching rates: symbol=%s tf=%s pos=%s count=%s", req.symbol, req.timeframe, req.pos, req.count)

    rates = mt5.copy_rates_from_pos(req.symbol, req.timeframe, req.pos, req.count)
    if rates is None:
        code, msg = mt5.last_error()
        raise RuntimeError(f"copy_rates_from_pos failed: {code} {msg}")

    df = rates_to_df(rates, require_ohlc=require_ohlc)
    logger.info("Fetched %d bars for %s", len(df), req.symbol)
    return df


def fetch_rates_chunked(
    mt5,
    req: RatesRequest,
    *,
    chunk_size: int = 50_000,
    require_ohlc: bool = True,
) -> pd.DataFrame:
    """Fetch rates in chunks and stitch them together (no gaps / no duplicates).

    This exists because some brokers/terminals choke on very large `count` values.

    Stitching strategy:
      - Fetch sequential chunks from req.pos up to req.pos + req.count
      - Concatenate
      - De-dup on index (ts) keep last
    """
    _validate_rates_request(req)
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive int")

    if req.count <= chunk_size:
        logger.debug("Chunked fetch not needed (count=%s <= chunk_size=%s)", req.count, chunk_size)
        return fetch_rates(mt5, req, require_ohlc=require_ohlc)

    logger.info(
        "Chunked fetch start: symbol=%s tf=%s pos=%s count=%s chunk_size=%s",
        req.symbol,
        req.timeframe,
        req.pos,
        req.count,
        chunk_size,
    )

    remaining = req.count
    pos = req.pos
    chunks: list[pd.DataFrame] = []

    while remaining > 0:
        take = min(chunk_size, remaining)
        logger.debug("Fetching chunk: pos=%s count=%s (remaining=%s)", pos, take, remaining)

        rates = mt5.copy_rates_from_pos(req.symbol, req.timeframe, pos, take)
        if rates is None:
            code, msg = mt5.last_error()
            raise RuntimeError(f"copy_rates_from_pos failed: {code} {msg}")

        df_chunk = rates_to_df(rates, require_ohlc=require_ohlc)

        # If MT5 returned empty early, stop.
        if df_chunk.empty:
            logger.warning("Received empty chunk at pos=%s. Stopping early.", pos)
            break

        chunks.append(df_chunk)
        got = len(df_chunk)
        pos += got
        remaining -= got

        # Safety: avoid infinite loops if MT5 returns fewer rows than requested without progress
        if got == 0:
            logger.error("MT5 returned 0 rows for pos=%s take=%s; aborting.", pos, take)
            break

    if not chunks:
        return pd.DataFrame()

    out = pd.concat(chunks, axis=0, ignore_index=False)

    # Hygiene after stitching
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]

    logger.info("Chunked fetch done: got=%d bars for %s", len(out), req.symbol)
    return out

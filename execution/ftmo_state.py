# execution/ftmo_state.py
"""
FTMO State Tracker - Equity-Based Daily/Total PnL

Tracks equity at challenge start and day boundaries (NY timezone).
Provides equity-based daily/total PnL for FTMO Guardian.

Key Features:
- Equity-based (not balance-based) to include open PnL
- NY timezone day reset (consistent with guardrails)
- Deterministic state tracking
- No persistence (in-memory for live session)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FTMOState:
    """
    Tracks equity-based FTMO state.

    CRITICAL: Uses equity (balance + open PnL), not just balance.
    This matches FTMO MetriX behavior where open positions affect limits.
    """

    equity_start_of_challenge: float
    equity_start_of_day: float
    current_day: date | None
    day_tz: str = "America/New_York"
    trading_days: int = 0  # Count of days with trades

    @classmethod
    def initialize(cls, initial_equity: float, day_tz: str = "America/New_York") -> "FTMOState":
        """
        Initialize FTMO state at challenge start.

        Args:
            initial_equity: Account equity at challenge start (from MT5)
            day_tz: Timezone for day boundaries (must match guardrails)

        Returns:
            New FTMOState instance
        """
        logger.info("🎯 FTMO State initialized: equity=%.2f, tz=%s", initial_equity, day_tz)
        return cls(
            equity_start_of_challenge=initial_equity,
            equity_start_of_day=initial_equity,
            current_day=None,
            day_tz=day_tz,
            trading_days=0,
        )

    def update(self, equity_now: float, timestamp_utc: pd.Timestamp) -> bool:
        """
        Update state with current equity and time.
        Handles NY day reset automatically.

        Args:
            equity_now: Current account equity (from MT5)
            timestamp_utc: Current timestamp (UTC, tz-aware)

        Returns:
            True if day reset occurred, False otherwise
        """
        if timestamp_utc.tzinfo is None:
            raise ValueError("timestamp_utc must be tz-aware (UTC)")

        # Convert to NY day
        ny_day = timestamp_utc.tz_convert(self.day_tz).date()

        # Check for day reset
        if self.current_day is None:
            # First call - initialize but DON'T update equity_start_of_day
            # (it was set in __init__ with challenge start equity)
            self.current_day = ny_day
            logger.info(
                "📅 FTMO day initialized: %s, equity_start=%.2f", ny_day, self.equity_start_of_day
            )
            return False

        if ny_day != self.current_day:
            # Day boundary crossed - RESET
            old_day = self.current_day
            daily_pnl = equity_now - self.equity_start_of_day

            # Reset day state
            self.current_day = ny_day
            self.equity_start_of_day = equity_now

            logger.info(
                "🔄 FTMO day reset: %s → %s, daily_pnl=$%.2f, new_equity_start=%.2f",
                old_day,
                ny_day,
                daily_pnl,
                equity_now,
            )
            return True

        # Same day - no changes to equity_start_of_day
        return False

    def get_daily_pnl(self, equity_now: float) -> float:
        """
        Get daily PnL in USD (equity-based).

        Args:
            equity_now: Current account equity

        Returns:
            Daily PnL in USD (can be negative)
        """
        return equity_now - self.equity_start_of_day

    def get_total_pnl(self, equity_now: float) -> float:
        """
        Get total PnL since challenge start in USD (equity-based).

        Args:
            equity_now: Current account equity

        Returns:
            Total PnL in USD (can be negative)
        """
        return equity_now - self.equity_start_of_challenge

    def get_total_drawdown(self, equity_now: float, running_max_equity: float) -> float:
        """
        Get total drawdown from running max equity.

        Args:
            equity_now: Current account equity
            running_max_equity: Highest equity reached during challenge

        Returns:
            Drawdown in USD (always <= 0)
        """
        return min(0.0, equity_now - running_max_equity)

    def increment_trading_day(self) -> None:
        """
        Increment trading days counter.
        Should be called when a trade is executed on a new day.
        """
        self.trading_days += 1
        logger.info("📊 Trading days: %d", self.trading_days)

    def get_status_summary(self, equity_now: float, running_max_equity: float) -> dict:
        """
        Get complete status summary for logging/monitoring.

        Args:
            equity_now: Current account equity
            running_max_equity: Highest equity reached

        Returns:
            Dict with all FTMO metrics
        """
        daily_pnl = self.get_daily_pnl(equity_now)
        total_pnl = self.get_total_pnl(equity_now)
        total_dd = self.get_total_drawdown(equity_now, running_max_equity)

        return {
            "equity_now": equity_now,
            "equity_start_challenge": self.equity_start_of_challenge,
            "equity_start_day": self.equity_start_of_day,
            "daily_pnl_usd": daily_pnl,
            "total_pnl_usd": total_pnl,
            "total_dd_usd": total_dd,
            "current_day": str(self.current_day),
            "trading_days": self.trading_days,
        }

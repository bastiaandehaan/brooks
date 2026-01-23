# execution/trade_gate.py
"""
FTMO Trade Gate - Central Kill-Switch

Single choke point for all trade placement.
Enforces FTMO limits BEFORE order execution.

Flow:
1. Check FTMO Guardian can_trade()
2. Cap risk to FTMO headroom
3. Return (allowed, reason, capped_risk)

Non-negotiable: NO trade bypasses this gate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from execution.ftmo_guardian import FTMOGuardian
from execution.ftmo_state import FTMOState

logger = logging.getLogger(__name__)


@dataclass
class TradeGateResult:
    """Result from trade gate check"""

    allowed: bool
    reason: str
    capped_risk_usd: float

    @property
    def blocked(self) -> bool:
        return not self.allowed


def check_ftmo_trade_gate(
    *,
    equity_now: float,
    ftmo_state: FTMOState,
    ftmo_guardian: FTMOGuardian,
    requested_risk_usd: float,
    min_risk_threshold: float = 10.0,
) -> TradeGateResult:
    """
    FTMO Trade Gate - Check if trade is allowed and cap risk.

    This is the SINGLE CENTRAL GATE for all trade placement.
    Must be called BEFORE order placement, not just in planning.

    Args:
        equity_now: Current account equity (from MT5)
        ftmo_state: FTMO state tracker
        ftmo_guardian: FTMO rule enforcer
        requested_risk_usd: Requested risk in USD from strategy
        min_risk_threshold: Minimum risk USD to proceed (avoid noise trades)

    Returns:
        TradeGateResult with (allowed, reason, capped_risk_usd)

    Example:
        result = check_ftmo_trade_gate(
            equity_now=10500.0,
            ftmo_state=state,
            ftmo_guardian=guardian,
            requested_risk_usd=100.0,
        )

        if result.blocked:
            logger.warning("Trade blocked: %s", result.reason)
            return

        # Use result.capped_risk_usd for position sizing
        risk_pct = 100.0 * (result.capped_risk_usd / equity_now)
    """
    # Get current FTMO state
    daily_pnl = ftmo_state.get_daily_pnl(equity_now)

    # Step 1: Check if FTMO Guardian allows trade
    allowed, reason = ftmo_guardian.can_trade(
        current_balance=equity_now,  # Use equity, not balance!
        daily_pnl=daily_pnl,
        open_risk=requested_risk_usd,
    )

    if not allowed:
        logger.warning("⛔ FTMO GATE BLOCKED: %s", reason)
        return TradeGateResult(
            allowed=False,
            reason=reason,
            capped_risk_usd=0.0,
        )

    # Step 2: Cap risk to FTMO headroom
    max_allowed_risk = ftmo_guardian.get_max_allowed_risk(equity_now, daily_pnl)
    capped_risk = min(requested_risk_usd, max_allowed_risk)

    # Step 3: Check minimum threshold
    if capped_risk < min_risk_threshold:
        logger.warning(
            "⛔ FTMO GATE BLOCKED: Risk too small (%.2f < %.2f threshold)",
            capped_risk,
            min_risk_threshold,
        )
        return TradeGateResult(
            allowed=False,
            reason=f"FTMO headroom too small (${capped_risk:.2f} < ${min_risk_threshold:.2f})",
            capped_risk_usd=0.0,
        )

    # Trade allowed!
    if capped_risk < requested_risk_usd:
        logger.info(
            "✅ FTMO GATE: Trade allowed, risk capped: $%.2f → $%.2f (headroom: $%.2f)",
            requested_risk_usd,
            capped_risk,
            max_allowed_risk,
        )
    else:
        logger.info(
            "✅ FTMO GATE: Trade allowed, risk=$%.2f (headroom: $%.2f)",
            capped_risk,
            max_allowed_risk,
        )

    return TradeGateResult(
        allowed=True,
        reason="ALLOWED",
        capped_risk_usd=capped_risk,
    )


def convert_risk_to_pct(risk_usd: float, equity: float) -> float:
    """
    Convert USD risk to percentage of equity.

    Args:
        risk_usd: Risk in USD
        equity: Current account equity

    Returns:
        Risk as percentage (e.g., 1.0 for 1%)
    """
    if equity <= 0:
        raise ValueError("Equity must be > 0")
    return 100.0 * (risk_usd / equity)


def log_ftmo_status(
    equity_now: float,
    ftmo_state: FTMOState,
    ftmo_guardian: FTMOGuardian,
    running_max_equity: float,
) -> None:
    """
    Log FTMO status for monitoring (console/telegram).
    Should be called periodically in live loop.

    Args:
        equity_now: Current equity
        ftmo_state: FTMO state tracker
        ftmo_guardian: FTMO rule enforcer
        running_max_equity: Highest equity reached
    """
    status_summary = ftmo_state.get_status_summary(equity_now, running_max_equity)
    daily_pnl = status_summary["daily_pnl_usd"]

    # Get guardian account status
    account_status = ftmo_guardian.get_account_status(
        current_balance=equity_now,
        daily_pnl=daily_pnl,
        trading_days=status_summary["trading_days"],
    )

    # Get headroom
    max_risk = ftmo_guardian.get_max_allowed_risk(equity_now, daily_pnl)

    print("\n" + "=" * 70)
    print("  💼 FTMO STATUS")
    print("=" * 70)
    print(f"  Equity: ${equity_now:,.2f}")
    print(f"  Daily P&L: ${daily_pnl:+,.2f} / ${ftmo_guardian.max_daily_loss_usd:,.2f}")
    print(
        f"  Total Profit: ${account_status['total_profit']:+,.2f} / Target: ${account_status['profit_target']:,.2f}"
    )
    print(
        f"  Total DD: ${account_status['total_drawdown']:,.2f} / ${ftmo_guardian.max_total_loss_usd:,.2f}"
    )
    print(f"  Health: {account_status['health']}")
    print(
        f"  Trading Days: {status_summary['trading_days']} / {ftmo_guardian.rules.min_trading_days}"
    )
    print(f"  Max Risk Headroom: ${max_risk:,.2f}")
    print("=" * 70)

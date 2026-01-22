# execution/ftmo_guardian.py
"""
FTMO Challenge Rule Enforcer
Prevents violations that would fail the challenge
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class FTMOAccountType(Enum):
    """FTMO account types with different rules"""

    CHALLENGE_10K = "10k"
    CHALLENGE_25K = "25k"
    CHALLENGE_50K = "50k"
    CHALLENGE_100K = "100k"
    CHALLENGE_200K = "200k"
    VERIFICATION = "verification"
    FUNDED = "funded"


@dataclass
class FTMORules:
    """
    FTMO Challenge Rules (as of 2026)

    Key Rules:
    1. Max Daily Loss: 5% of initial balance
    2. Max Total Drawdown: 10% of initial balance
    3. Profit Target: 10% for Challenge, 5% for Verification
    4. Minimum Trading Days: 4 days (at least 1 trade per day)
    5. No weekend holding (optional but recommended)
    6. No news trading in first 2 minutes
    """

    account_type: FTMOAccountType
    initial_balance: float

    # Loss limits (STRICT - account closes if breached!)
    max_daily_loss_pct: float = 5.0  # 5% max daily loss
    max_total_loss_pct: float = 10.0  # 10% max total drawdown

    # Profit targets
    profit_target_pct: float = 10.0  # 10% for Challenge, 5% for Verification

    # Trading constraints
    min_trading_days: int = 4
    max_lot_size: float = 5.0  # Adjust based on account size

    # Conservative buffers (stop BEFORE hitting limits)
    daily_loss_buffer_pct: float = 1.0  # Stop at 4% instead of 5%
    total_loss_buffer_pct: float = 2.0  # Stop at 8% instead of 10%

    @classmethod
    def for_10k_challenge(cls) -> FTMORules:
        """Standard 10k FTMO Challenge rules"""
        return cls(
            account_type=FTMOAccountType.CHALLENGE_10K,
            initial_balance=10000.0,
            max_daily_loss_pct=5.0,
            max_total_loss_pct=10.0,
            profit_target_pct=10.0,
            daily_loss_buffer_pct=1.0,  # Stop at 4% daily
            total_loss_buffer_pct=2.0,  # Stop at 8% total
        )

    @classmethod
    def for_verification(cls, balance: float) -> FTMORules:
        """FTMO Verification phase rules (after passing Challenge)"""
        return cls(
            account_type=FTMOAccountType.VERIFICATION,
            initial_balance=balance,
            max_daily_loss_pct=5.0,
            max_total_loss_pct=10.0,
            profit_target_pct=5.0,  # Lower target for Verification
            daily_loss_buffer_pct=1.0,
            total_loss_buffer_pct=2.0,
        )


class FTMOGuardian:
    """
    Enforces FTMO rules and prevents account violations

    Usage:
        guardian = FTMOGuardian(rules=FTMORules.for_10k_challenge())

        # Before taking trade
        can_trade, reason = guardian.can_trade(
            current_balance=9800,
            daily_pnl=-150,
            open_risk=50
        )

        if not can_trade:
            print(f"TRADE BLOCKED: {reason}")
    """

    def __init__(self, rules: FTMORules):
        self.rules = rules
        self.initial_balance = rules.initial_balance

        # Calculate absolute limits
        self.max_daily_loss_usd = rules.initial_balance * rules.max_daily_loss_pct / 100
        self.max_total_loss_usd = rules.initial_balance * rules.max_total_loss_pct / 100

        # Calculate safe buffers (stop BEFORE limits)
        self.safe_daily_loss_usd = (
            rules.initial_balance * (rules.max_daily_loss_pct - rules.daily_loss_buffer_pct) / 100
        )
        self.safe_total_loss_usd = (
            rules.initial_balance * (rules.max_total_loss_pct - rules.total_loss_buffer_pct) / 100
        )

        logger.info("FTMO Guardian initialized:")
        logger.info(f"  Account Type: {rules.account_type.value}")
        logger.info(f"  Initial Balance: ${rules.initial_balance:,.2f}")
        logger.info(f"  Max Daily Loss: ${self.max_daily_loss_usd:,.2f} (HARD LIMIT)")
        logger.info(f"  Safe Daily Loss: ${self.safe_daily_loss_usd:,.2f} (with buffer)")
        logger.info(f"  Max Total Loss: ${self.max_total_loss_usd:,.2f} (HARD LIMIT)")
        logger.info(f"  Safe Total Loss: ${self.safe_total_loss_usd:,.2f} (with buffer)")

    def can_trade(
        self,
        current_balance: float,
        daily_pnl: float,
        open_risk: float = 0.0,
        check_time: bool = True,
    ) -> tuple[bool, str]:
        """
        Check if trading is allowed

        Args:
            current_balance: Current account balance
            daily_pnl: Today's P&L so far (negative = loss)
            open_risk: Risk of proposed trade in USD
            check_time: Check for news/weekend restrictions

        Returns:
            (can_trade: bool, reason: str)
        """
        # 1. Check Daily Loss Limit
        projected_daily_loss = abs(min(daily_pnl, 0)) + open_risk

        if projected_daily_loss > self.safe_daily_loss_usd:
            return False, (
                f"DAILY LOSS LIMIT APPROACHING: "
                f"${projected_daily_loss:.2f} would exceed safe limit "
                f"(${self.safe_daily_loss_usd:.2f}). "
                f"STOP TRADING TODAY!"
            )

        # 2. Check Total Drawdown Limit
        total_drawdown = self.initial_balance - current_balance
        projected_total_loss = total_drawdown + open_risk

        if projected_total_loss > self.safe_total_loss_usd:
            return False, (
                f"TOTAL DRAWDOWN LIMIT APPROACHING: "
                f"${projected_total_loss:.2f} would exceed safe limit "
                f"(${self.safe_total_loss_usd:.2f}). "
                f"ACCOUNT IN DANGER!"
            )

        # 3. Check if account already breached HARD limits (should never happen)
        if abs(min(daily_pnl, 0)) >= self.max_daily_loss_usd:
            return False, (
                f"⛔ CRITICAL: DAILY LOSS HARD LIMIT BREACHED! "
                f"${abs(daily_pnl):.2f} >= ${self.max_daily_loss_usd:.2f}. "
                f"STOP ALL TRADING IMMEDIATELY! ACCOUNT MAY BE CLOSED!"
            )

        if total_drawdown >= self.max_total_loss_usd:
            return False, (
                f"⛔ CRITICAL: TOTAL DRAWDOWN HARD LIMIT BREACHED! "
                f"${total_drawdown:.2f} >= ${self.max_total_loss_usd:.2f}. "
                f"STOP ALL TRADING IMMEDIATELY! ACCOUNT WILL BE CLOSED!"
            )

        # 4. Warning if getting close to limits
        daily_loss_pct = (projected_daily_loss / self.max_daily_loss_usd) * 100
        total_loss_pct = (projected_total_loss / self.max_total_loss_usd) * 100

        warnings = []
        if daily_loss_pct > 70:
            warnings.append(
                f"⚠️  Daily loss at {daily_loss_pct:.1f}% of limit "
                f"(${projected_daily_loss:.2f}/${self.max_daily_loss_usd:.2f})"
            )

        if total_loss_pct > 70:
            warnings.append(
                f"⚠️  Total drawdown at {total_loss_pct:.1f}% of limit "
                f"(${projected_total_loss:.2f}/${self.max_total_loss_usd:.2f})"
            )

        if warnings:
            warning_msg = " | ".join(warnings)
            logger.warning(warning_msg)

        # All checks passed
        return True, "OK - All FTMO rules satisfied"

    def get_max_allowed_risk(self, current_balance: float, daily_pnl: float) -> float:
        """
        Calculate maximum allowed risk for next trade

        Args:
            current_balance: Current account balance
            daily_pnl: Today's P&L so far

        Returns:
            Max risk in USD
        """
        # Calculate headroom for both limits
        daily_headroom = self.safe_daily_loss_usd - abs(min(daily_pnl, 0))

        total_drawdown = self.initial_balance - current_balance
        total_headroom = self.safe_total_loss_usd - total_drawdown

        # Take the more restrictive limit
        max_risk = min(daily_headroom, total_headroom)

        # Never allow negative risk
        return max(0, max_risk)

    def get_account_status(
        self, current_balance: float, daily_pnl: float, total_trades: int = 0, trading_days: int = 0
    ) -> dict:
        """
        Get comprehensive account status report

        Returns:
            Dict with account health metrics
        """
        total_drawdown = self.initial_balance - current_balance
        daily_loss = abs(min(daily_pnl, 0))

        # Calculate percentages of limits used
        daily_loss_pct = (daily_loss / self.max_daily_loss_usd) * 100
        total_loss_pct = (total_drawdown / self.max_total_loss_usd) * 100

        # Calculate profit progress
        total_profit = current_balance - self.initial_balance
        profit_target_usd = self.initial_balance * self.rules.profit_target_pct / 100
        profit_progress_pct = (
            (total_profit / profit_target_usd) * 100 if profit_target_usd > 0 else 0
        )

        # Determine account health
        if daily_loss_pct > 90 or total_loss_pct > 90:
            health = "CRITICAL"
        elif daily_loss_pct > 70 or total_loss_pct > 70:
            health = "WARNING"
        elif daily_loss_pct > 50 or total_loss_pct > 50:
            health = "CAUTION"
        else:
            health = "HEALTHY"

        return {
            "health": health,
            "current_balance": current_balance,
            "initial_balance": self.initial_balance,
            "total_profit": total_profit,
            "total_drawdown": total_drawdown,
            "daily_pnl": daily_pnl,
            "daily_loss": daily_loss,
            "daily_loss_pct": daily_loss_pct,
            "daily_loss_limit": self.max_daily_loss_usd,
            "total_loss_pct": total_loss_pct,
            "total_loss_limit": self.max_total_loss_usd,
            "profit_target": profit_target_usd,
            "profit_progress_pct": profit_progress_pct,
            "total_trades": total_trades,
            "trading_days": trading_days,
            "min_trading_days": self.rules.min_trading_days,
        }

    def print_status(self, status: dict) -> None:
        """Print formatted account status"""
        print("\n" + "=" * 60)
        print(f"  📊 FTMO ACCOUNT STATUS: {status['health']}")
        print("=" * 60)
        print("\n💰 BALANCE:")
        print(f"  Current  : ${status['current_balance']:,.2f}")
        print(f"  Initial  : ${status['initial_balance']:,.2f}")
        print(f"  Profit   : ${status['total_profit']:+,.2f}")

        print("\n📉 DAILY RISK:")
        print(f"  Today P&L: ${status['daily_pnl']:+,.2f}")
        print(f"  Daily Loss: ${status['daily_loss']:,.2f} / ${status['daily_loss_limit']:,.2f}")
        print(f"  Usage    : {status['daily_loss_pct']:.1f}% of limit")

        print("\n📊 TOTAL DRAWDOWN:")
        print(f"  Drawdown : ${status['total_drawdown']:,.2f} / ${status['total_loss_limit']:,.2f}")
        print(f"  Usage    : {status['total_loss_pct']:.1f}% of limit")

        print("\n🎯 PROFIT TARGET:")
        print(f"  Target   : ${status['profit_target']:,.2f}")
        print(f"  Progress : {status['profit_progress_pct']:.1f}%")

        print("\n📅 TRADING ACTIVITY:")
        print(f"  Trades   : {status['total_trades']}")
        print(f"  Days     : {status['trading_days']} / {status['min_trading_days']} minimum")

        print("\n" + "=" * 60 + "\n")


# Example usage
if __name__ == "__main__":
    # Test FTMO Guardian
    print("Testing FTMO Guardian...\n")

    # 10k Challenge
    rules = FTMORules.for_10k_challenge()
    guardian = FTMOGuardian(rules)

    # Scenario 1: Normal trading
    print("\n📋 SCENARIO 1: Normal trading day")
    can_trade, reason = guardian.can_trade(
        current_balance=10050,  # Up $50
        daily_pnl=50,  # Profit today
        open_risk=50,  # Next trade risk
    )
    print(f"Can trade: {can_trade}")
    print(f"Reason: {reason}")

    # Scenario 2: After some losses
    print("\n📋 SCENARIO 2: After -$300 loss today")
    can_trade, reason = guardian.can_trade(current_balance=9700, daily_pnl=-300, open_risk=50)
    print(f"Can trade: {can_trade}")
    print(f"Reason: {reason}")

    # Scenario 3: Approaching daily limit
    print("\n📋 SCENARIO 3: Approaching daily limit (-$380)")
    can_trade, reason = guardian.can_trade(current_balance=9620, daily_pnl=-380, open_risk=50)
    print(f"Can trade: {can_trade}")
    print(f"Reason: {reason}")

    # Get account status
    print("\n📊 ACCOUNT STATUS:")
    status = guardian.get_account_status(
        current_balance=9620, daily_pnl=-380, total_trades=25, trading_days=5
    )
    guardian.print_status(status)

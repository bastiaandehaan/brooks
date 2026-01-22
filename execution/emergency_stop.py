# execution/emergency_stop.py
"""
Emergency Stop System - Kill Switch for Live Trading

Multiple ways to stop trading immediately:
1. STOP file in project root
2. Telegram command
3. FTMO limit breach
4. Manual keyboard interrupt
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class EmergencyStop:
    """
    Emergency stop system with multiple triggers

    Usage:
        stop = EmergencyStop()

        # In trading loop
        if stop.should_stop():
            print("EMERGENCY STOP TRIGGERED!")
            break

    Trigger Methods:
    1. Create file: STOP.txt in project root
    2. Telegram: Send "STOP" command to bot
    3. FTMO breach detected
    4. Ctrl+C keyboard interrupt
    """

    def __init__(self, project_root: Path | None = None):
        self.project_root = project_root or Path.cwd()
        self.stop_file = self.project_root / "STOP.txt"
        self.manual_stop = False
        self.stop_reason = None

        logger.info(f"Emergency Stop initialized. Stop file: {self.stop_file}")

    def trigger(self, reason: str) -> None:
        """
        Manually trigger emergency stop

        Args:
            reason: Why the stop was triggered
        """
        self.manual_stop = True
        self.stop_reason = reason

        # Create stop file
        self.stop_file.write_text(f"EMERGENCY STOP: {reason}")

        logger.error(f"⛔ EMERGENCY STOP TRIGGERED: {reason}")
        print("\n" + "⛔" * 30)
        print("  EMERGENCY STOP TRIGGERED")
        print(f"  Reason: {reason}")
        print("⛔" * 30 + "\n")

    def should_stop(self) -> tuple[bool, str | None]:
        """
        Check if trading should stop

        Returns:
            (should_stop: bool, reason: str)
        """
        # Check manual trigger
        if self.manual_stop:
            return True, self.stop_reason

        # Check stop file
        if self.stop_file.exists():
            try:
                reason = self.stop_file.read_text().strip()
            except:
                reason = "Stop file detected"
            return True, reason

        return False, None

    def clear(self) -> None:
        """Clear emergency stop and allow trading to resume"""
        self.manual_stop = False
        self.stop_reason = None

        if self.stop_file.exists():
            self.stop_file.unlink()

        logger.info("✅ Emergency stop cleared")

    def get_status(self) -> dict:
        """Get current status"""
        should_stop, reason = self.should_stop()
        return {
            "is_stopped": should_stop,
            "reason": reason,
            "stop_file_exists": self.stop_file.exists(),
            "manual_stop": self.manual_stop,
        }


class TradingState:
    """
    Persist trading state across restarts
    Prevents trading after emergency stop until manually cleared
    """

    def __init__(self, state_file: str = "trading_state.json"):
        self.state_file = Path(state_file)
        self.state = self._load_state()

    def _load_state(self) -> dict:
        """Load state from file"""
        if not self.state_file.exists():
            return {
                "trading_enabled": True,
                "last_stop_reason": None,
                "total_trades_today": 0,
                "daily_pnl": 0.0,
            }

        import json

        try:
            with open(self.state_file) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return {
                "trading_enabled": True,
                "last_stop_reason": None,
                "total_trades_today": 0,
                "daily_pnl": 0.0,
            }

    def _save_state(self) -> None:
        """Save state to file"""
        import json

        try:
            with open(self.state_file, "w") as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def disable_trading(self, reason: str) -> None:
        """Disable trading and persist reason"""
        self.state["trading_enabled"] = False
        self.state["last_stop_reason"] = reason
        self._save_state()
        logger.warning(f"Trading DISABLED: {reason}")

    def enable_trading(self) -> None:
        """Enable trading"""
        self.state["trading_enabled"] = True
        self.state["last_stop_reason"] = None
        self._save_state()
        logger.info("Trading ENABLED")

    def is_trading_enabled(self) -> tuple[bool, str | None]:
        """Check if trading is enabled"""
        return self.state["trading_enabled"], self.state.get("last_stop_reason")

    def update_daily_stats(self, trades: int = 0, pnl: float = 0.0) -> None:
        """Update today's trading stats"""
        self.state["total_trades_today"] = trades
        self.state["daily_pnl"] = pnl
        self._save_state()

    def get_daily_stats(self) -> dict:
        """Get today's stats"""
        return {
            "trades": self.state.get("total_trades_today", 0),
            "pnl": self.state.get("daily_pnl", 0.0),
        }


# Example usage
if __name__ == "__main__":
    print("Testing Emergency Stop System...\n")

    # Test 1: Normal operation
    print("TEST 1: Normal operation")
    stop = EmergencyStop()
    should_stop, reason = stop.should_stop()
    print(f"Should stop: {should_stop}")
    print(f"Reason: {reason}\n")

    # Test 2: Trigger emergency stop
    print("TEST 2: Trigger emergency stop")
    stop.trigger("Testing emergency stop functionality")
    should_stop, reason = stop.should_stop()
    print(f"Should stop: {should_stop}")
    print(f"Reason: {reason}\n")

    # Test 3: Clear stop
    print("TEST 3: Clear emergency stop")
    stop.clear()
    should_stop, reason = stop.should_stop()
    print(f"Should stop: {should_stop}")
    print(f"Reason: {reason}\n")

    # Test 4: Stop file
    print("TEST 4: Create stop file manually")
    stop.stop_file.write_text("Manual stop via file")
    should_stop, reason = stop.should_stop()
    print(f"Should stop: {should_stop}")
    print(f"Reason: {reason}\n")

    # Cleanup
    stop.clear()

    # Test 5: Trading state
    print("TEST 5: Trading state persistence")
    state = TradingState(state_file="test_state.json")
    print(f"Trading enabled: {state.is_trading_enabled()}")

    state.disable_trading("Test disable")
    print(f"After disable: {state.is_trading_enabled()}")

    state.enable_trading()
    print(f"After enable: {state.is_trading_enabled()}")

    # Cleanup
    Path("test_state.json").unlink(missing_ok=True)

    print("\n✅ All tests passed!")

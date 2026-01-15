# utils/logging_setup.py
"""
Advanced Logging Infrastructure for Brooks Trading Framework

Implements contextuele logging met Run ID voor traceability van:
- Parallelle backtest runs
- Live trading sessies
- Debugging en incident analysis
"""
import logging
import sys


class RunIdFilter(logging.Filter):
    """
    Injects a 'run_id' attribute into LogRecords.
    Allows tracking distinct backtest runs or live sessions in shared logs.
    Essential for correlating trades in institutional_audit.py with system events.
    """

    def __init__(self, run_id: str):
        super().__init__()
        self.run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        # Injecteer run_id alleen als het nog niet bestaat
        if not hasattr(record, "run_id"):
            record.run_id = self.run_id
        return True


def setup_logging(level: str = "INFO", run_id: str = "main") -> None:
    """
    Configures the root logger with a standardized format and Run ID.
    This ensures consistent observability across all modules.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        run_id: Unique identifier for this execution session (e.g. timestamp or UUID)

    Usage:
        from utils.logging_setup import setup_logging

        # In backtest runner:
        setup_logging("INFO", run_id="backtest_20260115_143022")

        # In main.py:
        setup_logging("INFO", run_id="live_20260115_143022")
    """
    # Define standard format with run_id
    # Format: [Timestamp] [Level] run=[RunID] [Module]: [Message]
    fmt = "%(asctime)s %(levelname)s run=%(run_id)s %(name)s: %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    # Configure StreamHandler (Console Output)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # Add the Context Filter to the handler
    run_filter = RunIdFilter(run_id)
    handler.addFilter(run_filter)

    # Configure Root Logger
    root = logging.getLogger()

    # Robustly set level (handles string input case-insensitively)
    log_level = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(log_level)

    # Clear existing handlers to prevent duplicate logs (double printing)
    if root.handlers:
        root.handlers.clear()

    root.addHandler(handler)

    # Suppress noisy libraries explicitly
    # Matplotlib en urllib3 genereren veel debug info die niet relevant is voor trading logica
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("execution.guardrails").setLevel(logging.WARNING)  # Voorkom guardrail spam

    # Log setup confirmation
    logger = logging.getLogger("logging_setup")
    logger.info(f"Logging initialized: level={level}, run_id={run_id}")


# Example usage en testing
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("  TESTING LOGGING SETUP")
    print("=" * 80 + "\n")

    # Setup logging met test run_id
    setup_logging("INFO", run_id="test_20260115_143500")

    # Test verschillende log levels
    test_logger = logging.getLogger("test_module")

    test_logger.debug("This is a DEBUG message (should not appear in INFO level)")
    test_logger.info("This is an INFO message ✓")
    test_logger.warning("This is a WARNING message ⚠️")
    test_logger.error("This is an ERROR message ❌")

    # Simuleer logs van verschillende modules
    strategies_logger = logging.getLogger("strategies.h2l2")
    strategies_logger.info("H2 LONG signal detected at 5847.50")

    execution_logger = logging.getLogger("execution.guardrails")
    execution_logger.info("Trade accepted: in session")

    backtest_logger = logging.getLogger("backtest.runner")
    backtest_logger.info("Backtest completed: 180 days, 380 trades")

    print("\n" + "=" * 80)
    print("  ✅ LOGGING TEST COMPLETE")
    print("=" * 80)
    print("\nObserve dat alle logs het run_id bevatten: run=test_20260115_143500")
    print("Dit maakt het mogelijk om logs van meerdere runs te correleren.\n")
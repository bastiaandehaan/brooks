import pandas as pd

from scripts.live_monitor import SessionConfig, session_state


def test_session_state_cutoff():
    cfg = SessionConfig(session_start="09:30", session_end="16:00", trade_cutoff="15:30")

    ts = pd.Timestamp("2026-01-13 15:29:59", tz="America/New_York")
    assert session_state(cfg, ts)[0] == "ACTIVE"

    ts = pd.Timestamp("2026-01-13 15:30:00", tz="America/New_York")
    assert session_state(cfg, ts)[0] == "CUTOFF"

    ts = pd.Timestamp("2026-01-13 16:00:01", tz="America/New_York")
    assert session_state(cfg, ts)[0] == "OUTSIDE"

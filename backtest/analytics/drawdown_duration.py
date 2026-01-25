# backtest/analytics/drawdown_duration.py
"""
Drawdown Duration Analysis

Analyzes underwater periods to understand:
- How long does the strategy stay in drawdown?
- Distribution of drawdown durations
- Recovery characteristics

Notes:
- Works with pandas Series indexed by datetime (NY dates, UTC, etc.).
- Avoids Timestamp + int arithmetic (pandas 2.x/3.x strictness).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DrawdownPeriod:
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    trough_date: pd.Timestamp
    duration_days: int
    recovery_days: int
    max_depth_r: float
    trough_equity_r: float


@dataclass(frozen=True)
class DrawdownStats:
    periods: List[DrawdownPeriod]
    avg_duration: float
    max_duration: int
    avg_recovery: float
    max_depth: float
    time_underwater_pct: float


def _to_series(equity_curve_r) -> pd.Series:
    if isinstance(equity_curve_r, pd.Series):
        s = equity_curve_r.copy()
    elif isinstance(equity_curve_r, (list, tuple, np.ndarray)):
        s = pd.Series(equity_curve_r)
    else:
        raise TypeError("equity_curve_r must be a pandas Series or array-like.")

    s = s.dropna()
    if s.empty:
        return s

    # Ensure datetime index for duration calculations
    if not isinstance(s.index, pd.DatetimeIndex):
        # If it’s already a column-like index (e.g., RangeIndex), we can’t compute days meaningfully.
        # Promote to DatetimeIndex only if values are parseable.
        try:
            s.index = pd.to_datetime(s.index)
        except Exception as e:
            raise TypeError(
                "equity_curve_r must have a DatetimeIndex (or index convertible to datetime) "
                "to compute drawdown durations in days."
            ) from e

    s = s.sort_index()
    return s


def analyze_drawdown_durations(
    equity_curve_r: pd.Series,
    min_depth_r: float = 0.5,
) -> DrawdownStats:
    """
    Identify drawdown periods where drawdown depth <= -min_depth_r.

    Parameters
    ----------
    equity_curve_r : pd.Series
        Equity curve in R units, indexed by datetime.
    min_depth_r : float
        Minimum drawdown depth (in R) to count as a drawdown period.

    Returns
    -------
    DrawdownStats
    """
    eq = _to_series(equity_curve_r)
    if eq.empty:
        return DrawdownStats(
            periods=[],
            avg_duration=0.0,
            max_duration=0,
            avg_recovery=0.0,
            max_depth=0.0,
            time_underwater_pct=0.0,
        )

    # Drawdown is equity - running peak (<= 0)
    peak = eq.cummax()
    dd = eq - peak

    # Underwater mask
    underwater = dd <= -float(min_depth_r)
    if underwater.sum() == 0:
        return DrawdownStats(
            periods=[],
            avg_duration=0.0,
            max_duration=0,
            avg_recovery=0.0,
            max_depth=float(dd.min()) if not dd.empty else 0.0,
            time_underwater_pct=0.0,
        )

    periods: List[DrawdownPeriod] = []
    in_dd = False
    dd_start_pos: Optional[int] = None

    idx = eq.index
    n = len(eq)

    def _pos_to_date(pos: int) -> pd.Timestamp:
        return idx[pos]

    for pos in range(n):
        if underwater.iloc[pos] and not in_dd:
            in_dd = True
            dd_start_pos = pos

        if in_dd:
            # Exit when we are no longer underwater
            if (not underwater.iloc[pos]) or (pos == n - 1):
                # end_pos should be the last underwater point (if we exited, that's pos-1)
                if underwater.iloc[pos]:
                    end_pos = pos
                else:
                    end_pos = pos - 1

                assert dd_start_pos is not None
                start_pos = dd_start_pos

                period_dd = dd.iloc[start_pos : end_pos + 1]
                period_eq = eq.iloc[start_pos : end_pos + 1]

                # Trough is the minimum dd (most negative)
                trough_pos_in_slice = int(period_dd.values.argmin())
                trough_pos = start_pos + trough_pos_in_slice

                start_date = _pos_to_date(start_pos)
                end_date = _pos_to_date(end_pos)
                trough_date = _pos_to_date(trough_pos)

                duration_days = int((end_date - start_date).days)

                # Recovery: first date AFTER end_pos where equity exceeds prior peak at start_pos
                start_peak = float(peak.iloc[start_pos])
                recovery_pos: Optional[int] = None
                for j in range(end_pos + 1, n):
                    if float(eq.iloc[j]) >= start_peak:
                        recovery_pos = j
                        break

                if recovery_pos is None:
                    # Not recovered within series
                    recovery_days = 0
                else:
                    recovery_days = int((_pos_to_date(recovery_pos) - end_date).days)

                max_depth_r = float(period_dd.min())
                trough_equity_r = float(eq.iloc[trough_pos])

                periods.append(
                    DrawdownPeriod(
                        start_date=start_date,
                        end_date=end_date,
                        trough_date=trough_date,
                        duration_days=duration_days,
                        recovery_days=recovery_days,
                        max_depth_r=max_depth_r,
                        trough_equity_r=trough_equity_r,
                    )
                )

                # Reset state
                in_dd = False
                dd_start_pos = None

    durations = [p.duration_days for p in periods]
    recoveries = [p.recovery_days for p in periods if p.recovery_days > 0]
    max_depth = (
        float(min([p.max_depth_r for p in periods])) if periods else float(dd.min())
    )

    # Time underwater over full series (based on mask)
    time_underwater_pct = float(underwater.mean())

    return DrawdownStats(
        periods=periods,
        avg_duration=float(np.mean(durations)) if durations else 0.0,
        max_duration=int(np.max(durations)) if durations else 0,
        avg_recovery=float(np.mean(recoveries)) if recoveries else 0.0,
        max_depth=max_depth,
        time_underwater_pct=time_underwater_pct,
    )


def print_drawdown_duration_report(stats: DrawdownStats, top_n: int = 5) -> None:
    print("\n" + "=" * 80)
    print("  DRAWDOWN DURATION ANALYSIS")
    print("=" * 80)

    if not stats.periods:
        print("No drawdown periods found above threshold.")
        print("=" * 80 + "\n")
        return

    print(f"Periods found           : {len(stats.periods)}")
    print(f"Avg duration (days)     : {stats.avg_duration:.1f}")
    print(f"Max duration (days)     : {stats.max_duration}")
    print(f"Avg recovery (days)     : {stats.avg_recovery:.1f}")
    print(f"Max depth (R)           : {stats.max_depth:.2f}R")
    print(f"Time underwater (%)     : {stats.time_underwater_pct * 100:.1f}%")

    deepest = sorted(stats.periods, key=lambda p: p.max_depth_r)[:top_n]
    print("\nDeepest drawdowns:")
    for i, p in enumerate(deepest, 1):
        print(
            f"  {i:>2}. {p.start_date.date()} -> {p.end_date.date()} | "
            f"depth={p.max_depth_r:.2f}R | dur={p.duration_days}d | rec={p.recovery_days}d | "
            f"trough={p.trough_date.date()}"
        )

    print("=" * 80 + "\n")

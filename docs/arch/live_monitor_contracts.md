# Live monitor contracts (MVP)

## Problem
Live monitor calls RiskManager.size_position with keyword args that RiskManager does not accept.
Also regime filter compares chop_ratio against a hardcoded/default threshold (1.0) instead of CLI value.

## Decision
1) Make RiskManager.size_position accept `risk_pct` (float, percent like 0.5) as an optional keyword,
   backwards compatible with existing callers.
2) Ensure RegimeParams gets `chop_threshold=args.chop_threshold` and logging compares against that value.

## DoD
- pytest green
- live_monitor runs without TypeError on sizing
- regime log prints: `chop_ratio=... > <your_threshold>`

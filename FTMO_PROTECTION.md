# FTMO Protection System

## Overview

Deterministisch FTMO loss limit enforcement VOOR order placement.

**Key Feature:** Equity-based limits (includes open PnL) zoals FTMO MetriX.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   TRADE FLOW                        │
├─────────────────────────────────────────────────────┤
│                                                     │
│  1. Strategy generates signal                       │
│  2. Calculate requested risk (USD)                  │
│  3. ✋ FTMO TRADE GATE (CRITICAL)                   │
│     ├── Check FTMOState (equity, daily PnL)        │
│     ├── Call FTMOGuardian.can_trade()              │
│     ├── Cap risk to FTMO headroom                  │
│     └── Return (allowed, capped_risk)              │
│  4. IF BLOCKED → Stop, log, alert                  │
│  5. IF ALLOWED → Size position with capped risk    │
│  6. Apply guardrails (session, max trades/day)     │
│  7. Place order                                     │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## Components

### 1. `execution/ftmo_state.py`

**Purpose:** Track equity-based FTMO state

**Features:**

- Equity at challenge start
- Equity at day start (NY timezone)
- Automatic day reset
- Daily/total PnL calculations

**Usage:**

```python
from execution.ftmo_state import FTMOState

# Initialize at challenge start
state = FTMOState.initialize(initial_equity=10000.0)

# Update every iteration
now_utc = pd.Timestamp.now(tz="UTC")
day_reset = state.update(equity_now, now_utc)

# Get metrics
daily_pnl = state.get_daily_pnl(equity_now)
total_pnl = state.get_total_pnl(equity_now)
```

### 2. `execution/trade_gate.py`

**Purpose:** Central kill-switch for trade placement

**Features:**

- Checks FTMOGuardian before trade
- Caps risk to FTMO headroom
- Converts USD risk to risk_pct for sizing

**Usage:**

```python
from execution.trade_gate import check_ftmo_trade_gate

result = check_ftmo_trade_gate(
    equity_now=equity,
    ftmo_state=state,
    ftmo_guardian=guardian,
    requested_risk_usd=100.0,
)

if result.blocked:
    logger.warning("Trade blocked: %s", result.reason)
    return

# Use capped risk
risk_pct = 100.0 * (result.capped_risk_usd / equity_now)
```

### 3. `config/environments/production.yaml`

**Purpose:** FTMO account configuration

**Critical Settings:**

```yaml
ftmo:
  enabled: true
  account_size: 10000    # Match your FTMO account
  profit_target: 500     # +5%
  max_daily_loss: 500    # -5%
  max_total_loss: 1000   # -10%
  min_trading_days: 2
  daily_buffer: 0.85     # Use 85% (15% safety margin)
  total_buffer: 0.90     # Use 90% (10% safety margin)
```

**IMPORTANT:**

- Buffers are MANDATORY - never set to 1.0
- Update values for 25k/50k/100k accounts

### 4. `scripts/live_monitor.py` (updated)

**Changes:**

- Initialize FTMOState at startup
- Update state every iteration
- Call trade_gate BEFORE sizing
- Use equity (not balance) everywhere
- Log FTMO status periodically

## Testing

### Unit Tests

```bash
# Run FTMO protection tests
python tests/test_ftmo_protection.py
```

**Expected output:**

```
✅ TEST 1: Day Reset Logic - PASSED
✅ TEST 2: Trade Gate Blocking - PASSED
✅ TEST 3: Risk Capping - PASSED
✅ TEST 4: Pass Mode Trigger - PASSED
```

### Dry-Run Test

```bash
# Test live monitor with FTMO protection (no real orders)
python scripts/live_monitor.py \
    --strategy config/strategies/us500_sniper.yaml \
    --env config/environments/production.yaml \
    --interval 60
```

**Monitor output:**

```
💼 FTMO STATUS
==================================================================
  Equity: $10,050.00
  Daily P&L: +$50.00 / $500.00
  Total P&L: +$50.00 / Target: $500.00
  Max DD: $0.00 / $1,000.00
  Status: ALLOWED
  Trading Days: 1 / 2
  Max Risk Headroom: $375.00
==================================================================
```

## Safety Guarantees

### 1. **Deterministic Kill-Switch**

- NO trade can bypass the gate
- Enforced BEFORE order placement
- Equity-based (includes open PnL)

### 2. **Safety Buffers**

```
Daily: 0.85 buffer = use $425 of $500 limit ($75 safety margin)
Total: 0.90 buffer = use $900 of $1000 limit ($100 safety margin)
```

**Why?** Protect against:

- Slippage on order fill
- Spread widening
- Gap opens
- MT5 equity calculation timing

### 3. **Day Reset Consistency**

- Uses EXACT same NY timezone as guardrails
- Day boundary = midnight NY time
- Daily limits reset automatically

### 4. **Equity-Based Tracking**

```python
# CORRECT (matches FTMO MetriX)
equity_now = mt5.account_info().equity  # balance + open PnL
daily_pnl = equity_now - equity_start_of_day

# WRONG (would miss open PnL)
balance_now = mt5.account_info().balance
daily_pnl = balance_now - balance_start_of_day  # ❌
```

## FTMO Account Configurations

### 10k Challenge

```yaml
ftmo:
  account_size: 10000
  profit_target: 500      # +5%
  max_daily_loss: 500     # -5%
  max_total_loss: 1000    # -10%
```

### 25k Challenge

```yaml
ftmo:
  account_size: 25000
  profit_target: 1250     # +5%
  max_daily_loss: 1250    # -5%
  max_total_loss: 2500    # -10%
```

### 50k Challenge

```yaml
ftmo:
  account_size: 50000
  profit_target: 2500     # +5%
  max_daily_loss: 2500    # -5%
  max_total_loss: 5000    # -10%
```

## Monitoring & Alerts

### Console Output

FTMO status logged every 5 minutes (configurable):

```yaml
logging:
  ftmo_status_interval: 300  # seconds
```

### Telegram (Optional)

```yaml
telegram:
  enabled: true
  bot_token: "YOUR_TOKEN"
  chat_id: "YOUR_CHAT_ID"
```

Sends alerts on:

- Daily limit approaching (>80% used)
- Total limit approaching (>90% used)
- Trade blocked by FTMO gate
- Profit target reached

## Emergency Stop

Create `STOP.txt` in project root to halt monitor immediately:

```bash
echo "Emergency stop" > STOP.txt
```

Monitor checks for this file every iteration.

## Common Issues

### Issue: "Trade blocked - STOP_DAILY"

**Cause:** Daily loss limit reached
**Fix:** Wait for NY day reset (midnight ET)

### Issue: "Risk too small"

**Cause:** FTMO headroom < $10 threshold
**Fix:** Normal - protection working, skip trading

### Issue: "Day reset not working"

**Cause:** Timezone mismatch with guardrails
**Fix:** Ensure both use `America/New_York`

### Issue: "Limits breached despite protection"

**Cause:** Buffers too aggressive (>0.95)
**Fix:** Use 0.85 daily, 0.90 total (mandatory)

## Production Checklist

Before going live:

- [ ] Run `python tests/test_ftmo_protection.py`
- [ ] Verify FTMO config matches your account
- [ ] Test dry-run for 1 hour
- [ ] Confirm day reset works (wait for midnight NY)
- [ ] Set buffers to 0.85 daily / 0.90 total
- [ ] Enable Telegram alerts (optional)
- [ ] Create STOP.txt emergency procedure
- [ ] Verify equity-based calculations in logs

## FAQ

**Q: Why equity-based instead of balance-based?**
A: FTMO MetriX uses equity (balance + open PnL). If you have a losing open position, it counts toward your limit
immediately.

**Q: Why buffers < 1.0?**
A: Slippage, spread, gaps can cause small overruns. 15% safety margin (0.85 buffer) protects against this.

**Q: What happens after profit target?**
A: Status shows target reached. Future: implement pass mode (reduce risk to 10%, max 1 trade/day).

**Q: Can I disable FTMO protection for testing?**
A: Yes, set `ftmo.enabled: false` in production.yaml. But TEST with protection ON before live!

**Q: How do I know if protection is working?**
A: Run tests, check console output, verify trades are blocked/capped as expected.

## Support

Issues? Check:

1. Logs in `logs/` directory
2. FTMO status output in console
3. Test results from `test_ftmo_protection.py`
4. Verify config values match your FTMO account

## Version

**FTMO Protection v1.0** - January 2026

- Equity-based tracking
- Deterministic kill-switch
- Safety buffers
- Day reset (NY timezone)
- Trade gate integration
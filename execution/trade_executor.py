# execution/trade_executor.py
"""
FTMO-Safe Trade Executor - Single Atomic Choke Point

This module provides the ONLY way to execute trades in the Brooks framework.
All trade execution MUST go through execute_trade_ftmo_safe() to guarantee
FTMO compliance and prevent bypasses.

NON-NEGOTIABLE: Direct order placement outside this module is FORBIDDEN.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import MetaTrader5 as mt5
import pandas as pd

from execution.ftmo_guardian import FTMOGuardian
from execution.ftmo_state import FTMOState
from execution.guardrails import Guardrails, apply_guardrails
from execution.risk_manager import RiskManager
from execution.trade_gate import check_ftmo_trade_gate, convert_risk_to_pct
from strategies.h2l2 import PlannedTrade
from utils.symbol_spec import SymbolSpec

logger = logging.getLogger(__name__)


@dataclass
class TradeExecutionResult:
    """Result from atomic trade execution"""

    success: bool
    reason: str

    # If success=True:
    ticket: Optional[int] = None
    filled_price: float = 0.0
    filled_lots: float = 0.0
    actual_risk_usd: float = 0.0

    # If success=False (blocked):
    block_stage: Optional[str] = None  # "FTMO_GATE" | "GUARDRAILS" | "ORDER_PLACEMENT"


def execute_trade_ftmo_safe(
    *,
    planned_trade: PlannedTrade,
    equity_now: float,
    ftmo_state: FTMOState,
    ftmo_guardian: FTMOGuardian,
    guardrails: Guardrails,
    risk_manager: RiskManager,
    spec: SymbolSpec,
    requested_risk_pct: float,
    mt5_module,
) -> TradeExecutionResult:
    """
    ATOMIC FTMO-SAFE TRADE EXECUTION

    This is the ONLY function that should execute trades in the Brooks framework.
    It enforces the complete execution pipeline in one non-bypassable sequence:

    1. FTMO Trade Gate (equity-based limits)
    2. Risk Sizing (with FTMO-capped risk)
    3. Guardrails (session/daily limits)
    4. Order Placement (MT5 execution)

    Args:
        planned_trade: Trade plan from strategy
        equity_now: Current account equity (from MT5)
        ftmo_state: FTMO state tracker
        ftmo_guardian: FTMO rule enforcer
        guardrails: Session/daily guardrails
        risk_manager: Position sizer
        spec: Symbol specification
        requested_risk_pct: Requested risk % (will be capped by FTMO)
        mt5_module: MetaTrader5 module reference

    Returns:
        TradeExecutionResult with success status and details

    CRITICAL: This function is the SINGLE CHOKE POINT for all trade execution.
    Bypassing this function violates FTMO protection guarantees.
    """

    # === STAGE 1: FTMO TRADE GATE ===
    # Convert risk % to USD for gate check
    requested_risk_usd = (requested_risk_pct / 100.0) * equity_now

    gate_result = check_ftmo_trade_gate(
        equity_now=equity_now,
        ftmo_state=ftmo_state,
        ftmo_guardian=ftmo_guardian,
        requested_risk_usd=requested_risk_usd,
        min_risk_threshold=10.0,
    )

    if gate_result.blocked:
        logger.warning("⛔ FTMO GATE BLOCKED: %s", gate_result.reason)
        return TradeExecutionResult(
            success=False,
            reason=f"FTMO Gate: {gate_result.reason}",
            block_stage="FTMO_GATE",
        )

    # Use FTMO-capped risk
    capped_risk_pct = convert_risk_to_pct(gate_result.capped_risk_usd, equity_now)

    logger.info(
        "✅ FTMO Gate: Trade allowed, risk=%.2f%% ($%.2f)",
        capped_risk_pct,
        gate_result.capped_risk_usd,
    )

    # === STAGE 2: POSITION SIZING ===
    try:
        lots, final_risk_usd = risk_manager.size_position(
            balance=equity_now,  # Use equity!
            entry=planned_trade.entry,
            stop=planned_trade.stop,
            risk_pct=capped_risk_pct,  # FTMO-capped
            spec=spec,
        )
    except Exception as e:
        logger.error("❌ Position sizing failed: %s", e)
        return TradeExecutionResult(
            success=False,
            reason=f"Sizing error: {e}",
            block_stage="SIZING",
        )

    if lots <= 0:
        logger.warning("⚠️  Position sizing returned 0 lots")
        return TradeExecutionResult(
            success=False,
            reason="Zero lots (position too small)",
            block_stage="SIZING",
        )

    logger.info("✅ Sizing: lots=%.2f, risk=$%.2f", lots, final_risk_usd)

    # === STAGE 3: GUARDRAILS ===
    accepted, rejected = apply_guardrails([planned_trade], guardrails)

    if not accepted:
        reason = rejected[0][1] if rejected else "Unknown guardrail rejection"
        logger.warning("⛔ Guardrails blocked trade: %s", reason)
        return TradeExecutionResult(
            success=False,
            reason=f"Guardrails: {reason}",
            block_stage="GUARDRAILS",
        )

    logger.info("✅ Guardrails: Trade accepted")

    # === STAGE 4: ORDER PLACEMENT ===
    # Build MT5 order request
    tick = mt5_module.symbol_info_tick(spec.name)
    if not tick:
        logger.error("❌ Failed to get tick for %s", spec.name)
        return TradeExecutionResult(
            success=False,
            reason="No tick data available",
            block_stage="ORDER_PLACEMENT",
        )

    # Determine order type and price
    if planned_trade.side.value == "LONG":
        order_type = mt5_module.ORDER_TYPE_BUY
        price = tick.ask
    else:  # SHORT
        order_type = mt5_module.ORDER_TYPE_SELL
        price = tick.bid

    # Build request
    request = {
        "action": mt5_module.TRADE_ACTION_DEAL,
        "symbol": spec.name,
        "volume": float(lots),
        "type": order_type,
        "price": float(price),
        "sl": float(planned_trade.stop),
        "tp": float(planned_trade.tp),
        "deviation": 10,  # Max slippage in points
        "magic": 777,  # Brooks magic number
        "comment": f"Brooks-{planned_trade.reason[:20]}",
        "type_time": mt5_module.ORDER_TIME_GTC,
        "type_filling": mt5_module.ORDER_FILLING_IOC,
    }

    logger.info(
        "📤 Sending order: %s %.2f lots @ %.2f (SL=%.2f TP=%.2f)",
        planned_trade.side.value,
        lots,
        price,
        planned_trade.stop,
        planned_trade.tp,
    )

    # Execute order
    result = mt5_module.order_send(request)

    if result.retcode != mt5_module.TRADE_RETCODE_DONE:
        logger.error(
            "❌ Order failed: retcode=%s, comment=%s",
            result.retcode,
            result.comment,
        )
        return TradeExecutionResult(
            success=False,
            reason=f"MT5 error: {result.retcode} - {result.comment}",
            block_stage="ORDER_PLACEMENT",
        )

    # === SUCCESS ===
    logger.info(
        "✅ ORDER FILLED: ticket=%s, price=%.2f, lots=%.2f",
        result.order,
        result.price,
        result.volume,
    )

    return TradeExecutionResult(
        success=True,
        reason="Trade executed successfully",
        ticket=result.order,
        filled_price=result.price,
        filled_lots=result.volume,
        actual_risk_usd=final_risk_usd,
    )


# === ENFORCEMENT: No direct order_send allowed ===
# Any code that calls mt5.order_send() outside this module is a VIOLATION
# and breaks FTMO protection guarantees.

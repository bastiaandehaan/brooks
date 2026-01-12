import logging

from utils.symbol_spec import SymbolSpec

logger = logging.getLogger(__name__)


class RiskManager:
    def __init__(self, risk_per_trade_pct: float = 1.0):
        self.risk_per_trade_pct = risk_per_trade_pct

    def calculate_lot_size(
        self,
        balance: float,
        spec: SymbolSpec,
        entry: float,
        stop: float,
    ) -> float:
        """
        Risk-based position sizing.
        Alle inputs worden gelogd zodat 'small risk' issues zichtbaar zijn.
        """

        if balance <= 0:
            logger.info("RISK: balance<=0 → no sizing")
            return 0.0

        # 1. USD risico
        risk_usd = balance * (self.risk_per_trade_pct / 100.0)

        # 2. Prijsafstand
        price_risk = abs(entry - stop)
        if price_risk <= 0:
            logger.info(
                "RISK: invalid price_risk entry=%.5f stop=%.5f → skip",
                entry,
                stop,
            )
            return 0.0

        # 3. Waarde per punt per lot
        value_per_lot_step = spec.usd_per_price_unit_per_lot
        if value_per_lot_step <= 0:
            logger.error(
                "RISK: invalid symbol spec usd_per_price_unit_per_lot=%s",
                value_per_lot_step,
            )
            return 0.0

        # 4. Ruwe lot size
        raw_lots = risk_usd / (price_risk * value_per_lot_step)

        # 5. Afronden naar broker constraints
        final_lots = spec.round_volume_down(raw_lots)

        logger.info(
            "RISK: balance=%.2f risk_pct=%.2f risk_usd=%.2f "
            "entry=%.5f stop=%.5f price_risk=%.5f "
            "usd_per_point=%.5f raw_lots=%.4f final_lots=%.4f",
            balance,
            self.risk_per_trade_pct,
            risk_usd,
            entry,
            stop,
            price_risk,
            value_per_lot_step,
            raw_lots,
            final_lots,
        )

        return final_lots

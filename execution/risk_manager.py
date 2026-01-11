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
            stop: float
    ) -> float:
        """
        Berekent het aantal lots gebaseerd op een percentage risico van het saldo.
        """
        if balance <= 0:
            return 0.0

        # 1. Hoeveel USD willen we riskeren?
        risk_usd = balance * (self.risk_per_trade_pct / 100.0)

        # 2. Wat is de afstand tot de stop in prijs-punten?
        price_risk = abs(entry - stop)
        if price_risk <= 0:
            return 0.0

        # 3. Bereken waarde per lot per prijs-unit
        # spec.usd_per_price_unit_per_lot geeft aan hoeveel 1.0 lot oplevert per 1.0 punt beweging
        value_per_lot_step = spec.usd_per_price_unit_per_lot

        if value_per_lot_step <= 0:
            logger.error(f"Symbol {spec.name} has invalid tick/contract settings.")
            return 0.0

        # 4. Bereken rauwe lot size
        # Lots = Dollar Risico / (Prijs Risico * Waarde per punt)
        raw_lots = risk_usd / (price_risk * value_per_lot_step)

        # 5. Afronden naar broker limieten (volume_step en volume_min/max)
        final_lots = spec.round_volume_down(raw_lots)

        logger.info(
            f"Risk Calc: Risk ${risk_usd:.2f} | Dist {price_risk:.2f} pts | "
            f"Raw {raw_lots:.4f} -> Final {final_lots} lots"
        )

        return final_lots
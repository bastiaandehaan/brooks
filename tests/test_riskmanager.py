import pytest
from execution.risk_manager import RiskManager, RiskParams

def test_size_position_accepts_risk_pct_kwarg():
    rm = RiskManager(RiskParams(min_risk_pts=1.0, fees_usd=0.0))
    lots, risk_usd = rm.size_position(
        balance=10000.0,
        entry=7000.0,
        stop=6990.0,          # 10 pts
        tick_size=0.01,
        contract_size=1.0,    # 1 USD per point per lot (example)
        risk_pct=0.5,         # 0.5% of 10k = $50
        fees_usd=0.0,
    )
    assert lots > 0
    assert abs(risk_usd - 50.0) < 1e-6

def test_size_position_rejects_too_small_risk():
    rm = RiskManager(RiskParams(min_risk_pts=5.0, fees_usd=0.0))
    with pytest.raises(ValueError):
        rm.size_position(
            balance=10000.0,
            entry=7000.0,
            stop=6998.0,        # 2 pts < min_risk_pts
            tick_size=0.01,
            contract_size=1.0,
            risk_pct=0.5,
        )

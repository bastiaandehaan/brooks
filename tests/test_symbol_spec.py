from utils.symbol_spec import SymbolSpec


def test_us500_usd_per_price_unit_per_lot():
    spec = SymbolSpec(
        name="US500.cash",
        digits=2,
        point=0.01,
        tick_size=0.01,
        tick_value=0.01,
        contract_size=1.0,
        volume_min=0.01,
        volume_step=0.01,
        volume_max=1000.0,
    )
    assert spec.usd_per_price_unit_per_lot == 1.0


def test_round_volume_down():
    spec = SymbolSpec(
        name="X",
        digits=2,
        point=0.01,
        tick_size=0.01,
        tick_value=0.01,
        contract_size=1.0,
        volume_min=0.01,
        volume_step=0.01,
        volume_max=1.0,
    )
    assert spec.round_volume_down(0.009) == 0.0
    assert spec.round_volume_down(0.019) == 0.01
    assert spec.round_volume_down(1.234) == 1.0

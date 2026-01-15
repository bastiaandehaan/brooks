# test_config.py
from strategies.config import StrategyConfig
import os


def test_load_production():
    yaml_path = "config/production.yaml"
    if not os.path.exists(yaml_path):
        print(f"❌ FOUT: {yaml_path} niet gevonden!")
        return

    try:
        config = StrategyConfig.from_yaml(yaml_path)
        is_valid, msg = config.validate()

        if is_valid:
            print("✅ SUCCES: production.yaml is correct ingelezen en gevalideerd.")
            print(f"   - Symbool: {config.symbol}")
            print(f"   - Risk: {config.risk_pct}%")
            print(f"   - Chop Threshold: {config.regime_params.chop_threshold}")
        else:
            print(f"❌ VALIDATIE FOUT: {msg}")
    except Exception as e:
        print(f"❌ CRASH: Er is iets mis in de code van config.py: {e}")


if __name__ == "__main__":
    test_load_production()
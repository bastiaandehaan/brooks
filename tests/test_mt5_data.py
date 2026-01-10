from utils.mt5_data import rates_to_df


def test_rates_to_df_sorts_and_dedupes():
    rates = [
        {"time": 200, "open": 2, "high": 3, "low": 1, "close": 2.5},
        {"time": 100, "open": 1, "high": 2, "low": 0.5, "close": 1.5},
        {"time": 100, "open": 1.1, "high": 2.1, "low": 0.6, "close": 1.6},
    ]
    df = rates_to_df(rates)
    assert len(df) == 2
    assert df.iloc[0]["open"] == 1.1

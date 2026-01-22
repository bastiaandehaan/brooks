import numpy as np
import pytest

from utils.mt5_data import RatesRequest, fetch_rates_chunked


class FakeMT5:
    def __init__(self, total: int):
        self.total = total
        self._last_error = (0, "OK")

    def last_error(self):
        return self._last_error

    def copy_rates_from_pos(self, symbol, timeframe, pos, count):
        # Simuleer MT5: geef max tot total terug, anders None/empty
        if pos < 0 or count <= 0:
            self._last_error = (-2, "Invalid params")
            return None
        if pos >= self.total:
            return np.array([], dtype=[("time", "i8")])

        end = min(self.total, pos + count)
        times = np.arange(pos, end, dtype=np.int64)
        return np.array(list(zip(times, strict=True)), dtype=[("time", "i8")])


def test_fetch_rates_chunked_stitches_no_gaps():
    mt5 = FakeMT5(total=120_000)
    req = RatesRequest("X", 1, count=103_680, pos=0)

    out = fetch_rates_chunked(mt5, req, chunk_size=50_000, require_ohlc=False)

    assert len(out) == 103_680
    # Controleer dat we exact pos..pos+count-1 hebben
    assert out["time"].iloc[0] == 0
    assert out["time"].iloc[-1] == 103_679
    # Geen gaten
    assert np.all(np.diff(out["time"].to_numpy()) == 1)


def test_fetch_rates_chunked_small_uses_single_path():
    mt5 = FakeMT5(total=10_000)
    req = RatesRequest("X", 1, count=9_000, pos=0)

    out = fetch_rates_chunked(mt5, req, chunk_size=50_000, require_ohlc=False)

    assert len(out) == 9_000


def test_fetch_rates_chunked_validates_inputs():
    mt5 = FakeMT5(total=10_000)
    with pytest.raises(ValueError):
        fetch_rates_chunked(mt5, RatesRequest("X", 1, count=0, pos=0))

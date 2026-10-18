import numpy as np
import pytest
from scipy.signal import freqz

from abvid.data.augmentation import RATE, augment, draw, excerpt, peaking, shape


def test_shape_is_zero_mean_unit_rms_and_rejects_silence():
    y = shape(np.random.default_rng(0).standard_normal(1000) * 5 + 3)
    assert abs(y.mean()) < 1e-12 and abs(np.sqrt(np.mean(y**2)) - 1) < 1e-12
    with pytest.raises(ValueError):
        shape(np.full(1000, 2.0))


@pytest.mark.parametrize("gain", [-6.0, 0.0, 4.5])
def test_peaking_filter_gain_at_centre_and_unity_far_away(gain):
    b, a = peaking(1000.0, 1.0, gain)
    w, h = freqz(b, a, worN=[2 * np.pi * 1000 / RATE, 2 * np.pi * 7900 / RATE])
    assert 20 * np.log10(abs(h[0])) == pytest.approx(gain, abs=1e-9)
    assert abs(20 * np.log10(abs(h[1]))) < 0.5


def test_draws_respect_declared_ranges_and_are_replayable():
    a, b = draw(np.random.default_rng(7)), draw(np.random.default_rng(7))
    assert a == b and len(a["filters"]) == 3 and 10 <= a["snr_db"] <= 30
    assert all(100 <= f["f0_hz"] <= 3000 and 0.7 <= f["q"] <= 2 and -6 <= f["gain_db"] <= 6 for f in a["filters"])


def test_augment_output_and_snr():
    rng = np.random.default_rng(1)
    x = np.sin(2 * np.pi * 220 * np.arange(32000) / RATE)
    background, start = excerpt(rng.standard_normal(50000), rng)
    params = {"filters": [{"f0_hz": 500.0, "q": 1.0, "gain_db": 0.0}] * 3, "snr_db": 20.0}
    y = augment(x, background, params)
    assert y.dtype == np.float32 and y.shape == (32000,) and 0 <= start <= 18000
    clean = shape(x)
    residual = shape(y) * np.sqrt(1 + 10 ** (-2)) - clean
    assert np.sqrt(np.mean(residual**2)) == pytest.approx(0.1, rel=0.05)

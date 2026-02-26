import numpy as np

from tsfresh.feature_extraction.feature_calculators import (
    _lempel_ziv_complexity_original,
    lempel_ziv_complexity,
)


def _make_signals(n, rng):
    t = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float64)
    trig = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 13 * t)
    damped = np.exp(-5 * t) * np.sin(2 * np.pi * 7 * t)
    poly = 0.5 * t + 0.1 * t**2 - 0.05 * t**3
    sin_step = np.sin(2 * np.pi * 3 * t) + (t > 0.6).astype(np.float64)
    plateau = np.sin(2 * np.pi * 4 * t)
    plateau = plateau.copy()
    for start in (0.2, 0.5, 0.75):
        i0 = int(start * n)
        i1 = min(n, i0 + int(0.02 * n))
        plateau[i0:i1] = plateau[i0]
    const = np.full(n, 3.0, dtype=np.float64)
    repeats = np.tile(np.array([0.0, 1.0, 1.0, 2.0], dtype=np.float64), n // 4 + 1)[
        :n
    ]
    rand = rng.normal(0.0, 1.0, size=n)
    nan_series = rand.copy()
    if n > 5:
        nan_series[3] = np.nan
    return [trig, damped, poly, sin_step, plateau, const, repeats, rand, nan_series]


def test_lempel_ziv_complexity_parity():
    rng = np.random.default_rng(0)
    for n in (50, 200, 1000):
        for x in _make_signals(n, rng):
            for bins in (2, 5, 10, 20):
                ref = _lempel_ziv_complexity_original(x, bins)
                new = lempel_ziv_complexity(x, bins)
                np.testing.assert_allclose(new, ref, rtol=1e-12, atol=1e-12)

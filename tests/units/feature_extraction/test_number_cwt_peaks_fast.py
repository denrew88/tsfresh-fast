import numpy as np

from tsfresh.feature_extraction.feature_calculators import (
    _number_cwt_peaks_original,
    number_cwt_peaks,
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
    rand = rng.normal(0.0, 1.0, size=n)
    return [trig, damped, poly, sin_step, plateau, rand]


def test_number_cwt_peaks_sparse_noise_parity():
    rng = np.random.default_rng(0)
    for n in (200, 1000):
        for x in _make_signals(n, rng):
            for nparam in (1, 5):
                ref = _number_cwt_peaks_original(x, nparam)
                new = number_cwt_peaks(x, nparam)
                assert ref == new

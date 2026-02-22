# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)

import numpy as np
import pytest

from tsfresh.feature_extraction.feature_calculators import (
    _approximate_entropy_original,
    _augmented_dickey_fuller_original,
    _change_quantiles_original,
    _sample_entropy_original,
    approximate_entropy_block_combiner,
    augmented_dickey_fuller,
    change_quantiles_qcut_exact_combiner,
    sample_entropy_block_combiner,
)


LENGTHS = [50, 200, 1000, 10000]


APPROX_PARAMS = [
    {"m": m, "r": r} for m in (2, 3) for r in (0.1, 0.2, 0.3)
]
APPROX_PARAMS_MEDIUM = [{"m": 2, "r": 0.1}, {"m": 3, "r": 0.2}]
APPROX_PARAMS_LARGE = [{"m": 2, "r": 0.2}]

SAMPEN_PARAMS = [{"m": m, "r": r} for m in (2, 3) for r in (0.1, 0.2)]
SAMPEN_PARAMS_MEDIUM = [{"m": 2, "r": 0.1}, {"m": 3, "r": 0.2}]
SAMPEN_PARAMS_LARGE = [{"m": 2, "r": 0.2}]

CHANGE_QUANTILES_PARAMS = [
    {"ql": ql, "qh": qh, "isabs": isabs, "f_agg": f_agg}
    for (ql, qh) in ((0.1, 0.9), (0.2, 0.8), (0.25, 0.75))
    for isabs in (True, False)
    for f_agg in ("mean", "var", "std", "median", "min", "max", "sum")
]

ADF_PARAMS = [
    {"autolag": autolag, "attr": attr}
    for autolag in ("AIC", "BIC", "t-stat", None)
    for attr in ("teststat", "pvalue", "usedlag")
]


def _smooth_signal_bank(length, rng):
    t = np.linspace(0.0, 1.0, length, endpoint=False, dtype=np.float64)
    trig = np.sin(2 * np.pi * 3 * t) + 0.3 * np.cos(2 * np.pi * 5 * t)
    damped = np.exp(-5.0 * t) * np.sin(2 * np.pi * 5 * t)
    poly = 0.1 * t + 0.2 * t**2 + 0.01 * rng.standard_normal(length)
    step = 0.5 * (1.0 / (1.0 + np.exp(-40.0 * (t - 0.5))))
    sin_step = np.sin(2 * np.pi * 2 * t) + step
    plateau = np.sin(2 * np.pi * 2 * t)
    seg_len = max(1, length // 20)
    plateau[length // 4 : length // 4 + seg_len] = 0.3
    plateau[length // 2 : length // 2 + seg_len] = -0.2
    return {
        "trig": trig,
        "damped": damped,
        "poly": poly,
        "sin_step": sin_step,
        "plateau": plateau,
    }


@pytest.fixture(scope="module")
def random_series():
    rng = np.random.default_rng(0)
    return {length: rng.standard_normal(length) for length in LENGTHS}


@pytest.fixture(scope="module")
def constant_series():
    return {length: np.full(length, 1.2345) for length in LENGTHS}


@pytest.fixture(scope="module")
def nan_series(random_series):
    data = {}
    for length, series in random_series.items():
        series_with_nan = series.copy()
        series_with_nan[length // 2] = np.nan
        data[length] = series_with_nan
    return data


@pytest.fixture(scope="module")
def smooth_series():
    rng = np.random.default_rng(0)
    data = []
    for length in LENGTHS:
        signals = _smooth_signal_bank(length, rng)
        for name, series in signals.items():
            data.append((name, length, series))
    return data


def _assert_keys_values_close(actual_items, expected_items, context=""):
    actual = dict(actual_items)
    expected = dict(expected_items)
    assert set(actual.keys()) == set(expected.keys()), context
    for key in expected:
        np.testing.assert_allclose(
            actual[key],
            expected[key],
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
            err_msg=f"{context} key={key}",
        )


def _approximate_entropy_expected(x, params):
    expected = []
    for config in params:
        key = f"m_{config['m']}__r_{config['r']:g}"
        expected.append(
            (key, _approximate_entropy_original(x, config["m"], config["r"]))
        )
    return expected


def _sample_entropy_expected(x, params):
    expected = []
    for config in params:
        key = f"m_{config['m']}__r_{config['r']:g}"
        expected.append(
            (key, _sample_entropy_original(x, config["m"], config["r"]))
        )
    return expected


def _change_quantiles_expected(x, params):
    return [
        (
            f'f_agg_"{config["f_agg"]}"__isabs_{config["isabs"]}__qh_{config["qh"]}__ql_{config["ql"]}',
            _change_quantiles_original(
                x, config["ql"], config["qh"], config["isabs"], config["f_agg"]
            ),
        )
        for config in params
    ]


def _params_for_length(length, full, medium, large):
    if length >= 10000:
        return large
    if length >= 1000:
        return medium
    return full


@pytest.mark.parametrize(
    ("length", "params"),
    [
        (50, APPROX_PARAMS),
        (200, APPROX_PARAMS),
        (1000, APPROX_PARAMS_MEDIUM),
        (10000, APPROX_PARAMS_LARGE),
    ],
)
def test_approximate_entropy_random(random_series, length, params):
    x = random_series[length]
    expected = _approximate_entropy_expected(x, params)
    actual = approximate_entropy_block_combiner(x, params)
    _assert_keys_values_close(actual, expected)


def test_approximate_entropy_constant(constant_series):
    x = constant_series[50]
    expected = _approximate_entropy_expected(x, APPROX_PARAMS)
    actual = approximate_entropy_block_combiner(x, APPROX_PARAMS)
    _assert_keys_values_close(actual, expected)


def test_approximate_entropy_nan(nan_series):
    x = nan_series[50]
    expected = _approximate_entropy_expected(x, APPROX_PARAMS)
    actual = approximate_entropy_block_combiner(x, APPROX_PARAMS)
    _assert_keys_values_close(actual, expected)


def test_approximate_entropy_short():
    x = np.array([1.0, 2.0, 3.0])
    expected = _approximate_entropy_expected(x, APPROX_PARAMS)
    actual = approximate_entropy_block_combiner(x, APPROX_PARAMS)
    _assert_keys_values_close(actual, expected)


def test_approximate_entropy_smooth(smooth_series):
    for name, length, series in smooth_series:
        params = _params_for_length(
            length, APPROX_PARAMS, APPROX_PARAMS_MEDIUM, APPROX_PARAMS_LARGE
        )
        expected = _approximate_entropy_expected(series, params)
        actual = approximate_entropy_block_combiner(series, params)
        _assert_keys_values_close(
            actual, expected, context=f"approximate_entropy {name} n={length}"
        )


@pytest.mark.parametrize(
    ("length", "params"),
    [
        (50, SAMPEN_PARAMS),
        (200, SAMPEN_PARAMS),
        (1000, SAMPEN_PARAMS_MEDIUM),
        (10000, SAMPEN_PARAMS_LARGE),
    ],
)
def test_sample_entropy_random(random_series, length, params):
    x = random_series[length]
    expected = _sample_entropy_expected(x, params)
    actual = sample_entropy_block_combiner(x, params)
    _assert_keys_values_close(actual, expected)


def test_sample_entropy_constant(constant_series):
    x = constant_series[50]
    expected = _sample_entropy_expected(x, SAMPEN_PARAMS)
    actual = sample_entropy_block_combiner(x, SAMPEN_PARAMS)
    _assert_keys_values_close(actual, expected)


def test_sample_entropy_nan(nan_series):
    x = nan_series[50]
    expected = _sample_entropy_expected(x, SAMPEN_PARAMS)
    actual = sample_entropy_block_combiner(x, SAMPEN_PARAMS)
    _assert_keys_values_close(actual, expected)


def test_sample_entropy_short():
    x = np.array([1.0, 2.0, 3.0])
    expected = _sample_entropy_expected(x, SAMPEN_PARAMS)
    actual = sample_entropy_block_combiner(x, SAMPEN_PARAMS)
    _assert_keys_values_close(actual, expected)


def test_sample_entropy_smooth(smooth_series):
    for name, length, series in smooth_series:
        params = _params_for_length(
            length, SAMPEN_PARAMS, SAMPEN_PARAMS_MEDIUM, SAMPEN_PARAMS_LARGE
        )
        expected = _sample_entropy_expected(series, params)
        actual = sample_entropy_block_combiner(series, params)
        _assert_keys_values_close(
            actual, expected, context=f"sample_entropy {name} n={length}"
        )


@pytest.mark.parametrize("length", LENGTHS)
def test_change_quantiles_random(random_series, length):
    x = random_series[length]
    expected = _change_quantiles_expected(x, CHANGE_QUANTILES_PARAMS)
    actual = change_quantiles_qcut_exact_combiner(x, CHANGE_QUANTILES_PARAMS)
    _assert_keys_values_close(actual, expected)


def test_change_quantiles_discrete():
    x = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5], dtype=float)
    expected = _change_quantiles_expected(x, CHANGE_QUANTILES_PARAMS)
    actual = change_quantiles_qcut_exact_combiner(x, CHANGE_QUANTILES_PARAMS)
    _assert_keys_values_close(actual, expected)


def test_change_quantiles_constant(constant_series):
    x = constant_series[50]
    expected = _change_quantiles_expected(x, CHANGE_QUANTILES_PARAMS)
    actual = change_quantiles_qcut_exact_combiner(x, CHANGE_QUANTILES_PARAMS)
    _assert_keys_values_close(actual, expected)


def test_change_quantiles_nan(nan_series):
    x = nan_series[50]
    expected = _change_quantiles_expected(x, CHANGE_QUANTILES_PARAMS)
    actual = change_quantiles_qcut_exact_combiner(x, CHANGE_QUANTILES_PARAMS)
    _assert_keys_values_close(actual, expected)


def test_change_quantiles_small_n():
    x = np.array([1.0])
    expected = _change_quantiles_expected(x, CHANGE_QUANTILES_PARAMS)
    actual = change_quantiles_qcut_exact_combiner(x, CHANGE_QUANTILES_PARAMS)
    _assert_keys_values_close(actual, expected)


def test_change_quantiles_smooth(smooth_series):
    for name, length, series in smooth_series:
        expected = _change_quantiles_expected(series, CHANGE_QUANTILES_PARAMS)
        actual = change_quantiles_qcut_exact_combiner(series, CHANGE_QUANTILES_PARAMS)
        _assert_keys_values_close(
            actual, expected, context=f"change_quantiles {name} n={length}"
        )


@pytest.mark.parametrize("length", LENGTHS)
def test_augmented_dickey_fuller_random(random_series, length):
    x = random_series[length]
    expected = _augmented_dickey_fuller_original(x, ADF_PARAMS)
    actual = augmented_dickey_fuller(x, ADF_PARAMS)
    _assert_keys_values_close(actual, expected)


def test_augmented_dickey_fuller_constant(constant_series):
    x = constant_series[50]
    expected = _augmented_dickey_fuller_original(x, ADF_PARAMS)
    actual = augmented_dickey_fuller(x, ADF_PARAMS)
    _assert_keys_values_close(actual, expected)


def test_augmented_dickey_fuller_smooth(smooth_series):
    for name, length, series in smooth_series:
        expected = _augmented_dickey_fuller_original(series, ADF_PARAMS)
        actual = augmented_dickey_fuller(series, ADF_PARAMS)
        _assert_keys_values_close(actual, expected, context=f"adf {name} n={length}")

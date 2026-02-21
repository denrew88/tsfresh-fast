# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)

import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

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


def _time_call(fn, *args, repeats=1, **kwargs):
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn(*args, **kwargs)
        best = min(best, time.perf_counter() - start)
    return best


def _assert_parity(actual_items, expected_items, name):
    actual = dict(actual_items)
    expected = dict(expected_items)
    if set(actual.keys()) != set(expected.keys()):
        raise AssertionError(f"{name}: key mismatch")
    for key in expected:
        if not np.allclose(
            actual[key],
            expected[key],
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        ):
            raise AssertionError(f"{name}: value mismatch for {key}")


def bench():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(10000)

    approx_params = [{"m": 2, "r": r} for r in (0.1, 0.2, 0.3)]
    sampen_params = [{"m": 2, "r": r} for r in (0.1, 0.2, 0.3)]
    change_params = [
        {"ql": ql, "qh": qh, "isabs": isabs, "f_agg": f_agg}
        for (ql, qh) in ((0.1, 0.9), (0.2, 0.8), (0.25, 0.75))
        for isabs in (True, False)
        for f_agg in ("mean", "var", "std", "median")
    ]
    adf_params = [
        {"autolag": autolag, "attr": attr}
        for autolag in ("AIC", "BIC", "t-stat", None)
        for attr in ("teststat", "pvalue", "usedlag")
    ]

    approx_expected = [
        (
            f"m_{config['m']}__r_{config['r']:g}",
            _approximate_entropy_original(x, config["m"], config["r"]),
        )
        for config in approx_params
    ]
    approx_actual = approximate_entropy_block_combiner(x, approx_params)
    _assert_parity(approx_actual, approx_expected, "approximate_entropy")

    sampen_expected = [
        (
            f"m_{config['m']}__r_{config['r']:g}",
            _sample_entropy_original(x, config["m"], config["r"]),
        )
        for config in sampen_params
    ]
    sampen_actual = sample_entropy_block_combiner(x, sampen_params)
    _assert_parity(sampen_actual, sampen_expected, "sample_entropy")

    change_expected = [
        (
            f"f_agg_{config['f_agg']}__isabs_{config['isabs']}__qh_{config['qh']}__ql_{config['ql']}",
            _change_quantiles_original(
                x, config["ql"], config["qh"], config["isabs"], config["f_agg"]
            ),
        )
        for config in change_params
    ]
    change_actual = change_quantiles_qcut_exact_combiner(x, change_params)
    _assert_parity(change_actual, change_expected, "change_quantiles")

    adf_expected = _augmented_dickey_fuller_original(x, adf_params)
    adf_actual = augmented_dickey_fuller(x, adf_params)
    _assert_parity(adf_actual, adf_expected, "augmented_dickey_fuller")

    repeats = int(os.getenv("BENCH_REPEATS", "1"))

    approx_old_t = _time_call(
        lambda: [  # noqa: E731
            _approximate_entropy_original(x, config["m"], config["r"])
            for config in approx_params
        ],
        repeats=repeats,
    )
    approx_new_t = _time_call(
        approximate_entropy_block_combiner, x, approx_params, repeats=repeats
    )

    sampen_old_t = _time_call(
        lambda: [  # noqa: E731
            _sample_entropy_original(x, config["m"], config["r"])
            for config in sampen_params
        ],
        repeats=repeats,
    )
    sampen_new_t = _time_call(
        sample_entropy_block_combiner, x, sampen_params, repeats=repeats
    )

    change_old_t = _time_call(
        lambda: [  # noqa: E731
            _change_quantiles_original(
                x, config["ql"], config["qh"], config["isabs"], config["f_agg"]
            )
            for config in change_params
        ],
        repeats=repeats,
    )
    change_new_t = _time_call(
        change_quantiles_qcut_exact_combiner, x, change_params, repeats=repeats
    )

    adf_old_t = _time_call(
        _augmented_dickey_fuller_original, x, adf_params, repeats=repeats
    )
    adf_new_t = _time_call(augmented_dickey_fuller, x, adf_params, repeats=repeats)

    print(f"Benchmarks (best of {repeats}):")
    print(
        "approximate_entropy: "
        f"{approx_old_t:.4f}s -> {approx_new_t:.4f}s "
        f"(x{approx_old_t/approx_new_t:.2f})"
    )
    print(
        "sample_entropy:      "
        f"{sampen_old_t:.4f}s -> {sampen_new_t:.4f}s "
        f"(x{sampen_old_t/sampen_new_t:.2f})"
    )
    print(
        "change_quantiles:    "
        f"{change_old_t:.4f}s -> {change_new_t:.4f}s "
        f"(x{change_old_t/change_new_t:.2f})"
    )
    print(
        f"ADF (best of {repeats}):      "
        f"{adf_old_t:.4f}s -> {adf_new_t:.4f}s "
        f"(x{adf_old_t/adf_new_t:.2f})"
    )


if __name__ == "__main__":
    bench()

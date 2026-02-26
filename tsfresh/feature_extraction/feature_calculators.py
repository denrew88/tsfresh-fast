# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016
"""
This module contains the feature calculators that take time series as input and calculate the values of the feature.
There are two types of features:

1. feature calculators which calculate a single number (simple)
2. feature calculators which calculate a bunch of features for a list of parameters at once,
   to use e.g. cached results (combiner). They return a list of (key, value) pairs for each input parameter.

They are specified using the "fctype" parameter of each feature calculator, which is added using the
set_property function. Only functions in this python module, which have a parameter called  "fctype" are
seen by tsfresh as a feature calculator. Others will not be calculated.

Feature calculators of type combiner should return the concatenated parameters sorted
alphabetically ascending.
"""

import functools
import itertools
import os
import warnings
from builtins import range
from collections import defaultdict
from typing import Literal, Optional

import numpy as np
import pandas as pd
import pywt
import stumpy
from numpy.linalg import LinAlgError
from scipy.signal import find_peaks_cwt, welch
from scipy.signal import _peak_finding as _scipy_peak_finding
from scipy.stats import linregress, scoreatpercentile
from statsmodels.tools.sm_exceptions import MissingDataError
from statsmodels.tools.tools import pinv_extended
from statsmodels.tsa.ar_model import AutoReg

try:
    import matrixprofile as mp
    from matrixprofile.exceptions import NoSolutionPossible
except ImportError:
    mp = None


from tsfresh.utilities.string_manipulation import convert_to_output_format

with warnings.catch_warnings():
    # Ignore warnings of the patsy package
    warnings.simplefilter("ignore", DeprecationWarning)


from statsmodels.tsa.stattools import acf, adfuller, mackinnonp, pacf
from statsmodels.tsa.tsatools import add_trend, lagmat

# todo: make sure '_' works in parameter names in all cases, add a warning if not


def _roll(a, shift):
    """
    Roll 1D array elements. Improves the performance of numpy.roll() by reducing the overhead introduced from the
    flexibility of the numpy.roll() method such as the support for rolling over multiple dimensions.

    Elements that roll beyond the last position are re-introduced at the beginning. Similarly, elements that roll
    back beyond the first position are re-introduced at the end (with negative shift).

    Examples
    --------
    >>> x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> _roll(x, shift=2)
    >>> array([8, 9, 0, 1, 2, 3, 4, 5, 6, 7])

    >>> x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> _roll(x, shift=-2)
    >>> array([2, 3, 4, 5, 6, 7, 8, 9, 0, 1])

    >>> x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> _roll(x, shift=12)
    >>> array([8, 9, 0, 1, 2, 3, 4, 5, 6, 7])

    Benchmark
    ---------
    >>> x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> %timeit _roll(x, shift=2)
    >>> 1.89 µs ± 341 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

    >>> x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> %timeit np.roll(x, shift=2)
    >>> 11.4 µs ± 776 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

    :param a: the input array
    :type a: array_like
    :param shift: the number of places by which elements are shifted
    :type shift: int

    :return: shifted array with the same shape as a
    :return type: ndarray
    """
    if not isinstance(a, np.ndarray):
        a = np.asarray(a)
    idx = shift % len(a)
    return np.concatenate([a[-idx:], a[:-idx]])


def _get_length_sequences_where(x):
    """
    This method calculates the length of all sub-sequences where the array x is either True or 1.

    Examples
    --------
    >>> x = [0,1,0,0,1,1,1,0,0,1,0,1,1]
    >>> _get_length_sequences_where(x)
    >>> [1, 3, 1, 2]

    >>> x = [0,True,0,0,True,True,True,0,0,True,0,True,True]
    >>> _get_length_sequences_where(x)
    >>> [1, 3, 1, 2]

    >>> x = [0,True,0,0,1,True,1,0,0,True,0,1,True]
    >>> _get_length_sequences_where(x)
    >>> [1, 3, 1, 2]

    :param x: An iterable containing only 1, True, 0 and False values
    :return: A list with the length of all sub-sequences where the array is either True or False. If no ones or Trues
    contained, the list [0] is returned.
    """
    if len(x) == 0:
        return [0]

    res = [len(list(group)) for value, group in itertools.groupby(x) if value == 1]
    return res if len(res) > 0 else [0]


def _estimate_friedrich_coefficients(x, m, r):
    """
    Coefficients of polynomial :math:`h(x)`, which has been fitted to
    the deterministic dynamics of Langevin model
    .. math::
        \\dot{x}(t) = h(x(t)) + \\mathcal{N}(0,R)

    As described by

        Friedrich et al. (2000): Physics Letters A 271, p. 217-222
        *Extracting model equations from experimental data*

    For short time-series this method is highly dependent on the parameters.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param m: order of polynomial to fit for estimating fixed points of dynamics
    :type m: int
    :param r: number of quantiles to use for averaging
    :type r: float

    :return: coefficients of polynomial of deterministic dynamics
    :return type: ndarray
    """
    assert m > 0, f"Order of polynomial need to be positive integer, found {m}"

    df = pd.DataFrame({"signal": x[:-1], "delta": np.diff(x)})
    try:
        df["quantiles"] = pd.qcut(df.signal, r)
    except (ValueError, IndexError):
        return [np.nan] * (m + 1)

    quantiles = df.groupby("quantiles")

    result = pd.DataFrame(
        {"x_mean": quantiles.signal.mean(), "y_mean": quantiles.delta.mean()}
    )
    result.dropna(inplace=True)

    try:
        return np.polyfit(result.x_mean, result.y_mean, deg=m)
    except (np.linalg.LinAlgError, ValueError):
        return [np.nan] * (m + 1)


def _aggregate_on_chunks(x, f_agg, chunk_len):
    """
    Takes the time series x and constructs a lower sampled version of it by applying the aggregation function f_agg on
    consecutive chunks of length chunk_len

    :param x: the time series to calculate the aggregation of
    :type x: numpy.ndarray
    :param f_agg: The name of the aggregation function that should be an attribute of the pandas.Series
    :type f_agg: str
    :param chunk_len: The size of the chunks where to aggregate the time series
    :type chunk_len: int
    :return: A list of the aggregation function over the chunks
    :return type: list
    """
    return [
        getattr(x[i * chunk_len : (i + 1) * chunk_len], f_agg)()
        for i in range(int(np.ceil(len(x) / chunk_len)))
    ]


def _into_subchunks(x, subchunk_length, every_n=1):
    """
    Split the time series x into subwindows of length "subchunk_length", starting every "every_n".

    For example, the input data if [0, 1, 2, 3, 4, 5, 6] will be turned into a matrix

        0  2  4
        1  3  5
        2  4  6

    with the settings subchunk_length = 3 and every_n = 2
    """
    len_x = len(x)

    assert subchunk_length > 1
    assert every_n > 0

    # how often can we shift a window of size subchunk_length over the input?
    num_shifts = (len_x - subchunk_length) // every_n + 1
    shift_starts = every_n * np.arange(num_shifts)
    indices = np.arange(subchunk_length)

    indexer = np.expand_dims(indices, axis=0) + np.expand_dims(shift_starts, axis=1)
    return np.asarray(x)[indexer]


def set_property(key, value):
    """
    This method returns a decorator that sets the property key of the function to value
    """

    def decorate_func(func):
        setattr(func, key, value)
        if func.__doc__ and key == "fctype":
            func.__doc__ = (
                func.__doc__ + "\n\n    *This function is of type: " + value + "*\n"
            )
        return func

    return decorate_func


@set_property("fctype", "simple")
def variance_larger_than_standard_deviation(x):
    """
    Is variance higher than the standard deviation?

    Boolean variable denoting if the variance of x is greater than its standard deviation. Is equal to variance of x
    being larger than 1

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: bool
    """
    y = np.var(x)
    return y > np.sqrt(y)


@set_property("fctype", "simple")
def ratio_beyond_r_sigma(x, r):
    """
    Ratio of values that are more than r * std(x) (so r times sigma) away from the mean of x.

    :param x: the time series to calculate the feature of
    :type x: iterable
    :param r: the ratio to compare with
    :type r: float
    :return: the value of this feature
    :return type: float
    """
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.sum(np.abs(x - np.mean(x)) > r * np.std(x)) / x.size


@set_property("fctype", "simple")
def large_standard_deviation(x, r):
    """
    Does time series have *large* standard deviation?

    Boolean variable denoting if the standard dev of x is higher than 'r' times the range = difference between max and
    min of x. Hence it checks if

    .. math::

        std(x) > r * (max(X)-min(X))

    According to a rule of the thumb, the standard deviation should be a forth of the range of the values.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param r: the percentage of the range to compare with
    :type r: float
    :return: the value of this feature
    :return type: bool
    """
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.std(x) > (r * (np.max(x) - np.min(x)))


@set_property("fctype", "combiner")
def symmetry_looking(x, param):
    """
    Boolean variable denoting if the distribution of x *looks symmetric*. This is the case if

    .. math::

        | mean(X)-median(X)| < r * (max(X)-min(X))

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param param: contains dictionaries {"r": x} with x (float) is the percentage of the range to compare with
    :type param: list
    :return: the value of this feature
    :return type: List[Tuple[str, bool]]
    """
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    mean_median_difference = np.abs(np.mean(x) - np.median(x))
    max_min_difference = np.max(x) - np.min(x)
    return [
        (f"r_{r['r']}", mean_median_difference < (r["r"] * max_min_difference))
        for r in param
    ]


@set_property("fctype", "simple")
def has_duplicate_max(x):
    """
    Checks if the maximum value of x is observed more than once

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: bool
    """
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.sum(x == np.max(x)) >= 2


@set_property("fctype", "simple")
def has_duplicate_min(x):
    """
    Checks if the minimal value of x is observed more than once

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: bool
    """
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.sum(x == np.min(x)) >= 2


@set_property("fctype", "simple")
def has_duplicate(x):
    """
    Checks if any value in x occurs more than once

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: bool
    """
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return x.size != np.unique(x).size


@set_property("fctype", "simple")
@set_property("minimal", True)
def sum_values(x):
    """
    Calculates the sum over the time series values

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    if len(x) == 0:
        return 0

    return np.sum(x)


@set_property("fctype", "combiner")
def agg_autocorrelation(x, param):
    """
    Descriptive statistics on the autocorrelation of the time series.

    Calculates the value of an aggregation function :math:`f_{agg}` (e.g. the variance or the mean) over the
    autocorrelation :math:`R(l)` for different lags. The autocorrelation :math:`R(l)` for lag :math:`l` is defined as

    .. math::

        R(l) = \\frac{1}{(n-l)\\sigma^2} \\sum_{t=1}^{n-l}(X_{t}-\\mu )(X_{t+l}-\\mu)

    where :math:`X_i` are the values of the time series, :math:`n` its length. Finally, :math:`\\sigma^2` and
    :math:`\\mu` are estimators for its variance and mean
    (See `Estimation of the Autocorrelation function <http://en.wikipedia.org/wiki/Autocorrelation#Estimation>`_).

    The :math:`R(l)` for different lags :math:`l` form a vector. This feature calculator applies the aggregation
    function :math:`f_{agg}` to this vector and returns

    .. math::

        f_{agg} \\left( R(1), \\ldots, R(m)\\right) \\quad \\text{for} \\quad m = max(n, maxlag).

    Here :math:`maxlag` is the second parameter passed to this function.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param param: contains dictionaries {"f_agg": x, "maxlag", n} with x str, the name of a numpy function
                  (e.g. "mean", "var", "std", "median"), its the name of the aggregator function that is applied to the
                  autocorrelations. Further, n is an int and the maximal number of lags to consider.
    :type param: list
    :return: the value of this feature
    :return type: List[Tuple[str, float]]
    """
    # if the time series is longer than the following threshold, we use fft to calculate the acf
    THRESHOLD_TO_USE_FFT = 1250
    var = np.var(x)
    n = len(x)
    max_maxlag = max([config["maxlag"] for config in param])

    if np.abs(var) < 10**-10 or n == 1:
        a = [0] * len(x)
    else:
        a = acf(x, adjusted=True, fft=n > THRESHOLD_TO_USE_FFT, nlags=max_maxlag)[1:]
    return [
        (
            f'f_agg_"{config["f_agg"]}"__maxlag_{config["maxlag"]}',
            getattr(np, config["f_agg"])(a[: int(config["maxlag"])]),
        )
        for config in param
    ]


@set_property("fctype", "combiner")
def partial_autocorrelation(x, param):
    """
    Calculates the value of the partial autocorrelation function at the given lag.

    The lag `k` partial autocorrelation of a time series :math:`\\lbrace x_t, t = 1 \\ldots T \\rbrace` equals the
    partial correlation of :math:`x_t` and :math:`x_{t-k}`, adjusted for the intermediate variables
    :math:`\\lbrace x_{t-1}, \\ldots, x_{t-k+1} \\rbrace` ([1]).

    Following [2], it can be defined as

    .. math::

        \\alpha_k = \\frac{ Cov(x_t, x_{t-k} | x_{t-1}, \\ldots, x_{t-k+1})}
        {\\sqrt{ Var(x_t | x_{t-1}, \\ldots, x_{t-k+1}) Var(x_{t-k} | x_{t-1}, \\ldots, x_{t-k+1} )}}

    with (a) :math:`x_t = f(x_{t-1}, \\ldots, x_{t-k+1})` and (b) :math:`x_{t-k} = f(x_{t-1}, \\ldots, x_{t-k+1})`
    being AR(k-1) models that can be fitted by OLS. Be aware that in (a), the regression is done on past values to
    predict :math:`x_t` whereas in (b), future values are used to calculate the past value :math:`x_{t-k}`.
    It is said in [1] that "for an AR(p), the partial autocorrelations [ :math:`\\alpha_k` ] will be nonzero for `k<=p`
    and zero for `k>p`."
    With this property, it is used to determine the lag of an AR-Process.

    .. rubric:: References

    |  [1] Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015).
    |  Time series analysis: forecasting and control. John Wiley & Sons.
    |  [2] https://onlinecourses.science.psu.edu/stat510/node/62

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param param: contains dictionaries {"lag": val} with int val indicating the lag to be returned
    :type param: list
    :return: the value of this feature
    :return type: List[Tuple[str, float]]
    """
    # Check the difference between demanded lags by param and possible lags to calculate (depends on len(x))
    max_demanded_lag = max([lag["lag"] for lag in param])
    n = len(x)

    # Check if list is too short to make calculations
    if n <= 1:
        pacf_coeffs = [np.nan] * (max_demanded_lag + 1)
    else:
        # https://github.com/statsmodels/statsmodels/pull/6846
        # PACF limits lag length to 50% of sample size.
        if max_demanded_lag >= n // 2:
            max_lag = n // 2 - 1
        else:
            max_lag = max_demanded_lag
        if max_lag > 0:
            pacf_coeffs = list(pacf(x, method="ld", nlags=max_lag))
            pacf_coeffs = pacf_coeffs + [np.nan] * max(0, (max_demanded_lag - max_lag))
        else:
            pacf_coeffs = [np.nan] * (max_demanded_lag + 1)

    return [(f"lag_{lag['lag']}", pacf_coeffs[lag["lag"]]) for lag in param]


def _augmented_dickey_fuller_original(x, param):
    @functools.lru_cache()
    def compute_adf(autolag):
        try:
            return adfuller(x, autolag=autolag)
        except LinAlgError:
            return np.nan, np.nan, np.nan
        except ValueError:  # occurs if sample size is too small
            return np.nan, np.nan, np.nan
        except MissingDataError:  # is thrown for e.g. inf or nan in the data
            return np.nan, np.nan, np.nan

    res = []
    for config in param:
        autolag = config.get("autolag", "AIC")

        adf = compute_adf(autolag)
        index = f'attr_"{config["attr"]}"__autolag_"{autolag}"'

        if config["attr"] == "teststat":
            res.append((index, adf[0]))
        elif config["attr"] == "pvalue":
            res.append((index, adf[1]))
        elif config["attr"] == "usedlag":
            res.append((index, adf[2]))
        else:
            res.append((index, np.nan))
    return res


_ADF_SOLVER_ENV = "TSFRESH_ADF_SOLVER"
_ADF_SOLVER_DEFAULT = "pinv"


def _get_adf_solver():
    solver = os.environ.get(_ADF_SOLVER_ENV, _ADF_SOLVER_DEFAULT).strip().lower()
    if solver in ("original", "orig", "reference", "ref"):
        return "original"
    if solver in ("pinv", "normal_eq", "ne", "xtx"):
        return "normal_eq" if solver in ("normal_eq", "ne", "xtx") else "pinv"
    return _ADF_SOLVER_DEFAULT


def _adf_ols_from_sufficient_stats(
    xtx: np.ndarray,
    xty: np.ndarray,
    yty: float,
    nobs: int,
    rcond_x: float = 1e-15,
    exog: Optional[np.ndarray] = None,
    endog: Optional[np.ndarray] = None,
):
    xtx = np.asarray(xtx, dtype=float)
    xty = np.asarray(xty, dtype=float)
    solver = _get_adf_solver()
    if solver == "original":
        solver = "pinv"

    if solver != "pinv" or exog is None or endog is None:
        eigvals = np.linalg.eigvalsh(xtx)
        max_eig = np.max(eigvals)
        rcond_xtx = rcond_x * rcond_x
        threshold = rcond_xtx * max_eig

        if not np.isfinite(max_eig) or max_eig <= 0 or np.min(eigvals) <= threshold:
            eigvals_full, eigvecs = np.linalg.eigh(xtx)
            inv_eigs = np.where(eigvals_full > threshold, 1.0 / eigvals_full, 0.0)
            inv_xtx = (eigvecs * inv_eigs) @ eigvecs.T
            rank = int(np.sum(eigvals_full > threshold))
            beta = inv_xtx @ xty
        else:
            beta = np.linalg.solve(xtx, xty)
            inv_xtx = np.linalg.inv(xtx)
            rank = xtx.shape[0]

        if exog is not None and endog is not None:
            resid = endog - exog @ beta
            sse = float(np.dot(resid, resid))
        else:
            sse = float(yty - np.dot(beta, xty))
        if sse < 0:
            sse = 0.0
        df_resid = nobs - rank
        sigma2 = sse / df_resid
        se = np.sqrt(np.diag(inv_xtx) * sigma2)
        tvalues = beta / se
        return beta, tvalues, sse, df_resid, rank

    exog = np.asarray(exog, dtype=float)
    endog = np.asarray(endog, dtype=float)
    pinv, singular_values = pinv_extended(exog, rcond=rcond_x)
    normalized_cov_params = pinv @ pinv.T
    rank = (
        np.linalg.matrix_rank(np.diag(singular_values)) if singular_values.size else 0
    )

    beta = pinv @ endog

    resid = endog - exog @ beta
    sse = float(np.dot(resid, resid))
    df_resid = nobs - rank
    sigma2 = sse / df_resid
    se = np.sqrt(np.diag(normalized_cov_params) * sigma2)
    tvalues = beta / se
    return beta, tvalues, sse, df_resid, rank


def _adf_pinv_prefix_solver(
    exog_full: np.ndarray,
    endog: np.ndarray,
    rcond_x: float = 1e-15,
    fallback_factor: float = 1e6,
):
    exog_full = np.asarray(exog_full, dtype=float)
    endog = np.asarray(endog, dtype=float)
    u_full, s_full, vt_full = np.linalg.svd(exog_full, full_matrices=False)
    z_full = u_full.T @ endog
    nobs = endog.shape[0]
    cache = {}

    def solve(k: int):
        if k in cache:
            return cache[k]

        kp = s_full[:, None] * vt_full[:, :k]
        u_p, s_p, vt_p = np.linalg.svd(kp, full_matrices=False)
        if s_p.size:
            cutoff = rcond_x * np.maximum.reduce(s_p)
            if np.min(s_p) <= cutoff * fallback_factor:
                exog_k = exog_full[:, :k]
                pinv, singular_values = pinv_extended(exog_k, rcond=rcond_x)
                normalized_cov_params = pinv @ pinv.T
                rank = (
                    np.linalg.matrix_rank(np.diag(singular_values))
                    if singular_values.size
                    else 0
                )
                beta = pinv @ endog
                resid = endog - exog_k @ beta
                sse = float(np.dot(resid, resid))
                df_resid = nobs - rank
                sigma2 = sse / df_resid
                se = np.sqrt(np.diag(normalized_cov_params) * sigma2)
                tvalues = beta / se
                result = beta, tvalues, sse, df_resid, rank
                cache[k] = result
                return result

            s_inv = np.where(s_p > cutoff, 1.0 / s_p, 0.0)
        else:
            s_inv = s_p

        z_p = u_p.T @ z_full
        beta = vt_p.T @ (s_inv * z_p)

        resid = endog - exog_full[:, :k] @ beta
        sse = float(np.dot(resid, resid))
        rank = np.linalg.matrix_rank(np.diag(s_p)) if s_p.size else 0
        df_resid = nobs - rank
        sigma2 = sse / df_resid

        if s_p.size:
            var_beta = (vt_p.T**2) @ (s_inv**2)
        else:
            var_beta = np.zeros(k, dtype=float)
        se = np.sqrt(var_beta * sigma2)
        tvalues = beta / se

        result = beta, tvalues, sse, df_resid, rank
        cache[k] = result
        return result

    return solve


def _adf_autolag_from_sufficient_stats(
    xtx_full: np.ndarray,
    xty_full: np.ndarray,
    yty: float,
    nobs: int,
    startlag: int,
    maxlag: int,
    method: str,
    exog_full: np.ndarray,
    endog: np.ndarray,
    rcond_x: float = 1e-15,
):
    prefix_solver = None
    if _get_adf_solver() == "pinv":
        prefix_solver = _adf_pinv_prefix_solver(
            exog_full, endog, rcond_x=rcond_x
        )

    k_start = startlag
    k_end = startlag + maxlag
    if method in ("aic", "bic"):
        icbest = np.inf
        best_k = k_start
        for k in range(k_start, k_end + 1):
            if prefix_solver is not None:
                _, _, sse, _, rank = prefix_solver(k)
            else:
                xtx_k = xtx_full[:k, :k]
                xty_k = xty_full[:k]
                exog_k = exog_full[:, :k]
                _, _, sse, _, rank = _adf_ols_from_sufficient_stats(
                    xtx_k, xty_k, yty, nobs, exog=exog_k, endog=endog
                )
            llf = -0.5 * nobs * (
                np.log(2.0 * np.pi) + np.log(sse / nobs) + 1.0
            )
            k_params = rank
            if method == "aic":
                ic = -2.0 * llf + 2.0 * k_params
            else:
                ic = -2.0 * llf + np.log(nobs) * k_params
            if ic < icbest:
                icbest = ic
                best_k = k
        return icbest, best_k

    if method == "t-stat":
        stop = 1.6448536269514722
        tvalues_last = []
        for k in range(k_start, k_end + 1):
            if prefix_solver is not None:
                beta, tvalues, _, _, _ = prefix_solver(k)
            else:
                xtx_k = xtx_full[:k, :k]
                xty_k = xty_full[:k]
                exog_k = exog_full[:, :k]
                beta, tvalues, _, _, _ = _adf_ols_from_sufficient_stats(
                    xtx_k, xty_k, yty, nobs, exog=exog_k, endog=endog
                )
            t_last = tvalues[-1]
            tvalues_last.append(t_last)

        best_k = k_end
        icbest = 0.0
        for k in range(k_end, k_start - 1, -1):
            t_last = tvalues_last[k - k_start]
            icbest = np.abs(t_last)
            best_k = k
            if np.abs(t_last) >= stop:
                break
        return icbest, best_k

    raise ValueError(f"Information Criterion {method} not understood.")


def _adf_fast(x, autolag):
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all():
        raise MissingDataError("exog contains inf or nans")

    if x.max() == x.min():
        raise ValueError("Invalid input, x is constant")

    regression = "c"
    nobs = x.shape[0]
    ntrend = len(regression) if regression != "n" else 0

    maxlag = int(np.ceil(12.0 * np.power(nobs / 100.0, 1 / 4.0)))
    maxlag = min(nobs // 2 - ntrend - 1, maxlag)
    if maxlag < 0:
        raise ValueError(
            "sample size is too short to use selected regression component"
        )

    xdiff = np.diff(x)
    xdall = lagmat(xdiff[:, None], maxlag, trim="both", original="in")
    nobs_used = xdall.shape[0]
    xdall[:, 0] = x[-nobs_used - 1 : -1]
    xdshort = xdiff[-nobs_used:]

    method = None
    if autolag is not None:
        if isinstance(autolag, str):
            method = autolag.lower()
        else:
            raise ValueError("autolag must be a string or None")

    usedlag = maxlag
    if method:
        if regression != "n":
            full_rhs = add_trend(xdall, regression, prepend=True)
        else:
            full_rhs = xdall
        startlag = full_rhs.shape[1] - xdall.shape[1] + 1
        xtx_full = full_rhs.T @ full_rhs
        xty_full = full_rhs.T @ xdshort
        yty = float(np.dot(xdshort, xdshort))

        icbest, best_k = _adf_autolag_from_sufficient_stats(
            xtx_full,
            xty_full,
            yty,
            nobs_used,
            startlag,
            maxlag,
            method,
            exog_full=full_rhs,
            endog=xdshort,
        )
        usedlag = best_k - startlag

        xdall = lagmat(xdiff[:, None], usedlag, trim="both", original="in")
        nobs_used = xdall.shape[0]
        xdall[:, 0] = x[-nobs_used - 1 : -1]
        xdshort = xdiff[-nobs_used:]

    if regression != "n":
        exog = add_trend(xdall[:, : usedlag + 1], regression)
    else:
        exog = xdall[:, : usedlag + 1]

    xtx = exog.T @ exog
    xty = exog.T @ xdshort
    yty = float(np.dot(xdshort, xdshort))
    _, tvalues, _, _, _ = _adf_ols_from_sufficient_stats(
        xtx, xty, yty, xdshort.shape[0], exog=exog, endog=xdshort
    )
    adfstat = tvalues[0]
    pvalue = mackinnonp(adfstat, regression=regression, N=1)
    return adfstat, pvalue, usedlag


@set_property("fctype", "combiner")
def augmented_dickey_fuller(x, param):
    """
    Does the time series have a unit root?

    The Augmented Dickey-Fuller test is a hypothesis test which checks whether a unit root is present in a time
    series sample. This feature calculator returns the value of the respective test statistic.

    See the statsmodels implementation for references and more details.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param param: contains dictionaries {"attr": x, "autolag": y} with x str, either "teststat", "pvalue" or "usedlag"
                  and with y str, either of "AIC", "BIC", "t-stats" or None (See the documentation of adfuller() in
                  statsmodels).
    :type param: list
    :return: the value of this feature
    :return type: List[Tuple[str, float]]
    """
    solver = _get_adf_solver()
    if solver == "original":
        return _augmented_dickey_fuller_original(x, param)

    res = []
    cache = {}
    for config in param:
        autolag = config.get("autolag", "AIC")
        if autolag not in cache:
            try:
                cache[autolag] = _adf_fast(x, autolag)
            except (LinAlgError, ValueError, MissingDataError):
                cache[autolag] = (np.nan, np.nan, np.nan)

        adf = cache[autolag]
        index = f'attr_"{config["attr"]}"__autolag_"{autolag}"'

        if config["attr"] == "teststat":
            res.append((index, adf[0]))
        elif config["attr"] == "pvalue":
            res.append((index, adf[1]))
        elif config["attr"] == "usedlag":
            res.append((index, adf[2]))
        else:
            res.append((index, np.nan))

    return res


@set_property("fctype", "simple")
def abs_energy(x):
    """
    Returns the absolute energy of the time series which is the sum over the squared values

    .. math::

        E = \\sum_{i=1,\\ldots, n} x_i^2

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.dot(x, x)


@set_property("fctype", "simple")
def cid_ce(x, normalize):
    """
    This function calculator is an estimate for a time series complexity [1] (A more complex time series has more peaks,
    valleys etc.). It calculates the value of

    .. math::

        \\sqrt{ \\sum_{i=1}^{n-1} ( x_{i} - x_{i-1})^2 }

    .. rubric:: References

    |  [1] Batista, Gustavo EAPA, et al (2014).
    |  CID: an efficient complexity-invariant distance for time series.
    |  Data Mining and Knowledge Discovery 28.3 (2014): 634-669.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param normalize: should the time series be z-transformed?
    :type normalize: bool

    :return: the value of this feature
    :return type: float
    """
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    if normalize:
        s = np.std(x)
        if s != 0:
            x = (x - np.mean(x)) / s
        else:
            return 0.0

    x = np.diff(x)
    return np.sqrt(np.dot(x, x))


@set_property("fctype", "simple")
def mean_abs_change(x):
    """
    Average over first differences.

    Returns the mean over the absolute differences between subsequent time series values which is

    .. math::

        \\frac{1}{n-1} \\sum_{i=1,\\ldots, n-1} | x_{i+1} - x_{i}|


    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    return np.mean(np.abs(np.diff(x)))


@set_property("fctype", "simple")
def mean_change(x):
    """
    Average over time series differences.

    Returns the mean over the differences between subsequent time series values which is

    .. math::

        \\frac{1}{n-1} \\sum_{i=1,\\ldots, n-1}  x_{i+1} - x_{i} = \\frac{1}{n-1} (x_{n} - x_{1})

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    x = np.asarray(x)
    return (x[-1] - x[0]) / (len(x) - 1) if len(x) > 1 else np.nan


@set_property("fctype", "simple")
def mean_second_derivative_central(x):
    """
    Returns the mean value of a central approximation of the second derivative

    .. math::

        \\frac{1}{2(n-2)} \\sum_{i=1,\\ldots, n-1}  \\frac{1}{2} (x_{i+2} - 2 \\cdot x_{i+1} + x_i)

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    x = np.asarray(x)
    return (x[-1] - x[-2] - x[1] + x[0]) / (2 * (len(x) - 2)) if len(x) > 2 else np.nan


@set_property("fctype", "simple")
@set_property("minimal", True)
def median(x):
    """
    Returns the median of x

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    return np.median(x)


@set_property("fctype", "simple")
@set_property("minimal", True)
def mean(x):
    """
    Returns the mean of x

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    return np.mean(x)


@set_property("fctype", "simple")
@set_property("minimal", True)
def length(x):
    """
    Returns the length of x

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: int
    """
    return len(x)


@set_property("fctype", "simple")
@set_property("minimal", True)
def standard_deviation(x):
    """
    Returns the standard deviation of x

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    return np.std(x)


@set_property("fctype", "simple")
def variation_coefficient(x):
    """
    Returns the variation coefficient (standard error / mean, give relative value of variation around mean) of x.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    avg = np.mean(x)
    if avg == 0:
        return np.nan
    return np.std(x) / avg


@set_property("fctype", "simple")
@set_property("minimal", True)
def variance(x):
    """
    Returns the variance of x

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    return np.var(x)


@set_property("fctype", "simple")
@set_property("input", "pd.Series")
def skewness(x):
    """
    Returns the sample skewness of x (calculated with the adjusted Fisher-Pearson standardized
    moment coefficient G1).

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    return pd.Series.skew(x, skipna=False)


@set_property("fctype", "simple")
@set_property("input", "pd.Series")
def kurtosis(x):
    """
    Returns the kurtosis of x (calculated with the adjusted Fisher-Pearson standardized
    moment coefficient G2).

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    return pd.Series.kurtosis(x)


@set_property("fctype", "simple")
@set_property("minimal", True)
def root_mean_square(x):
    """
    Returns the root mean square (rms) of the time series.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    return np.sqrt(np.mean(np.square(x))) if len(x) > 0 else np.nan


@set_property("fctype", "simple")
def absolute_sum_of_changes(x):
    """
    Returns the sum over the absolute value of consecutive changes in the series x

    .. math::

        \\sum_{i=1, \\ldots, n-1} \\mid x_{i+1}- x_i \\mid

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    return np.sum(np.abs(np.diff(x)))


@set_property("fctype", "simple")
def longest_strike_below_mean(x):
    """
    Returns the length of the longest consecutive subsequence in x that is smaller than the mean of x

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.max(_get_length_sequences_where(x < np.mean(x))) if x.size > 0 else 0


@set_property("fctype", "simple")
def longest_strike_above_mean(x):
    """
    Returns the length of the longest consecutive subsequence in x that is bigger than the mean of x

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.max(_get_length_sequences_where(x > np.mean(x))) if x.size > 0 else 0


@set_property("fctype", "simple")
def count_above_mean(x):
    """
    Returns the number of values in x that are higher than the mean of x

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    m = np.mean(x)
    return np.where(x > m)[0].size


@set_property("fctype", "simple")
def count_below_mean(x):
    """
    Returns the number of values in x that are lower than the mean of x

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    m = np.mean(x)
    return np.where(x < m)[0].size


@set_property("fctype", "simple")
def last_location_of_maximum(x):
    """
    Returns the relative last location of the maximum value of x.
    The position is calculated relatively to the length of x.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    x = np.asarray(x)
    return 1.0 - np.argmax(x[::-1]) / len(x) if len(x) > 0 else np.nan


@set_property("fctype", "simple")
def first_location_of_maximum(x):
    """
    Returns the first location of the maximum value of x.
    The position is calculated relatively to the length of x.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.argmax(x) / len(x) if len(x) > 0 else np.nan


@set_property("fctype", "simple")
def last_location_of_minimum(x):
    """
    Returns the last location of the minimal value of x.
    The position is calculated relatively to the length of x.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    x = np.asarray(x)
    return 1.0 - np.argmin(x[::-1]) / len(x) if len(x) > 0 else np.nan


@set_property("fctype", "simple")
def first_location_of_minimum(x):
    """
    Returns the first location of the minimal value of x.
    The position is calculated relatively to the length of x.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.argmin(x) / len(x) if len(x) > 0 else np.nan


@set_property("fctype", "simple")
def percentage_of_reoccurring_values_to_all_values(x):
    """
    Returns the percentage of values that are present in the time series
    more than once.

        len(different values occurring more than once) / len(different values)

    This means the percentage is normalized to the number of unique values,
    in contrast to the percentage_of_reoccurring_datapoints_to_all_datapoints.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    if len(x) == 0:
        return np.nan

    _, counts = np.unique(x, return_counts=True)

    if counts.shape[0] == 0:
        return 0.0

    return np.sum(counts > 1) / float(counts.shape[0])


@set_property("fctype", "simple")
@set_property("input", "pd.Series")
def percentage_of_reoccurring_datapoints_to_all_datapoints(x):
    """
    Returns the percentage of non-unique data points. Non-unique means that they are
    contained another time in the time series again.

        # of data points occurring more than once / # of all data points

    This means the ratio is normalized to the number of data points in the time series,
    in contrast to the percentage_of_reoccurring_values_to_all_values.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    if len(x) == 0:
        return np.nan

    if not isinstance(x, pd.Series):
        x = pd.Series(x)

    value_counts = x.value_counts()
    reoccuring_values = value_counts[value_counts > 1].sum()

    if np.isnan(reoccuring_values):
        return 0.0

    return reoccuring_values / x.size


@set_property("fctype", "simple")
def sum_of_reoccurring_values(x):
    """
    Returns the sum of all values, that are present in the time series
    more than once.

    For example

        sum_of_reoccurring_values([2, 2, 2, 2, 1]) = 2

    as 2 is a reoccurring value, so it is summed up with all
    other reoccuring values (there is none), so the result is 2.

    This is in contrast to ``sum_of_reoccurring_data_points``,
    where each reoccuring value is only counted as often as
    it is present in the data.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    unique, counts = np.unique(x, return_counts=True)
    counts[counts < 2] = 0
    counts[counts > 1] = 1
    return np.sum(counts * unique)


@set_property("fctype", "simple")
def sum_of_reoccurring_data_points(x):
    """
    Returns the sum of all data points, that are present in the time series
    more than once.

    For example

        sum_of_reoccurring_data_points([2, 2, 2, 2, 1]) = 8

    as 2 is a reoccurring value, so all 2's are summed up.

    This is in contrast to ``sum_of_reoccurring_values``,
    where each reoccuring value is only counted once.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    unique, counts = np.unique(x, return_counts=True)
    counts[counts < 2] = 0
    return np.sum(counts * unique)


@set_property("fctype", "simple")
def ratio_value_number_to_time_series_length(x):
    """
    Returns a factor which is 1 if all values in the time series occur only once,
    and below one if this is not the case.
    In principle, it just returns

        # unique values / # values

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    if x.size == 0:
        return np.nan

    return np.unique(x).size / x.size


@set_property("fctype", "combiner")
def fft_coefficient(x, param):
    """
    Calculates the fourier coefficients of the one-dimensional discrete Fourier Transform for real input by fast
    fourier transformation algorithm

    .. math::
        A_k =  \\sum_{m=0}^{n-1} a_m \\exp \\left \\{ -2 \\pi i \\frac{m k}{n} \\right \\}, \\qquad k = 0,
        \\ldots , n-1.

    The resulting coefficients will be complex, this feature calculator can return the real part (attr=="real"),
    the imaginary part (attr=="imag), the absolute value (attr=""abs) and the angle in degrees (attr=="angle).

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param param: contains dictionaries {"coeff": x, "attr": s} with x int and x >= 0, s str and in ["real", "imag",
        "abs", "angle"]
    :type param: list
    :return: the different feature values
    :return type: Iterator[Tuple[str, float]]
    """

    assert (
        min((config["coeff"] for config in param)) >= 0
    ), "Coefficients must be positive or zero."
    assert {config["attr"] for config in param} <= {
        "imag",
        "real",
        "abs",
        "angle",
    }, 'Attribute must be "real", "imag", "angle" or "abs"'

    fft = np.fft.rfft(x)

    def complex_agg(x, agg: Literal["real", "imag", "angle", "abs"]):
        if agg == "real":
            return x.real
        elif agg == "imag":
            return x.imag
        elif agg == "abs":
            return np.abs(x)
        elif agg == "angle":
            return np.angle(x, deg=True)
        else:
            raise ValueError('`agg` must be "real", "imag", "angle" or "abs"')

    res = [
        complex_agg(fft[config["coeff"]], config["attr"])
        if config["coeff"] < len(fft)
        else np.nan
        for config in param
    ]
    index = [f'attr_"{config["attr"]}"__coeff_{config["coeff"]}' for config in param]
    return zip(index, res)


@set_property("fctype", "combiner")
def fft_aggregated(x, param):
    """
    Returns the spectral centroid (mean), variance, skew, and kurtosis of the absolute fourier transform spectrum.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param param: contains dictionaries {"aggtype": s} where s str and in ["centroid", "variance",
        "skew", "kurtosis"]
    :type param: list
    :return: the different feature values
    :return type: Iterator[Tuple[str, float]]
    """

    assert {config["aggtype"] for config in param} <= {
        "centroid",
        "variance",
        "skew",
        "kurtosis",
    }, 'Attribute must be "centroid", "variance", "skew", "kurtosis"'

    def get_moment(y, moment):
        """
        Returns the (non centered) moment of the distribution y:
        E[y**moment] = \\sum_i[index(y_i)^moment * y_i] / \\sum_i[y_i]

        :param y: the discrete distribution from which one wants to calculate the moment
        :type y: pandas.Series or np.array
        :param moment: the moment one wants to calcalate (choose 1,2,3, ... )
        :type moment: int
        :return: the moment requested
        :return type: float
        """
        return y.dot(np.arange(len(y), dtype=float) ** moment) / y.sum()

    def get_centroid(y):
        """
        :param y: the discrete distribution from which one wants to calculate the centroid
        :type y: pandas.Series or np.array
        :return: the centroid of distribution y (aka distribution mean, first moment)
        :return type: float
        """
        return get_moment(y, 1)

    def get_variance(y):
        """
        :param y: the discrete distribution from which one wants to calculate the variance
        :type y: pandas.Series or np.array
        :return: the variance of distribution y
        :return type: float
        """
        return get_moment(y, 2) - get_centroid(y) ** 2

    def get_skew(y):
        """
        Calculates the skew as the third standardized moment.
        Ref: https://en.wikipedia.org/wiki/Skewness#Definition

        :param y: the discrete distribution from which one wants to calculate the skew
        :type y: pandas.Series or np.array
        :return: the skew of distribution y
        :return type: float
        """

        var = get_variance(y)
        # In the limit of a dirac delta, skew should be 0 and variance 0.  However, in the discrete limit,
        # the skew blows up as variance --> 0, hence return nan when variance is smaller than a resolution of 0.5:
        if var < 0.5:
            return np.nan
        centroid = get_centroid(y)
        return (get_moment(y, 3) - 3 * centroid * var - centroid**3) / get_variance(
            y
        ) ** (1.5)

    def get_kurtosis(y):
        """
        Calculates the kurtosis as the fourth standardized moment.
        Ref: https://en.wikipedia.org/wiki/Kurtosis#Pearson_moments

        :param y: the discrete distribution from which one wants to calculate the kurtosis
        :type y: pandas.Series or np.array
        :return: the kurtosis of distribution y
        :return type: float
        """

        var = get_variance(y)
        # In the limit of a dirac delta, kurtosis should be 3 and variance 0.  However, in the discrete limit,
        # the kurtosis blows up as variance --> 0, hence return nan when variance is smaller than a resolution of 0.5:
        if var < 0.5:
            return np.nan
        centroid = get_centroid(y)
        return (
            get_moment(y, 4)
            - 4 * centroid * get_moment(y, 3)
            + 6 * get_moment(y, 2) * centroid**2
            - 3 * centroid
        ) / get_variance(y) ** 2

    calculation = {
        "centroid": get_centroid,
        "variance": get_variance,
        "skew": get_skew,
        "kurtosis": get_kurtosis,
    }

    fft_abs = np.abs(np.fft.rfft(x))

    res = [calculation[config["aggtype"]](fft_abs) for config in param]
    index = [f'aggtype_"{config["aggtype"]}"' for config in param]
    return zip(index, res)


@set_property("fctype", "simple")
def number_peaks(x, n):
    """
    Calculates the number of peaks of at least support n in the time series x. A peak of support n is defined as a
    subsequence of x where a value occurs, which is bigger than its n neighbours to the left and to the right.

    Hence in the sequence

    >>> x = [3, 0, 0, 4, 0, 0, 13]

    4 is a peak of support 1 and 2 because in the subsequences

    >>> [0, 4, 0]
    >>> [0, 0, 4, 0, 0]

    4 is still the highest value. Here, 4 is not a peak of support 3 because 13 is the 3th neighbour to the right of 4
    and its bigger than 4.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param n: the support of the peak
    :type n: int
    :return: the value of this feature
    :return type: float
    """
    x_reduced = x[n:-n]

    res = None
    for i in range(1, n + 1):
        result_first = x_reduced > _roll(x, i)[n:-n]

        if res is None:
            res = result_first
        else:
            res &= result_first

        res &= x_reduced > _roll(x, -i)[n:-n]
    return np.sum(res)


@set_property("fctype", "combiner")
def index_mass_quantile(x, param):
    """
    Calculates the relative index i of time series x where q% of the mass of x lies left of i.
    For example for q = 50% this feature calculator will return the mass center of the time series.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param param: contains dictionaries {"q": x} with x float
    :type param: list
    :return: the different feature values
    :return type: List[Tuple[str, float]]
    """

    x = np.asarray(x)
    abs_x = np.abs(x)
    s = np.sum(abs_x)

    if s == 0:
        # all values in x are zero or it has length 0
        return [(f"q_{config['q']}", np.nan) for config in param]

    # at least one value is not zero
    mass_centralized = np.cumsum(abs_x) / s
    return [
        (
            f"q_{config['q']}",
            (np.argmax(mass_centralized >= config["q"]) + 1) / len(x),
        )
        for config in param
    ]


def _ricker(points, a):
    """Custom implementation of the ricker wavelet, copied from scipy as scipy dropped it."""
    A = 2 / (np.sqrt(3 * a) * (np.pi**0.25))
    wsq = a**2
    vec = np.arange(0, points) - (points - 1.0) / 2
    xsq = vec**2
    mod = 1 - xsq / wsq
    gauss = np.exp(-xsq / (2 * wsq))
    total = A * mod * gauss
    return total


def _filter_ridge_lines_sparse_noise(
    cwt, ridge_lines, window_size=None, min_length=None, min_snr=1, noise_perc=10
):
    """
    A drop-in replacement for scipy.signal._peak_finding._filter_ridge_lines that
    computes the noise floor only at ridge peak columns.
    """
    num_points = cwt.shape[1]
    if min_length is None:
        min_length = np.ceil(cwt.shape[0] / 4)
    if window_size is None:
        window_size = np.ceil(num_points / 20)

    window_size = int(window_size)
    hf_window, odd = divmod(window_size, 2)

    if len(ridge_lines) == 0:
        return []

    row_one = cwt[0, :]
    peak_cols = np.array([line[1][0] for line in ridge_lines], dtype=np.int64)
    unique_cols, inv = np.unique(peak_cols, return_inverse=True)

    noises = np.empty(unique_cols.shape[0], dtype=row_one.dtype)
    for i, col in enumerate(unique_cols):
        window_start = max(col - hf_window, 0)
        window_end = min(col + hf_window + odd, num_points)
        noises[i] = scoreatpercentile(
            row_one[window_start:window_end], per=noise_perc
        )

    noises_for_line = noises[inv]
    out = []
    for line, noise_val in zip(ridge_lines, noises_for_line):
        if len(line[0]) < min_length:
            continue
        snr = abs(cwt[line[0][0], line[1][0]] / noise_val)
        if snr < min_snr:
            continue
        out.append(line)

    return out


def _find_peaks_cwt_sparse_noise(
    vector,
    widths,
    wavelet=None,
    max_distances=None,
    gap_thresh=None,
    min_length=None,
    min_snr=1,
    noise_perc=10,
    window_size=None,
):
    widths = np.atleast_1d(np.asarray(widths))

    if gap_thresh is None:
        gap_thresh = np.ceil(widths[0])
    if max_distances is None:
        max_distances = widths / 4.0
    if wavelet is None:
        wavelet = _ricker

    cwt_dat = _scipy_peak_finding._cwt(vector, wavelet, widths)
    ridge_lines = _scipy_peak_finding._identify_ridge_lines(
        cwt_dat, max_distances, gap_thresh
    )
    filtered = _filter_ridge_lines_sparse_noise(
        cwt_dat,
        ridge_lines,
        min_length=min_length,
        window_size=window_size,
        min_snr=min_snr,
        noise_perc=noise_perc,
    )
    max_locs = np.asarray([x[1][0] for x in filtered])
    max_locs.sort()

    return max_locs


def _number_cwt_peaks_original(x, n):
    return len(
        find_peaks_cwt(
            vector=x, widths=np.array(list(range(1, n + 1))), wavelet=_ricker
        )
    )


@set_property("fctype", "simple")
def number_cwt_peaks(x, n):
    """
    Number of different peaks in x.

    To estimamte the numbers of peaks, x is smoothed by a ricker wavelet for widths ranging from 1 to n. This feature
    calculator returns the number of peaks that occur at enough width scales and with sufficiently high
    Signal-to-Noise-Ratio (SNR)

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param n: maximum width to consider
    :type n: int
    :return: the value of this feature
    :return type: int
    """
    return len(
        _find_peaks_cwt_sparse_noise(
            vector=x, widths=np.array(list(range(1, n + 1))), wavelet=_ricker
        )
    )


@set_property("fctype", "combiner")
def linear_trend(x, param):
    """
    Calculate a linear least-squares regression for the values of the time series versus the sequence from 0 to
    length of the time series minus one.
    This feature assumes the signal to be uniformly sampled. It will not use the time stamps to fit the model.
    The parameters control which of the characteristics are returned.

    Possible extracted attributes are "pvalue", "rvalue", "intercept", "slope", "stderr", see the documentation of
    linregress for more information.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param param: contains dictionaries {"attr": x} with x an string, the attribute name of the regression model
    :type param: list
    :return: the different feature values
    :return type: List[Tuple[str, float]]
    """
    # todo: we could use the index of the DataFrame here
    lin_reg = linregress(range(len(x)), x)

    return [
        (f'attr_"{config["attr"]}"', getattr(lin_reg, config["attr"]))
        for config in param
    ]


@set_property("fctype", "combiner")
def cwt_coefficients(x, param):
    """
    Calculates a Continuous wavelet transform for the Ricker wavelet, also known as the "Mexican hat wavelet" which is
    defined by

    .. math::
        \\frac{2}{\\sqrt{3a} \\pi^{\\frac{1}{4}}} (1 - \\frac{x^2}{a^2}) exp(-\\frac{x^2}{2a^2})

    where :math:`a` is the width parameter of the wavelet function.

    This feature calculator takes three different parameter: widths, coeff and w. The feature calculator takes all the
    different widths arrays and then calculates the cwt one time for each different width array. Then the values for the
    different coefficient for coeff and width w are returned. (For each dic in param one feature is returned)

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param param: contains dictionaries {"widths":x, "coeff": y, "w": z} with x array of int and y,z int
    :type param: list
    :return: the different feature values
    :return type: Iterator[Tuple[str, float]]
    """

    calculated_cwt = {}
    res = []
    indices = []

    for parameter_combination in param:
        widths = tuple(parameter_combination["widths"])
        w = parameter_combination["w"]
        coeff = parameter_combination["coeff"]

        if widths not in calculated_cwt:
            calculated_cwt[widths], _ = pywt.cwt(x, scales=widths, wavelet="mexh")

        calculated_cwt_for_widths = calculated_cwt[widths]

        indices += [f"coeff_{coeff}__w_{w}__widths_{widths}"]

        i = widths.index(w)
        if calculated_cwt_for_widths.shape[1] <= coeff:
            res += [np.nan]
        else:
            res += [calculated_cwt_for_widths[i, coeff]]

    return zip(indices, res)


@set_property("fctype", "combiner")
def spkt_welch_density(x, param):
    """
    This feature calculator estimates the cross power spectral density of the time series x at different frequencies.
    To do so, the time series is first shifted from the time domain to the frequency domain.

    The feature calculators returns the power spectrum of the different frequencies.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param param: contains dictionaries {"coeff": x} with x int
    :type param: list
    :return: the different feature values
    :return type: Iterator[Tuple[str, float]]
    """

    max_length_per_segment = 256
    x = np.asarray(x)
    _, pxx = welch(x, nperseg=min(len(x), max_length_per_segment))
    coeff = [config["coeff"] for config in param]
    indices = [f"coeff_{i}" for i in coeff]

    if len(pxx) <= np.max(
        coeff
    ):  # There are fewer data points in the time series than requested coefficients

        # filter coefficients that are not contained in pxx
        reduced_coeff = [coefficient for coefficient in coeff if len(pxx) > coefficient]
        not_calculated_coefficients = [
            coefficient for coefficient in coeff if coefficient not in reduced_coeff
        ]

        # Fill up the rest of the requested coefficients with np.nans
        return zip(
            indices,
            list(pxx[reduced_coeff]) + [np.nan] * len(not_calculated_coefficients),
        )

    return zip(indices, pxx[coeff])


@set_property("fctype", "combiner")
def ar_coefficient(x, param):
    """
    This feature calculator fits the unconditional maximum likelihood
    of an autoregressive AR(k) process.
    The k parameter is the maximum lag of the process

    .. math::

        X_{t}=\\varphi_0 +\\sum _{{i=1}}^{k}\\varphi_{i}X_{{t-i}}+\\varepsilon_{t}

    For the configurations from param which should contain the maxlag "k" and such an AR process is calculated. Then
    the coefficients :math:`\\varphi_{i}` whose index :math:`i` contained from "coeff" are returned.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param param: contains dictionaries {"coeff": x, "k": y} with x,y int
    :type param: list
    :return x: the different feature values
    :return type: List[Tuple[str, float]]
    """
    calculated_ar_params = {}

    x_as_list = list(x)

    res = {}

    for parameter_combination in param:
        k = parameter_combination["k"]
        p = parameter_combination["coeff"]

        column_name = f"coeff_{p}__k_{k}"

        if k not in calculated_ar_params:
            try:
                calculated_ar = AutoReg(x_as_list, lags=k, trend="c")
                calculated_ar_params[k] = calculated_ar.fit().params
            except (ZeroDivisionError, LinAlgError, ValueError):
                calculated_ar_params[k] = [np.nan] * k

        mod = calculated_ar_params[k]
        if p <= k:
            try:
                res[column_name] = mod[p]
            except IndexError:
                res[column_name] = 0
        else:
            res[column_name] = np.nan

    return list(res.items())


def _change_quantiles_original(x, ql, qh, isabs, f_agg):
    if ql >= qh:
        return 0.0

    div = np.diff(x)
    if isabs:
        div = np.abs(div)
    # All values that originate from the corridor between the quantiles ql and qh will have the category 0,
    # other will be np.nan
    try:
        bin_cat = pd.qcut(x, [ql, qh], labels=False)
        bin_cat_0 = bin_cat == 0
    except ValueError:  # Occurs when ql are qh effectively equal, e.g. x is not long enough or is too categorical
        return 0.0
    # We only count changes that start and end inside the corridor
    ind = (bin_cat_0 & _roll(bin_cat_0, 1))[1:]
    if np.sum(ind) == 0:
        return 0.0

    ind_inside_corridor = np.where(ind == 1)
    aggregator = getattr(np, f_agg)
    return aggregator(div[ind_inside_corridor])


def _qcut_inside_len2_exact_from_edges(x, lo, hi):
    bins = np.array([lo, hi], dtype=np.float64)
    ids = np.searchsorted(bins, x, side="left")
    ids[x == lo] = 1
    na_mask = np.isnan(x) | (ids == 0) | (ids == len(bins))
    codes = (ids - 1).astype(np.float64)
    codes[na_mask] = np.nan
    return codes == 0


def change_quantiles_qcut_exact_combiner(x, param):
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    results = [None] * len(param)

    if n < 2:
        for idx, config in enumerate(param):
            ql = config["ql"]
            qh = config["qh"]
            isabs = config["isabs"]
            f_agg = config["f_agg"]
            key = f'f_agg_"{f_agg}"__isabs_{isabs}__qh_{qh}__ql_{ql}'
            results[idx] = (key, 0.0)
        return results

    div = np.diff(x)
    abs_div = np.abs(div)

    params_by_quant = defaultdict(list)
    q_values = set()
    invalid_pairs = set()
    for idx, config in enumerate(param):
        ql = config["ql"]
        qh = config["qh"]
        params_by_quant[(ql, qh)].append((idx, config))
        if (0 <= ql <= 1) and (0 <= qh <= 1):
            q_values.add(ql)
            q_values.add(qh)
        else:
            invalid_pairs.add((ql, qh))

    s = pd.Series(x).dropna()
    if len(q_values) > 0:
        q_list = sorted(q_values)
        qvals = s.quantile(q_list)
        if np.isscalar(qvals):
            edge_map = {q_list[0]: float(qvals)}
        else:
            edge_map = {q: float(qvals.loc[q]) for q in q_list}
    else:
        edge_map = {}

    for (ql, qh), entries in params_by_quant.items():
        if ql >= qh:
            for idx, config in entries:
                key = (
                    f'f_agg_"{config["f_agg"]}"__isabs_{config["isabs"]}__qh_{qh}__ql_{ql}'
                )
                results[idx] = (key, 0.0)
            continue
        if (ql, qh) in invalid_pairs:
            for idx, config in entries:
                key = (
                    f'f_agg_"{config["f_agg"]}"__isabs_{config["isabs"]}__qh_{qh}__ql_{ql}'
                )
                results[idx] = (key, 0.0)
            continue

        lo = edge_map.get(ql, np.nan)
        hi = edge_map.get(qh, np.nan)
        if np.isnan(lo) or np.isnan(hi) or lo == hi:
            for idx, config in entries:
                key = (
                    f'f_agg_"{config["f_agg"]}"__isabs_{config["isabs"]}__qh_{qh}__ql_{ql}'
                )
                results[idx] = (key, 0.0)
            continue

        inside = _qcut_inside_len2_exact_from_edges(x, lo, hi)
        ind = inside[1:] & inside[:-1]

        if not np.any(ind):
            for idx, config in entries:
                key = (
                    f'f_agg_"{config["f_agg"]}"__isabs_{config["isabs"]}__qh_{qh}__ql_{ql}'
                )
                results[idx] = (key, 0.0)
            continue

        for idx, config in entries:
            f_agg = config["f_agg"]
            isabs = config["isabs"]
            key = f'f_agg_"{f_agg}"__isabs_{isabs}__qh_{qh}__ql_{ql}'
            values = abs_div if isabs else div
            aggregator = getattr(np, f_agg)
            results[idx] = (key, aggregator(values[ind]))

    return results


@set_property("fctype", "combiner")
def change_quantiles(x, param):
    """
    First fixes a corridor given by the quantiles ql and qh of the distribution of x.
    Then calculates the average, absolute value of consecutive changes of the series x inside this corridor.

    Think about selecting a corridor on the
    y-Axis and only calculating the mean of the absolute change of the time series inside this corridor.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param param: contains dictionaries {"ql": ql, "qh": qh, "isabs": isabs, "f_agg": f_agg}
    :type param: list

    :return: the value of this feature
    :return type: list
    """
    return change_quantiles_qcut_exact_combiner(x, param)


@set_property("fctype", "simple")
def time_reversal_asymmetry_statistic(x, lag):
    """
    Returns the time reversal asymmetry statistic.

    This function calculates the value of

    .. math::

        \\frac{1}{n-2lag} \\sum_{i=1}^{n-2lag} x_{i + 2 \\cdot lag}^2 \\cdot x_{i + lag} - x_{i + lag} \\cdot  x_{i}^2

    which is

    .. math::

        \\mathbb{E}[L^2(X)^2 \\cdot L(X) - L(X) \\cdot X^2]

    where :math:`\\mathbb{E}` is the mean and :math:`L` is the lag operator. It was proposed in [1] as a
    promising feature to extract from time series.

    .. rubric:: References

    |  [1] Fulcher, B.D., Jones, N.S. (2014).
    |  Highly comparative feature-based time-series classification.
    |  Knowledge and Data Engineering, IEEE Transactions on 26, 3026–3037.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param lag: the lag that should be used in the calculation of the feature
    :type lag: int
    :return: the value of this feature
    :return type: float
    """
    n = len(x)
    x = np.asarray(x)
    if 2 * lag >= n:
        return 0.0

    one_lag = _roll(x, -lag)
    two_lag = _roll(x, 2 * -lag)
    return np.mean((two_lag * two_lag * one_lag - one_lag * x * x)[0 : (n - 2 * lag)])


@set_property("fctype", "simple")
def c3(x, lag):
    """
    Uses c3 statistics to measure non linearity in the time series

    This function calculates the value of

    .. math::

        \\frac{1}{n-2lag} \\sum_{i=1}^{n-2lag} x_{i + 2 \\cdot lag} \\cdot x_{i + lag} \\cdot x_{i}

    which is

    .. math::

        \\mathbb{E}[L^2(X) \\cdot L(X) \\cdot X]

    where :math:`\\mathbb{E}` is the mean and :math:`L` is the lag operator. It was proposed in [1] as a measure of
    non linearity in the time series.

    .. rubric:: References

    |  [1] Schreiber, T. and Schmitz, A. (1997).
    |  Discrimination power of measures for nonlinearity in a time series
    |  PHYSICAL REVIEW E, VOLUME 55, NUMBER 5

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param lag: the lag that should be used in the calculation of the feature
    :type lag: int
    :return: the value of this feature
    :return type: float
    """
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    n = x.size

    if 2 * lag >= n:
        return 0.0

    return np.mean((_roll(x, 2 * -lag) * _roll(x, -lag) * x)[0 : (n - 2 * lag)])


@set_property("fctype", "simple")
def mean_n_absolute_max(x, number_of_maxima):
    """
    Calculates the arithmetic mean of the n absolute maximum values of the time series.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param number_of_maxima: the number of maxima, which should be considered
    :type number_of_maxima: int

    :return: the value of this feature
    :return type: float
    """

    assert (
        number_of_maxima > 0
    ), f" number_of_maxima={number_of_maxima} which is not greater than 1"

    n_absolute_maximum_values = np.sort(np.absolute(x))[-number_of_maxima:]

    return np.mean(n_absolute_maximum_values) if len(x) > number_of_maxima else np.nan


@set_property("fctype", "simple")
def binned_entropy(x, max_bins):
    """
    First bins the values of x into max_bins equidistant bins.
    Then calculates the value of

    .. math::

        - \\sum_{k=0}^{min(max\\_bins, len(x))} p_k log(p_k) \\cdot \\mathbf{1}_{(p_k > 0)}

    where :math:`p_k` is the percentage of samples in bin :math:`k`.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param max_bins: the maximal number of bins
    :type max_bins: int
    :return: the value of this feature
    :return type: float
    """
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)

    # nan makes no sense here
    if np.isnan(x).any():
        return np.nan

    hist, bin_edges = np.histogram(x, bins=max_bins)
    probs = hist / x.size
    probs[probs == 0] = 1.0
    return -np.sum(probs * np.log(probs))


# todo - include latex formula
# todo - check if vectorizable
def _sample_entropy_original(x, m, r):
    x = np.array(x)

    # if one of the values is NaN, we can not compute anything meaningful
    if np.isnan(x).any():
        return np.nan

    tolerance = r * np.std(x)

    # Split time series and save all templates of length m
    # Basically we turn [1, 2, 3, 4] into [1, 2], [2, 3], [3, 4]
    xm = _into_subchunks(x, m)

    # Now calculate the maximum distance between each of those pairs
    #   np.abs(xmi - xm).max(axis=1)
    # and check how many are below the tolerance.
    # For speed reasons, we are not doing this in a nested for loop,
    # but with numpy magic.
    # Example:
    # if x = [1, 2, 3]
    # then xm = [[1, 2], [2, 3]]
    # so we will substract xm from [1, 2] => [[0, 0], [-1, -1]]
    # and from [2, 3] => [[1, 1], [0, 0]]
    # taking the abs and max gives us:
    # [0, 1] and [1, 0]
    # as the diagonal elements are always 0, we substract 1.
    B = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= tolerance) - 1 for xmi in xm])

    # Similar for computing A
    xmp1 = _into_subchunks(x, m + 1)

    A = np.sum(
        [np.sum(np.abs(xmi - xmp1).max(axis=1) <= tolerance) - 1 for xmi in xmp1]
    )

    # Return SampEn
    return -np.log(A / B)


def _count_chebyshev_matches_multi_tol(
    x: np.ndarray, m: int, tols: np.ndarray, block_size: int = 64
) -> np.ndarray:
    x = np.asarray(x)
    if tols.size == 0:
        return np.array([], dtype=np.int64)

    n = x.size
    k = n - m + 1
    if k <= 0:
        return np.zeros(tols.size, dtype=np.int64)

    windows = np.lib.stride_tricks.sliding_window_view(x, m)
    total_counts = np.zeros(tols.size, dtype=np.int64)

    for start in range(0, k, block_size):
        end = min(start + block_size, k)
        windows_i = windows[start:end]

        d = np.abs(windows_i[:, None, 0] - windows[None, :, 0])
        for t in range(1, m):
            dt = np.abs(windows_i[:, None, t] - windows[None, :, t])
            np.maximum(d, dt, out=d)

        for tol_idx, tol in enumerate(tols):
            total_counts[tol_idx] += np.sum(d <= tol)

    return total_counts


@set_property("high_comp_cost", True)
def sample_entropy_block_combiner(x, param):
    x = np.asarray(x)
    if np.isnan(x).any():
        res = []
        for config in param:
            key = f"m_{config['m']}__r_{config['r']:g}"
            res.append((key, np.nan))
        return res

    n = x.size
    std_x = np.std(x)
    params_by_m = defaultdict(list)
    for idx, config in enumerate(param):
        params_by_m[config["m"]].append((idx, config["r"]))

    results = [None] * len(param)
    for m, entries in params_by_m.items():
        tols = np.array([r * std_x for _, r in entries], dtype=float)

        k_m = n - m + 1
        if k_m <= 0:
            b_vals = np.zeros(len(entries), dtype=float)
        else:
            total_m = _count_chebyshev_matches_multi_tol(x, m, tols)
            b_vals = total_m.astype(float) - float(k_m)

        k_m1 = n - (m + 1) + 1
        if k_m1 <= 0:
            a_vals = np.zeros(len(entries), dtype=float)
        else:
            total_m1 = _count_chebyshev_matches_multi_tol(x, m + 1, tols)
            a_vals = total_m1.astype(float) - float(k_m1)

        values = -np.log(a_vals / b_vals)

        for (value, (idx, r)) in zip(values, entries):
            key = f"m_{m}__r_{r:g}"
            results[idx] = (key, value)

    return results


@set_property("high_comp_cost", True)
@set_property("fctype", "simple")
def sample_entropy(x):
    """
    Calculate and return sample entropy of x.

    .. rubric:: References

    |  [1] http://en.wikipedia.org/wiki/Sample_Entropy
    |  [2] https://www.ncbi.nlm.nih.gov/pubmed/10843903?dopt=Abstract

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray

    :return: the value of this feature
    :return type: float
    """
    return sample_entropy_block_combiner(x, [{"m": 2, "r": 0.2}])[0][1]


def _approximate_entropy_original(x, m, r):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)

    N = x.size
    r *= np.std(x)
    if r < 0:
        raise ValueError("Parameter r must be positive.")
    if N <= m + 1:
        return 0

    def _phi(m):
        x_re = np.array([x[i : i + m] for i in range(N - m + 1)])
        C = np.sum(
            np.max(np.abs(x_re[:, np.newaxis] - x_re[np.newaxis, :]), axis=2) <= r,
            axis=0,
        ) / (N - m + 1)
        return np.sum(np.log(C)) / (N - m + 1.0)

    return np.abs(_phi(m) - _phi(m + 1))


def _phi_block_chebyshev_multi_tol(
    x: np.ndarray, m: int, tols: np.ndarray, block_size: int = 64
) -> np.ndarray:
    x = np.asarray(x)
    if tols.size == 0:
        return np.array([], dtype=float)

    n = x.size
    k = n - m + 1
    windows = np.lib.stride_tricks.sliding_window_view(x, m)
    counts = np.zeros((tols.size, k), dtype=np.int64)

    for start in range(0, k, block_size):
        end = min(start + block_size, k)
        windows_i = windows[start:end]

        d = np.abs(windows_i[:, None, 0] - windows[None, :, 0])
        for t in range(1, m):
            dt = np.abs(windows_i[:, None, t] - windows[None, :, t])
            np.maximum(d, dt, out=d)

        for tol_idx, tol in enumerate(tols):
            counts[tol_idx] += np.sum(d <= tol, axis=0)

    c_vals = counts / k
    return np.sum(np.log(c_vals), axis=1) / k


@set_property("high_comp_cost", True)
def approximate_entropy_block_combiner(x, param):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    x = np.asarray(x)

    n = x.size
    std_x = np.std(x)
    params_by_m = defaultdict(list)
    for idx, config in enumerate(param):
        params_by_m[config["m"]].append((idx, config["r"]))

    results = [None] * len(param)
    for m, entries in params_by_m.items():
        tols = np.array([r * std_x for _, r in entries], dtype=float)
        if np.any(tols < 0):
            raise ValueError("Parameter r must be positive.")

        if n <= m + 1:
            values = np.zeros(len(entries), dtype=float)
        else:
            phi_m = _phi_block_chebyshev_multi_tol(x, m, tols)
            phi_m1 = _phi_block_chebyshev_multi_tol(x, m + 1, tols)
            values = np.abs(phi_m - phi_m1)

        for (value, (idx, r)) in zip(values, entries):
            key = f"m_{m}__r_{r:g}"
            results[idx] = (key, value)

    return results


@set_property("fctype", "combiner")
@set_property("high_comp_cost", True)
def approximate_entropy(x, param):
    """
    Implements a vectorized Approximate entropy algorithm.

        https://en.wikipedia.org/wiki/Approximate_entropy

    For short time-series this method is highly dependent on the parameters,
    but should be stable for N > 2000, see:

        Yentes et al. (2012) -
        *The Appropriate Use of Approximate Entropy and Sample Entropy with Short Data Sets*


    Other shortcomings and alternatives discussed in:

        Richman & Moorman (2000) -
        *Physiological time-series analysis using approximate entropy and sample entropy*

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param param: contains dictionaries {"m": m, "r": r}
    :type param: list

    :return: Approximate entropy
    :return type: list
    """
    return approximate_entropy_block_combiner(x, param)


@set_property("fctype", "simple")
def fourier_entropy(x, bins):
    """
    Calculate the binned entropy of the power spectral density of the time series
    (using the welch method).

    Ref: https://hackaday.io/project/707-complexity-of-a-time-series/details
    Ref: https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.welch.html

    """
    max_length_per_segment = 256
    x = np.asarray(x)
    _, pxx = welch(x, nperseg=min(len(x), max_length_per_segment))
    return binned_entropy(pxx / np.max(pxx), bins)


def _lempel_ziv_complexity_original(x, bins):
    """
    Calculate a complexity estimate based on the Lempel-Ziv compression
    algorithm.

    The complexity is defined as the number of dictionary entries (or sub-words) needed
    to encode the time series when viewed from left to right.
    For this, the time series is first binned into the given number of bins.
    Then it is converted into sub-words with different prefixes.
    The number of sub-words needed for this divided by the length of the time
    series is the complexity estimate.

    For example, if the time series (after binning in only 2 bins) would look like "100111",
    the different sub-words would be 1, 0, 01 and 11 and therefore the result is 4/6 = 0.66.

    Ref: https://github.com/Naereen/Lempel-Ziv_Complexity/blob/master/src/lempel_ziv_complexity.py

    """
    x = np.asarray(x)

    bins = np.linspace(np.min(x), np.max(x), bins + 1)[1:]
    sequence = np.searchsorted(bins, x, side="left")

    sub_strings = set()
    n = len(sequence)

    ind = 0
    inc = 1
    while ind + inc <= n:
        # convert to tuple in order to make it hashable
        sub_str = tuple(sequence[ind : ind + inc])
        if sub_str in sub_strings:
            inc += 1
        else:
            sub_strings.add(sub_str)
            ind += inc
            inc = 1
    return len(sub_strings) / n


def _lempel_ziv_complexity_bytes(sequence):
    n = len(sequence)
    seq_max = int(sequence.max()) if n else 0
    if seq_max <= np.iinfo(np.uint8).max:
        dtype = np.uint8
    elif seq_max <= np.iinfo(np.uint16).max:
        dtype = np.uint16
    elif seq_max <= np.iinfo(np.uint32).max:
        dtype = np.uint32
    else:
        dtype = np.uint64

    seq = np.ascontiguousarray(sequence, dtype=dtype)
    seq_bytes = seq.tobytes()
    step = seq.itemsize

    sub_strings = set()
    ind = 0
    inc = 1
    while ind + inc <= n:
        start = ind * step
        end = (ind + inc) * step
        sub_str = seq_bytes[start:end]
        if sub_str in sub_strings:
            inc += 1
        else:
            sub_strings.add(sub_str)
            ind += inc
            inc = 1
    return len(sub_strings) / n


@set_property("fctype", "simple")
def lempel_ziv_complexity(x, bins):
    x = np.asarray(x)
    bins = np.linspace(np.min(x), np.max(x), bins + 1)[1:]
    sequence = np.searchsorted(bins, x, side="left")
    return _lempel_ziv_complexity_bytes(sequence)


@set_property("fctype", "simple")
def permutation_entropy(x, tau, dimension):
    """
    Calculate the permutation entropy.

    Three steps are needed for this:

    1. chunk the data into sub-windows of length D starting every tau.
       Following the example from the reference, a vector

        x = [4, 7, 9, 10, 6, 11, 3

       with D = 3 and tau = 1 is turned into

           [[ 4,  7,  9],
            [ 7,  9, 10],
            [ 9, 10,  6],
            [10,  6, 11],
            [ 6, 11,  3]]

    2. replace each D-window by the permutation, that
       captures the ordinal ranking of the data.
       That gives

           [[0, 1, 2],
            [0, 1, 2],
            [1, 2, 0],
            [1, 0, 2],
            [1, 2, 0]]

    3. Now we just need to count the frequencies of every permutation
       and return their entropy (we use log_e and not log_2).

    Ref: https://www.aptech.com/blog/permutation-entropy/
         Bandt, Christoph and Bernd Pompe.
         “Permutation entropy: a natural complexity measure for time series.”
         Physical review letters 88 17 (2002): 174102 .
    """

    X = _into_subchunks(x, dimension, tau)
    if len(X) == 0:
        return np.nan
    # Now that is clearly black, magic, but see here:
    # https://stackoverflow.com/questions/54459554/numpy-find-index-in-sorted-array-in-an-efficient-way
    permutations = np.argsort(np.argsort(X))
    # Count the number of occurences
    _, counts = np.unique(permutations, axis=0, return_counts=True)
    # turn them into frequencies
    probs = counts / len(permutations)
    # and return their entropy
    return -np.sum(probs * np.log(probs))


@set_property("fctype", "simple")
def autocorrelation(x, lag):
    """
    Calculates the autocorrelation of the specified lag, according to the formula [1]

    .. math::

        \\frac{1}{(n-l)\\sigma^{2}} \\sum_{t=1}^{n-l}(X_{t}-\\mu )(X_{t+l}-\\mu)

    where :math:`n` is the length of the time series :math:`X_i`, :math:`\\sigma^2` its variance and :math:`\\mu` its
    mean. `l` denotes the lag.

    .. rubric:: References

    [1] https://en.wikipedia.org/wiki/Autocorrelation#Estimation

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param lag: the lag
    :type lag: int
    :return: the value of this feature
    :return type: float
    """
    # This is important: If a series is passed, the product below is calculated
    # based on the index, which corresponds to squaring the series.
    if isinstance(x, pd.Series):
        x = x.values
    if len(x) < lag:
        return np.nan
    # Slice the relevant subseries based on the lag
    y1 = x[: (len(x) - lag)]
    y2 = x[lag:]
    # Subtract the mean of the whole series x
    x_mean = np.mean(x)
    # The result is sometimes referred to as "covariation"
    sum_product = np.sum((y1 - x_mean) * (y2 - x_mean))
    # Return the normalized unbiased covariance
    v = np.var(x)
    if np.isclose(v, 0):
        return np.nan

    return sum_product / ((len(x) - lag) * v)


@set_property("fctype", "simple")
def quantile(x, q):
    """
    Calculates the q quantile of x. This is the value of x greater than q% of the ordered values from x.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param q: the quantile to calculate
    :type q: float
    :return: the value of this feature
    :return type: float
    """
    if len(x) == 0:
        return np.nan
    return np.quantile(x, q)


@set_property("fctype", "simple")
def number_crossing_m(x, m):
    """
    Calculates the number of crossings of x on m. A crossing is defined as two sequential values where the first value
    is lower than m and the next is greater, or vice-versa. If you set m to zero, you will get the number of zero
    crossings.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param m: the threshold for the crossing
    :type m: float
    :return: the value of this feature
    :return type: int
    """
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    # From https://stackoverflow.com/questions/3843017/efficiently-detect-sign-changes-in-python
    # However, we are not going with the fastest version as it breaks with pandas
    positive = x > m
    return np.where(np.diff(positive))[0].size


@set_property("fctype", "simple")
@set_property("minimal", True)
def maximum(x):
    """
    Calculates the highest value of the time series x.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    return np.max(x)


@set_property("fctype", "simple")
@set_property("minimal", True)
def absolute_maximum(x):
    """
    Calculates the highest absolute value of the time series x.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    return np.max(np.absolute(x)) if len(x) > 0 else np.nan


@set_property("fctype", "simple")
@set_property("minimal", True)
def minimum(x):
    """
    Calculates the lowest value of the time series x.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    return np.min(x)


@set_property("fctype", "simple")
def value_count(x, value):
    """
    Count occurrences of `value` in time series x.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param value: the value to be counted
    :type value: int or float
    :return: the count
    :rtype: int
    """
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)

    if np.isnan(value):
        return np.isnan(x).sum()

    return x[x == value].size


@set_property("fctype", "simple")
def range_count(x, min, max):
    """
    Count observed values within the interval [min, max).

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param min: the inclusive lower bound of the range
    :type min: int or float
    :param max: the exclusive upper bound of the range
    :type max: int or float
    :return: the count of values within the range
    :rtype: int
    """
    return np.sum((x >= min) & (x < max))


@set_property("fctype", "combiner")
def friedrich_coefficients(x, param):
    """
    Coefficients of polynomial :math:`h(x)`, which has been fitted to
    the deterministic dynamics of Langevin model

    .. math::
        \\dot{x}(t) = h(x(t)) + \\mathcal{N}(0,R)

    as described by [1].

    For short time-series this method is highly dependent on the parameters.

    .. rubric:: References

    |  [1] Friedrich et al. (2000): Physics Letters A 271, p. 217-222
    |  *Extracting model equations from experimental data*

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param param: contains dictionaries {"m": x, "r": y, "coeff": z} with x being positive integer,
                  the order of polynomial to fit for estimating fixed points of
                  dynamics, y positive float, the number of quantiles to use for averaging and finally z,
                  a positive integer corresponding to the returned coefficient
    :type param: list
    :return: the different feature values
    :return type: List[Tuple[str, float]]
    """
    # calculated is dictionary storing the calculated coefficients {m: {r: friedrich_coefficients}}
    calculated = defaultdict(dict)
    # res is a dictionary containing the results {"m_10__r_2__coeff_3": 15.43}
    res = {}

    for parameter_combination in param:
        m = parameter_combination["m"]
        r = parameter_combination["r"]
        coeff = parameter_combination["coeff"]

        assert coeff >= 0, f"Coefficients must be positive or zero. Found {coeff}"

        # calculate the current friedrich coefficients if they do not exist yet
        if m not in calculated or r not in calculated[m]:
            calculated[m][r] = _estimate_friedrich_coefficients(x, m, r)

        try:
            res[f"coeff_{coeff}__m_{m}__r_{r}"] = calculated[m][r][coeff]
        except IndexError:
            res[f"coeff_{coeff}__m_{m}__r_{r}"] = np.nan

    return list(res.items())


@set_property("fctype", "simple")
def max_langevin_fixed_point(x, r, m):
    """
    Largest fixed point of dynamics  :math:argmax_x {h(x)=0}` estimated from polynomial :math:`h(x)`,
    which has been fitted to the deterministic dynamics of Langevin model

    .. math::
        \\dot(x)(t) = h(x(t)) + R \\mathcal(N)(0,1)

    as described by

        Friedrich et al. (2000): Physics Letters A 271, p. 217-222
        *Extracting model equations from experimental data*

    For short time-series this method is highly dependent on the parameters.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param m: order of polynomial to fit for estimating fixed points of dynamics
    :type m: int
    :param r: number of quantiles to use for averaging
    :type r: float

    :return: Largest fixed point of deterministic dynamics
    :return type: float
    """

    coeff = _estimate_friedrich_coefficients(x, m, r)

    try:
        max_fixed_point = np.max(np.real(np.roots(coeff)))
    except (np.linalg.LinAlgError, ValueError):
        return np.nan

    return max_fixed_point


@set_property("fctype", "combiner")
def agg_linear_trend(x, param):
    """
    Calculates a linear least-squares regression for values of the time series that were aggregated over chunks versus
    the sequence from 0 up to the number of chunks minus one.

    This feature assumes the signal to be uniformly sampled. It will not use the time stamps to fit the model.

    The parameters attr controls which of the characteristics are returned. Possible extracted attributes are "pvalue",
    "rvalue", "intercept", "slope", "stderr", see the documentation of linregress for more information.

    The chunksize is regulated by "chunk_len". It specifies how many time series values are in each chunk.

    Further, the aggregation function is controlled by "f_agg", which can use "max", "min" or , "mean", "median"

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param param: contains dictionaries {"attr": x, "chunk_len": l, "f_agg": f} with x, f an string and l an int
    :type param: list
    :return: the different feature values
    :return type: Iterator[Tuple[str, float]]
    """
    # todo: we could use the index of the DataFrame here

    calculated_agg = defaultdict(dict)
    res_data = []
    res_index = []

    for parameter_combination in param:

        chunk_len = parameter_combination["chunk_len"]
        f_agg = parameter_combination["f_agg"]

        if f_agg not in calculated_agg or chunk_len not in calculated_agg[f_agg]:
            if chunk_len >= len(x):
                calculated_agg[f_agg][chunk_len] = np.nan
            else:
                aggregate_result = _aggregate_on_chunks(x, f_agg, chunk_len)
                lin_reg_result = linregress(
                    range(len(aggregate_result)), aggregate_result
                )
                calculated_agg[f_agg][chunk_len] = lin_reg_result

        attr = parameter_combination["attr"]

        if chunk_len >= len(x):
            res_data.append(np.nan)
        else:
            res_data.append(getattr(calculated_agg[f_agg][chunk_len], attr))

        res_index.append(f'attr_"{attr}"__chunk_len_{chunk_len}__f_agg_"{f_agg}"')

    return zip(res_index, res_data)


@set_property("fctype", "combiner")
def energy_ratio_by_chunks(x, param):
    """
    Calculates the sum of squares of chunk i out of N chunks expressed as a ratio with the sum of squares over the whole
    series.

    Takes as input parameters the number num_segments of segments to divide the series into and segment_focus
    which is the segment number (starting at zero) to return a feature on.

    If the length of the time series is not a multiple of the number of segments, the remaining data points are
    distributed on the bins starting from the first. For example, if your time series consists of 8 entries, the
    first two bins will contain 3 and the last two values, e.g. `[ 0.,  1.,  2.], [ 3.,  4.,  5.]` and `[ 6.,  7.]`.

    Note that the answer for `num_segments = 1` is a trivial "1" but we handle this scenario
    in case somebody calls it. Sum of the ratios should be 1.0.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param param: contains dictionaries {"num_segments": N, "segment_focus": i} with N, i both ints
    :return: the feature values
    :return type: List[Tuple[str, float]]
    """
    res_data = []
    res_index = []
    full_series_energy = np.sum(x**2)

    for parameter_combination in param:
        num_segments = parameter_combination["num_segments"]
        segment_focus = parameter_combination["segment_focus"]
        assert segment_focus < num_segments
        assert num_segments > 0

        if full_series_energy == 0:
            res_data.append(np.nan)
        else:
            res_data.append(
                np.sum(np.array_split(x, num_segments)[segment_focus] ** 2.0)
                / full_series_energy
            )

        res_index.append(f"num_segments_{num_segments}__segment_focus_{segment_focus}")

    # Materialize as list for Python 3 compatibility with name handling
    return list(zip(res_index, res_data))


@set_property("fctype", "combiner")
@set_property("input", "pd.Series")
@set_property("index_type", pd.DatetimeIndex)
def linear_trend_timewise(x, param):
    """
    Calculate a linear least-squares regression for the values of the time series versus the sequence from 0 to
    length of the time series minus one.
    This feature uses the index of the time series to fit the model, which must be of a datetime
    dtype.
    The parameters control which of the characteristics are returned.

    Possible extracted attributes are "pvalue", "rvalue", "intercept", "slope", "stderr", see the documentation of
    linregress for more information.

    :param x: the time series to calculate the feature of. The index must be datetime.
    :type x: pandas.Series
    :param param: contains dictionaries {"attr": x} with x an string, the attribute name of the regression model
    :type param: list
    :return: the different feature values
    :return type: List[Tuple[str, float]]
    """
    ix = x.index

    # Get differences between each timestamp and the first timestamp in seconds.
    # Then convert to hours and reshape for linear regression
    seconds_per_hr = 3600
    times_seconds = (ix - ix[0]).total_seconds()
    times_hours = np.asarray(times_seconds / float(seconds_per_hr))

    lin_reg = linregress(times_hours, x.values)

    return [
        (f'attr_"{config["attr"]}"', getattr(lin_reg, config["attr"]))
        for config in param
    ]


@set_property("fctype", "simple")
def count_above(x, t):
    """
    Returns the percentage of values in x that are higher than t

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param t: value used as threshold
    :type t: float

    :return: the value of this feature
    :return type: float
    """
    return np.sum(x >= t) / len(x)


@set_property("fctype", "simple")
def count_below(x, t):
    """
    Returns the percentage of values in x that are lower than t

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param t: value used as threshold
    :type t: float

    :return: the value of this feature
    :return type: float
    """
    return np.sum(x <= t) / len(x)


@set_property("fctype", "simple")
def benford_correlation(x):
    """
     Useful for anomaly detection applications [1][2]. Returns the correlation from first digit distribution when
     compared to the Newcomb-Benford's Law distribution [3][4].

     .. math::

         P(d)=\\log_{10}\\left(1+\\frac{1}{d}\\right)

     where :math:`P(d)` is the Newcomb-Benford distribution for :math:`d` that is the leading digit of the number
     {1, 2, 3, 4, 5, 6, 7, 8, 9}.

     .. rubric:: References

     |  [1] A Statistical Derivation of the Significant-Digit Law, Theodore P. Hill, Statistical Science, 1995
     |  [2] The significant-digit phenomenon, Theodore P. Hill, The American Mathematical Monthly, 1995
     |  [3] The law of anomalous numbers, Frank Benford, Proceedings of the American philosophical society, 1938
     |  [4] Note on the frequency of use of the different digits in natural numbers, Simon Newcomb, American Journal of
     |  mathematics, 1881

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    x = np.asarray(x)

    # retrieve first digit from data
    x = np.array(
        [int(str(np.format_float_scientific(i))[:1]) for i in np.abs(np.nan_to_num(x))]
    )

    # benford distribution
    benford_distribution = np.array([np.log10(1 + 1 / n) for n in range(1, 10)])

    data_distribution = np.array([(x == n).mean() for n in range(1, 10)])

    # np.corrcoef outputs the normalized covariance (correlation) between benford_distribution and data_distribution.
    # In this case returns a 2x2 matrix, the  [0, 1] and [1, 1] are the values between the two arrays
    return np.corrcoef(benford_distribution, data_distribution)[0, 1]


@set_property("fctype", "combiner")
@set_property("dependency_available", mp is not None)
def matrix_profile(x, param):
    """
    Calculates the 1-D Matrix Profile[1] and returns Tukey's Five Number Set plus the mean of that Matrix Profile.

    This feature is not supported anymore, since matrixprofile does not up to date with latest Python releases. To use
    it, you can install the extra with pip install tsfresh[matrixprofile].

    .. rubric:: References

    |  [1] Yeh et.al (2016), IEEE ICDM

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param param: contains dictionaries
                  {"sample_pct": x, "threshold": y, "feature": z}
                  with sample_pct and threshold being parameters of the matrixprofile
                  package https://matrixprofile.docs.matrixprofile.org/api.html#matrixprofile-compute
                  and feature being one of "min", "max", "mean", "median", "25", "75"
                  and decides which feature of the matrix profile to extract
    :type param: list
    :return: the different feature values
    :return type: List[Tuple[str, float]]
    """
    if mp is None:
        raise ImportError(
            "Could not import matrixprofile but it was required. Please install the extra to use it."
        )

    x = np.asarray(x)

    def _calculate_mp(**kwargs):
        """Calculate the matrix profile using the specified window, or the max subsequence if no window is specified"""
        try:
            if "windows" in kwargs:
                m_p = mp.compute(x, **kwargs)["mp"]

            else:
                m_p = mp.algorithms.maximum_subsequence(x, include_pmp=True, **kwargs)[
                    "pmp"
                ][-1]

            return m_p

        except NoSolutionPossible:
            return [np.nan]

    # The already calculated matrix profiles
    matrix_profiles = {}

    # The results
    res = {}

    for kwargs in param:
        kwargs = kwargs.copy()
        key = convert_to_output_format(kwargs)
        feature = kwargs.pop("feature")

        featureless_key = convert_to_output_format(kwargs)
        if featureless_key not in matrix_profiles:
            matrix_profiles[featureless_key] = _calculate_mp(**kwargs)

        m_p = matrix_profiles[featureless_key]

        # Set all features to nan if Matrix Profile is nan (cannot be computed)
        if len(m_p) == 1:
            res[key] = np.nan

        # Handle all other Matrix Profile instances
        else:

            finite_indices = np.isfinite(m_p)

            feature_map = {
                "min": np.min,
                "max": np.max,
                "mean": np.mean,
                "median": np.median,
                "25": lambda data: np.percentile(data, 25),
                "75": lambda data: np.percentile(data, 75),
            }

            if feature in feature_map:
                res[key] = feature_map[feature](m_p[finite_indices])
            else:
                raise ValueError(f"Unknown feature {feature} for the matrix profile")

    return list(res.items())


@set_property("fctype", "combiner")
def query_similarity_count(x, param):
    """
    This feature calculator accepts an input query subsequence parameter,
    compares the query (under z-normalized Euclidean distance) to all
    subsequences within the time series, and returns a count of the number
    of times the query was found in the time series (within some predefined
    maximum distance threshold). Note that this feature will always return
    `np.nan` when no query subsequence is provided and so users will need
    to enable this feature themselves.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param param: contains dictionaries
                  {"query": Q, "threshold": thr, "normalize": norm}
                  with `Q` (numpy.ndarray), the query subsequence to compare the
                  time series against. If `Q` is omitted then a value of zero
                  is returned. Additionally, `thr` (float), the maximum
                  z-normalized Euclidean distance threshold for which to
                  increment the query similarity count. If `thr` is omitted
                  then a default threshold of `thr=0.0` is used, which
                  corresponds to finding exact matches to `Q`. Finally, for
                  non-normalized (i.e., without z-normalization) Euclidean set
                  `norm` (bool) to `False.
    :type param: list[dict]
    :return x: the different feature values
    :return type: List[Tuple[str, int | np.nan]]
    """
    res = {}
    T = np.asarray(x).astype(float)

    for kwargs in param:
        key = convert_to_output_format(kwargs)
        normalize = kwargs.get("normalize", True)
        threshold = kwargs.get("threshold", 0.0)
        Q = kwargs.get("query", None)
        Q = np.asarray(Q).astype(float)
        count = np.nan
        if Q is not None and Q.size >= 3:
            if normalize:
                distance_profile = stumpy.core.mass(Q, T)
            else:
                distance_profile = stumpy.core.mass_absolute(Q, T)
            count = np.sum(distance_profile <= threshold)

        res[key] = count

    return list(res.items())

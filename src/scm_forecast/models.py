"""Univariate forecasting methods.

Every forecaster has the signature ``f(y, h) -> np.ndarray`` where ``y`` is the
observed history (1-D, oldest first) and ``h`` the number of months ahead. All
forecasts are clipped at zero because demand cannot be negative.

Baselines (same family as the spreadsheet exercise):
    naive, seasonal_naive, ma3, wma3, ses, holt, linear_trend
Advanced:
    holt_winters_add / holt_winters_mul (statsmodels ETS family), croston, sba
Global ML model (trained across all SKUs): see :mod:`scm_forecast.ml`.
"""
from __future__ import annotations

import warnings
from typing import Callable

import numpy as np

SEASON = 12


def _clip(a) -> np.ndarray:
    return np.maximum(np.asarray(a, dtype=float), 0.0)


# --------------------------------------------------------------------------- baselines
def naive(y, h: int) -> np.ndarray:
    return _clip(np.repeat(y[-1], h))


def seasonal_naive(y, h: int, m: int = SEASON) -> np.ndarray:
    if len(y) < m:
        return naive(y, h)
    last = np.asarray(y[-m:], dtype=float)
    return _clip(last[np.arange(h) % m])


def moving_average(y, h: int, n: int = 3) -> np.ndarray:
    """Simple moving average of the last ``n`` observations (flat forecast)."""
    return _clip(np.repeat(np.mean(y[-n:]), h))


def weighted_moving_average(y, h: int, weights=(0.5, 0.3, 0.2)) -> np.ndarray:
    """WMA; ``weights[0]`` applies to the most recent observation."""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    recent = np.asarray(y[-len(w):], dtype=float)[::-1]
    return _clip(np.repeat(np.dot(w, recent), h))


def ses_level(y, alpha: float = 0.3) -> float:
    """Final level of simple exponential smoothing, initialised at y[0]."""
    level = float(y[0])
    for v in y[1:]:
        level = alpha * v + (1 - alpha) * level
    return level


def ses(y, h: int, alpha: float = 0.3) -> np.ndarray:
    """Simple exponential smoothing with a fixed alpha (spreadsheet default 0.3)."""
    return _clip(np.repeat(ses_level(np.asarray(y, float), alpha), h))


def linear_trend(y, h: int) -> np.ndarray:
    """Least-squares trend line fitted on the full history, extrapolated h steps."""
    y = np.asarray(y, dtype=float)
    t = np.arange(len(y))
    b, a = np.polyfit(t, y, 1)
    return _clip(a + b * (len(y) + np.arange(h)))


# --------------------------------------------------------------------------- statsmodels
def _fit_es(y, **kwargs):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ExponentialSmoothing(np.asarray(y, float), initialization_method="estimated", **kwargs)
        return model.fit(optimized=True)


def holt(y, h: int) -> np.ndarray:
    """Holt's linear trend (additive, damped) with optimised smoothing parameters."""
    try:
        fit = _fit_es(y, trend="add", damped_trend=True)
        out = fit.forecast(h)
        if not np.all(np.isfinite(out)):
            raise FloatingPointError
        return _clip(out)
    except Exception:
        return ses(y, h)


def holt_winters_add(y, h: int, m: int = SEASON) -> np.ndarray:
    """Holt-Winters, additive damped trend and additive seasonality."""
    if len(y) < 2 * m:
        return holt(y, h)
    try:
        fit = _fit_es(y, trend="add", damped_trend=True, seasonal="add", seasonal_periods=m)
        out = fit.forecast(h)
        if not np.all(np.isfinite(out)):
            raise FloatingPointError
        return _clip(out)
    except Exception:
        return holt(y, h)


def holt_winters_mul(y, h: int, m: int = SEASON) -> np.ndarray:
    """Holt-Winters, additive damped trend and multiplicative seasonality.

    Multiplicative seasonality needs strictly positive data. For series that
    contain zero months the method falls back to the additive variant, so it is
    defined for every SKU and aggregate comparisons use the same 50 series.
    """
    y = np.asarray(y, dtype=float)
    if len(y) < 2 * m or y.min() <= 0:
        return holt_winters_add(y, h, m)
    try:
        fit = _fit_es(y, trend="add", damped_trend=True, seasonal="mul", seasonal_periods=m)
        out = fit.forecast(h)
        if not np.all(np.isfinite(out)):
            raise FloatingPointError
        return _clip(out)
    except Exception:
        return holt_winters_add(y, h, m)


# --------------------------------------------------------------------------- intermittent
def croston_rate(y, alpha: float = 0.1, variant: str = "classic") -> float:
    """Croston's method: smooth non-zero sizes (z) and inter-demand intervals (p).

    The demand rate per period is ``z / p``. ``variant="sba"`` applies the
    Syntetos-Boylan bias correction factor ``(1 - alpha / 2)``.
    """
    y = np.asarray(y, dtype=float)
    nz = np.flatnonzero(y > 0)
    if nz.size == 0:
        return 0.0
    z = y[nz[0]]
    p = float(nz[0] + 1)
    q = 1
    for t in range(nz[0] + 1, len(y)):
        if y[t] > 0:
            z = z + alpha * (y[t] - z)
            p = p + alpha * (q - p)
            q = 1
        else:
            q += 1
    rate = z / p
    if variant == "sba":
        rate *= 1.0 - alpha / 2.0
    elif variant != "classic":
        raise ValueError(f"unknown Croston variant {variant!r}")
    return float(rate)


def croston(y, h: int, alpha: float = 0.1) -> np.ndarray:
    return _clip(np.repeat(croston_rate(y, alpha, "classic"), h))


def sba(y, h: int, alpha: float = 0.1) -> np.ndarray:
    return _clip(np.repeat(croston_rate(y, alpha, "sba"), h))


# --------------------------------------------------------------------------- registry
BASELINE_METHODS: dict[str, Callable] = {
    "naive": naive,
    "ma3": moving_average,
    "wma3": weighted_moving_average,
    "ses": ses,
    "holt": holt,
    "linear_trend": linear_trend,
}
ADVANCED_METHODS: dict[str, Callable] = {
    "seasonal_naive": seasonal_naive,
    "holt_winters_add": holt_winters_add,
    "holt_winters_mul": holt_winters_mul,
    "croston": croston,
    "sba": sba,
}
LOCAL_METHODS: dict[str, Callable] = {**BASELINE_METHODS, **ADVANCED_METHODS}
GLOBAL_METHODS = ("lightgbm",)

METHOD_LABELS = {
    "naive": "Naive",
    "ma3": "Moving average (3)",
    "wma3": "Weighted MA (3)",
    "ses": "SES (alpha=0.3)",
    "holt": "Holt (damped)",
    "linear_trend": "Linear trend (OLS)",
    "seasonal_naive": "Seasonal naive",
    "holt_winters_add": "Holt-Winters (add.)",
    "holt_winters_mul": "Holt-Winters (mult.)",
    "croston": "Croston",
    "sba": "Croston-SBA",
    "lightgbm": "LightGBM (global)",
}

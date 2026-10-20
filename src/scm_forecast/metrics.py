"""Forecast accuracy metrics."""
from __future__ import annotations

import numpy as np


def _arr(a):
    return np.asarray(a, dtype=float)


def mae(y_true, y_pred) -> float:
    """Mean absolute error (Excel: MAD)."""
    return float(np.mean(np.abs(_arr(y_true) - _arr(y_pred))))


def mse(y_true, y_pred) -> float:
    return float(np.mean((_arr(y_true) - _arr(y_pred)) ** 2))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def bias(y_true, y_pred) -> float:
    """Mean error, positive = over-forecast."""
    return float(np.mean(_arr(y_pred) - _arr(y_true)))


def mape(y_true, y_pred) -> float:
    """Classic MAPE in percent; periods with zero actuals are skipped.

    Returns NaN if every actual is zero. Shown for continuity with a
    spreadsheet workflow; sMAPE/MASE are preferred for intermittent demand.
    """
    a, f = _arr(y_true), _arr(y_pred)
    m = a != 0
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs((a[m] - f[m]) / a[m])) * 100.0)


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE in percent (0-200). 0/0 terms count as zero error."""
    a, f = _arr(y_true), _arr(y_pred)
    den = np.abs(a) + np.abs(f)
    num = 2.0 * np.abs(a - f)
    terms = np.where(den == 0, 0.0, num / np.where(den == 0, 1.0, den))
    return float(np.mean(terms) * 100.0)


def wape(y_true, y_pred) -> float:
    """Weighted absolute percentage error: sum|e| / sum(actual), in percent."""
    a, f = _arr(y_true), _arr(y_pred)
    s = np.abs(a).sum()
    return float("nan") if s == 0 else float(np.abs(a - f).sum() / s * 100.0)


def mase_scale(y_train, m: int = 1) -> float:
    """In-sample MAE of the (seasonal) naive forecast, the MASE denominator."""
    y = _arr(y_train)
    if len(y) <= m:
        return float("nan")
    return float(np.mean(np.abs(y[m:] - y[:-m])))


def mase(y_true, y_pred, y_train=None, m: int = 1, scale: float | None = None) -> float:
    """Mean absolute scaled error (Hyndman & Koehler, 2006).

    ``scale`` may be passed directly; otherwise it is computed from ``y_train``.
    A value below 1 means the forecast beats the in-sample naive forecast.
    Returns NaN when the scale is zero (a constant training series).
    """
    if scale is None:
        if y_train is None:
            raise ValueError("provide y_train or scale")
        scale = mase_scale(y_train, m)
    if not np.isfinite(scale) or scale == 0:
        return float("nan")
    return mae(y_true, y_pred) / scale

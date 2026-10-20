"""Rolling-origin backtesting, scoring and per-SKU method selection."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import metrics as M
from .ml import MIN_ORIGIN, lightgbm_forecast
from .models import LOCAL_METHODS

FC_COLUMNS = ["sku", "method", "origin", "h", "target_idx", "y_true", "y_pred"]


def _local_records(i: int, sku: str, y_full: np.ndarray, origins, horizon, methods):
    rows = []
    T = len(y_full)
    for o in origins:
        h_max = min(horizon, T - o)
        if h_max <= 0:
            continue
        train = y_full[:o]
        actual = y_full[o : o + h_max]
        for name in methods:
            pred = LOCAL_METHODS[name](train, h_max)
            for k in range(h_max):
                rows.append((sku, name, o, k + 1, o + k, actual[k], pred[k]))
    return rows


def rolling_origin_forecasts(
    Y: pd.DataFrame | np.ndarray,
    origins: Iterable[int],
    horizon: int,
    methods: Sequence[str] | None = None,
    include_lightgbm: bool = True,
    category_codes: np.ndarray | None = None,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Expanding-window rolling-origin forecasts.

    For each origin ``o`` the model sees only the first ``o`` months and forecasts
    ``o .. o+horizon-1``; horizons running past the end of ``Y`` are truncated.
    Returns a long frame with columns :data:`FC_COLUMNS`.
    """
    if isinstance(Y, pd.DataFrame):
        skus = list(Y.index)
        arr = Y.to_numpy(float)
    else:
        arr = np.asarray(Y, float)
        skus = [f"S{i}" for i in range(arr.shape[0])]
    origins = list(origins)
    methods = list(methods) if methods is not None else list(LOCAL_METHODS)

    chunks = Parallel(n_jobs=n_jobs)(
        delayed(_local_records)(i, s, arr[i], origins, horizon, methods)
        for i, s in enumerate(skus)
    )
    rows = [r for c in chunks for r in c]

    if include_lightgbm:
        if category_codes is None:
            category_codes = np.zeros(len(skus), int)
        T = arr.shape[1]

        def _one(o):
            h_max = min(horizon, T - o)
            if h_max <= 0 or o < MIN_ORIGIN + 1:
                return []
            pred = lightgbm_forecast(arr[:, : o], o, h_max, category_codes)
            return [
                (skus[i], "lightgbm", o, k + 1, o + k, arr[i, o + k], pred[i, k])
                for i in range(len(skus))
                for k in range(h_max)
            ]

        for rec in Parallel(n_jobs=n_jobs)(delayed(_one)(o) for o in origins):
            rows.extend(rec)

    return pd.DataFrame(rows, columns=FC_COLUMNS)


def score_forecasts(
    fc: pd.DataFrame, scale: pd.Series, horizons: Sequence[int] | None = None
) -> pd.DataFrame:
    """Per (sku, method) accuracy. ``scale`` maps sku -> MASE denominator.

    If any forecast for a (sku, method) is NaN the method is ineligible and all
    its metrics are NaN.
    """
    if horizons is not None:
        fc = fc[fc["h"].isin(list(horizons))]
    out = []
    for (sku, method), g in fc.groupby(["sku", "method"], sort=False):
        a, f = g["y_true"].to_numpy(), g["y_pred"].to_numpy()
        if np.isnan(f).any():
            vals = dict.fromkeys(["mae", "rmse", "smape", "mase", "wape", "bias", "n"], np.nan)
        else:
            vals = dict(
                mae=M.mae(a, f),
                rmse=M.rmse(a, f),
                smape=M.smape(a, f),
                mase=M.mase(a, f, scale=scale[sku]),
                wape=M.wape(a, f),
                bias=M.bias(a, f),
                n=len(a),
            )
        out.append({"sku": sku, "method": method, **vals})
    return pd.DataFrame(out)


# Level-only methods considered for intermittent / lumpy SKUs (Syntetos-Boylan class).
# Seasonal and trend models fitted to mostly-zero series are not meaningful, so
# they are excluded from the candidate set for those SKUs.
INTERMITTENT_ELIGIBLE = ("naive", "ma3", "wma3", "ses", "croston", "sba")


def select_best(
    scores: pd.DataFrame,
    metric: str = "mase",
    candidates=None,
    sbc_class: pd.Series | None = None,
) -> pd.DataFrame:
    """Pick the method with the lowest ``metric`` for each SKU.

    If ``sbc_class`` (sku -> smooth/erratic/intermittent/lumpy) is given,
    intermittent and lumpy SKUs may only choose from :data:`INTERMITTENT_ELIGIBLE`.
    """
    s = scores if candidates is None else scores[scores["method"].isin(list(candidates))]
    s = s.dropna(subset=[metric])
    if sbc_class is not None:
        cls = s["sku"].map(sbc_class)
        restricted = cls.isin(["intermittent", "lumpy"]) & ~s["method"].isin(INTERMITTENT_ELIGIBLE)
        s = s[~restricted]
    idx = s.groupby("sku")[metric].idxmin()
    best = s.loc[idx, ["sku", "method", metric]].rename(
        columns={"method": "best_method", metric: f"best_{metric}"}
    )
    return best.set_index("sku")


def aggregate_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Portfolio-level summary per method (mean over SKUs)."""
    g = scores.groupby("method")
    return pd.DataFrame(
        {
            "mean_mase": g["mase"].mean(),
            "median_mase": g["mase"].median(),
            "mean_smape": g["smape"].mean(),
            "mean_mae": g["mae"].mean(),
            "mean_rmse": g["rmse"].mean(),
            "mean_wape": g["wape"].mean(),
            "n_skus": g["mase"].count(),
        }
    ).sort_values("mean_mase")

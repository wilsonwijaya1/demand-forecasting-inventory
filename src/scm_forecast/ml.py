"""Global LightGBM forecaster trained across all SKUs with lag features.

One model per forecast horizon (direct strategy). Each SKU is rescaled by the
mean of its last 12 observations so that a single model can learn shared
dynamics (seasonality, mean reversion, intermittency) across SKUs of very
different volume.
"""
from __future__ import annotations

import warnings

import numpy as np

LAGS = (1, 2, 3, 6, 12)
MIN_ORIGIN = 13  # need 12 lags + at least one observation


def _features(Y: np.ndarray, o: int, target_idx: int, cat_codes: np.ndarray):
    """Feature matrix (n_sku, n_feat) for forecasting index ``target_idx``
    using only Y[:, :o]. Also returns the per-SKU scale."""
    hist = Y[:, :o]
    last12 = hist[:, -12:]
    scale = np.maximum(last12.mean(axis=1), 0.5)
    cols = [hist[:, o - lag] / scale for lag in LAGS]
    cols.append(hist[:, -3:].mean(axis=1) / scale)
    cols.append(hist[:, -6:].mean(axis=1) / scale)
    cols.append(last12.std(axis=1) / scale)
    cols.append((last12 == 0).mean(axis=1))  # share of zero months
    month = target_idx % 12
    cols.append(np.full(len(Y), np.sin(2 * np.pi * month / 12)))
    cols.append(np.full(len(Y), np.cos(2 * np.pi * month / 12)))
    cols.append(cat_codes.astype(float))
    return np.column_stack(cols), scale


def lightgbm_forecast(
    Y: np.ndarray, o: int, horizon: int, cat_codes: np.ndarray, seed: int = 0
) -> np.ndarray:
    """Forecast ``horizon`` steps ahead for all SKUs from origin ``o``.

    ``Y`` is (n_sku, T); only ``Y[:, :o]`` is used. Returns (n_sku, horizon).
    """
    import lightgbm as lgb

    n_sku = Y.shape[0]
    out = np.zeros((n_sku, horizon))
    for h in range(1, horizon + 1):
        Xs, ys = [], []
        for o2 in range(MIN_ORIGIN, o - h + 1):
            X, scale = _features(Y, o2, o2 + h - 1, cat_codes)
            Xs.append(X)
            ys.append(Y[:, o2 + h - 1] / scale)
        if not Xs:
            raise ValueError("not enough history to train LightGBM")
        X_train, y_train = np.vstack(Xs), np.concatenate(ys)
        model = lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=1,
            deterministic=True,
            force_col_wise=True,
            verbose=-1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
            X_new, scale = _features(Y, o, o + h - 1, cat_codes)
            out[:, h - 1] = model.predict(X_new) * scale
    return np.maximum(out, 0.0)

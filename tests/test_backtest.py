import numpy as np
import pandas as pd

from scm_forecast import backtest as bt


def _toy(n_sku=3, T=48, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    rows = [100 + 10 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 3, T) + 0.5 * t * i for i in range(n_sku)]
    return pd.DataFrame(np.round(rows), index=[f"S{i}" for i in range(n_sku)])


def test_no_look_ahead_leakage():
    Y = _toy()
    methods = ["naive", "ma3", "ses", "holt_winters_add", "croston"]
    a = bt.rolling_origin_forecasts(Y, [30, 36], 3, methods, include_lightgbm=False, n_jobs=1)
    Y2 = Y.copy()
    Y2.iloc[:, 30:] += 1000  # corrupt everything from the first origin onwards
    b = bt.rolling_origin_forecasts(Y2, [30], 3, methods, include_lightgbm=False, n_jobs=1)
    a30 = a[a.origin == 30].reset_index(drop=True)
    pd.testing.assert_series_equal(a30["y_pred"], b["y_pred"].reset_index(drop=True))


def test_horizon_is_truncated_at_end_of_data_and_selection_picks_lowest_error():
    Y = _toy(T=40)
    fc = bt.rolling_origin_forecasts(Y, [38], 3, ["naive"], include_lightgbm=False, n_jobs=1)
    assert fc["h"].max() == 2  # only two months remain after origin 38
    scale = pd.Series(1.0, index=Y.index)
    scores = bt.score_forecasts(
        pd.DataFrame(
            {
                "sku": ["a"] * 4, "method": ["good", "good", "bad", "bad"],
                "origin": 0, "h": 1, "target_idx": 0,
                "y_true": [10, 10, 10, 10], "y_pred": [10, 11, 20, 0],
            }
        ),
        pd.Series({"a": 1.0}),
    )
    assert bt.select_best(scores).loc["a", "best_method"] == "good"

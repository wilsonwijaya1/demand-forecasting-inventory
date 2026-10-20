import numpy as np
import pytest

from scm_forecast import metrics as M


def test_mae_rmse_mse_known_values():
    a = [10, 20, 30]
    f = [12, 18, 33]
    assert M.mae(a, f) == pytest.approx((2 + 2 + 3) / 3)
    assert M.mse(a, f) == pytest.approx((4 + 4 + 9) / 3)
    assert M.rmse(a, f) == pytest.approx(np.sqrt(17 / 3))
    assert M.bias(a, f) == pytest.approx((2 - 2 + 3) / 3)


def test_perfect_forecast_gives_zero_error():
    a = np.array([5.0, 0.0, 7.0, 3.0])
    assert M.mae(a, a) == 0
    assert M.rmse(a, a) == 0
    assert M.smape(a, a) == 0
    assert M.mase(a, a, y_train=[1, 2, 3, 4]) == 0


def test_smape_handles_zero_zero_and_is_bounded():
    # 0/0 counts as zero error; a forecast of 0 for a positive actual is the worst case (200%)
    assert M.smape([0, 0], [0, 0]) == 0
    assert M.smape([10], [0]) == pytest.approx(200.0)
    assert M.smape([100], [110]) == pytest.approx(200 * 10 / 210)


def test_mape_skips_zero_actuals():
    assert M.mape([0, 100], [5, 90]) == pytest.approx(10.0)
    assert np.isnan(M.mape([0, 0], [1, 1]))


def test_mase_matches_hand_calculation_and_naive_equals_one():
    train = [10, 12, 11, 15]  # naive in-sample MAE = (2 + 1 + 4) / 3
    scale = 7 / 3
    assert M.mase_scale(train) == pytest.approx(scale)
    assert M.mase([20, 20], [18, 23], y_train=train) == pytest.approx(2.5 / scale)
    # Forecasting by the naive method on the training data itself scores exactly 1
    y = np.array([3.0, 8.0, 5.0, 9.0, 4.0])
    assert M.mase(y[1:], y[:-1], y_train=y) == pytest.approx(1.0)


def test_mase_constant_training_series_is_nan():
    assert np.isnan(M.mase([1, 2], [1, 2], y_train=[4, 4, 4]))


def test_wape():
    assert M.wape([100, 100], [90, 120]) == pytest.approx(15.0)

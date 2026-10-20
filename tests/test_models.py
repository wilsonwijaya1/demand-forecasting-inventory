import numpy as np
import pytest

from scm_forecast import models as m


def test_croston_matches_hand_calculation():
    # first demand at index 1 -> z=4, p=2; zeros push q to 3; demand 2 arrives:
    # z = 4 + .1(2-4) = 3.8, p = 2 + .1(3-2) = 2.1  ->  rate 3.8 / 2.1
    y = [0, 4, 0, 0, 2]
    assert m.croston_rate(y, alpha=0.1) == pytest.approx(3.8 / 2.1)
    assert m.croston(y, 3, alpha=0.1) == pytest.approx(np.full(3, 3.8 / 2.1))


def test_croston_sba_applies_bias_correction():
    y = [0, 4, 0, 0, 2]
    assert m.croston_rate(y, 0.1, "sba") == pytest.approx(m.croston_rate(y, 0.1) * 0.95)


def test_croston_edge_cases():
    assert m.croston_rate([0, 0, 0, 0]) == 0.0
    # demand every period -> rate equals the constant demand
    assert m.croston_rate([5, 5, 5, 5, 5]) == pytest.approx(5.0)


def test_croston_recovers_rate_of_deterministic_intermittent_series():
    # a deterministic intermittent pattern: demand 6 every 3rd month -> true rate 2
    y = [0, 0, 6] * 10
    assert m.croston_rate(y, alpha=0.1) == pytest.approx(2.0, rel=0.05)


def test_baseline_formulas():
    y = np.array([10.0, 20.0, 30.0, 40.0])
    assert m.naive(y, 2).tolist() == [40.0, 40.0]
    assert m.moving_average(y, 1)[0] == pytest.approx(30.0)  # (20+30+40)/3
    assert m.weighted_moving_average(y, 1)[0] == pytest.approx(0.5 * 40 + 0.3 * 30 + 0.2 * 20)
    # SES alpha=.3 initialised at y[0]: 10 -> 13 -> 18.1 -> 24.67
    assert m.ses(y, 1)[0] == pytest.approx(24.67)


def test_linear_trend_extrapolates_a_line_exactly():
    y = 5 + 2 * np.arange(10)
    assert m.linear_trend(y, 3) == pytest.approx([25, 27, 29])


def test_forecasts_are_never_negative():
    y = np.array([50.0, 40.0, 30.0, 20.0, 10.0, 1.0])
    assert (m.linear_trend(y, 6) >= 0).all()


def test_holt_winters_recovers_seasonal_pattern():
    t = np.arange(48)
    y = 100 + 20 * np.sin(2 * np.pi * t / 12)
    pred = m.holt_winters_add(y, 12)
    truth = 100 + 20 * np.sin(2 * np.pi * np.arange(48, 60) / 12)
    assert np.mean(np.abs(pred - truth)) < 3.0
    # and clearly beats the plain moving average on a strongly seasonal series
    assert np.mean(np.abs(pred - truth)) < np.mean(np.abs(m.moving_average(y, 12) - truth))

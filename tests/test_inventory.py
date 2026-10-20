import numpy as np
import pandas as pd
import pytest

from scm_forecast import inventory as inv
from scm_forecast.simulation import CostParams, simulate_sku


def test_eoq_formula():
    # sqrt(2 * 1200 * 50 / 2) = sqrt(60000)
    assert inv.eoq(1200, 50, 2) == pytest.approx(np.sqrt(60000))
    with pytest.raises(ValueError):
        inv.eoq(1200, 50, 0)


def test_eoq_increases_with_demand_and_decreases_with_holding_cost():
    assert inv.eoq(2400, 50, 2) > inv.eoq(1200, 50, 2)
    assert inv.eoq(1200, 50, 4) < inv.eoq(1200, 50, 2)


def test_z_score_known_service_levels():
    assert inv.z_score(0.95) == pytest.approx(1.6449, abs=1e-3)
    assert inv.z_score(0.90) == pytest.approx(1.2816, abs=1e-3)
    assert inv.z_score(0.99) == pytest.approx(2.3263, abs=1e-3)


def test_safety_stock_classic_matches_formula():
    # sigma_d = 30/month -> 30/sqrt(30) per day, LT = 16 days
    z = 1.65
    expected = z * (30 / np.sqrt(30)) * np.sqrt(16)
    assert inv.safety_stock_classic(z, 30, 16) == pytest.approx(expected)


def test_safety_stock_forecast_error_matches_formula_and_lead_time_term():
    z, se, lt, d, slt = 1.645, 20.0, 15.0, 5.0, 3.0
    expected = z * np.sqrt(lt / 30 * se**2 + (d * slt) ** 2)
    assert inv.safety_stock_forecast_error(z, se, lt, d, slt) == pytest.approx(expected)
    # without LT variability it reduces to the pure demand-error term
    assert inv.safety_stock_forecast_error(z, se, lt) == pytest.approx(z * se * np.sqrt(lt / 30))
    # lower forecast error -> lower safety stock
    assert inv.safety_stock_forecast_error(z, 10, lt) < inv.safety_stock_forecast_error(z, 20, lt)


def test_reorder_point():
    assert inv.reorder_point(daily_demand=10, lead_time_days=7, safety_stock=25) == pytest.approx(95)


def test_abc_xyz_classification():
    value = pd.Series({"a": 800, "b": 100, "c": 60, "d": 25, "e": 15})
    abc = inv.abc_classification(value)
    assert abc["a"] == "A"
    assert abc["b"] == "B"
    assert abc["e"] == "C"
    xyz = inv.xyz_classification(pd.Series({"p": 0.1, "q": 0.4, "r": 0.9}))
    assert xyz.to_dict() == {"p": "X", "q": "Y", "r": "Z"}


def test_simulation_accounts_for_demand_and_is_deterministic():
    daily = np.full(360, 5)
    lts = np.full(50, 7)
    s = np.full(12, 5 * 7 + 20.0)
    q = np.full(12, 150)
    kw = dict(unit_cost=10.0, costs=CostParams())
    r1 = simulate_sku(daily, lts, s, q, **kw)
    r2 = simulate_sku(daily, lts, s, q, **kw)
    assert r1 == r2
    assert r1["demand_units"] == 1800
    assert r1["fill_rate"] == pytest.approx(1.0)  # ample safety stock, constant demand
    assert r1["total_cost"] == pytest.approx(
        r1["ordering_cost"] + r1["holding_cost"] + r1["shortage_cost"]
    )
    # with no safety stock and no reorder point the policy must stock out
    bad = simulate_sku(daily, lts, np.zeros(12), np.full(12, 10), **kw)
    assert bad["fill_rate"] < 1.0

"""Inventory policy building blocks: ABC-XYZ, EOQ, safety stock, reorder point.

Units: demand in units per month, lead time in days, costs in currency units.
A month is treated as ``DAYS_PER_MONTH`` days (360-day year) throughout.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

DAYS_PER_MONTH = 30


def z_score(service_level: float) -> float:
    """Normal quantile for a cycle service level (e.g. 0.95 -> 1.645)."""
    if not 0.5 <= service_level < 1.0:
        raise ValueError("service_level must be in [0.5, 1)")
    return float(norm.ppf(service_level))


def eoq(annual_demand: float, order_cost: float, holding_cost_per_unit_year: float) -> float:
    """Economic order quantity: sqrt(2 D S / H)."""
    if annual_demand < 0 or order_cost < 0:
        raise ValueError("demand and order cost must be non-negative")
    if holding_cost_per_unit_year <= 0:
        raise ValueError("holding cost must be positive")
    return float(np.sqrt(2.0 * annual_demand * order_cost / holding_cost_per_unit_year))


def safety_stock_classic(
    z: float, sigma_demand_month: float, lead_time_days: float, days_per_month: int = DAYS_PER_MONTH
) -> float:
    """Textbook safety stock from demand variability: SS = z * sigma_d * sqrt(LT).

    ``sigma_demand_month`` is the std. dev. of monthly *demand* (not of forecast
    error); it is converted to a per-day figure before scaling by sqrt(LT).
    """
    sigma_daily = sigma_demand_month / np.sqrt(days_per_month)
    return float(z * sigma_daily * np.sqrt(lead_time_days))


def safety_stock_forecast_error(
    z: float,
    sigma_error_month: float,
    lead_time_days: float,
    daily_demand: float = 0.0,
    sigma_lead_time_days: float = 0.0,
    days_per_month: int = DAYS_PER_MONTH,
) -> float:
    """Safety stock sized on one-step forecast error plus lead-time variability.

    SS = z * sqrt( LT/30 * sigma_e^2 + (d * sigma_LT)^2 )

    where ``sigma_e`` is the RMSE of monthly one-step-ahead forecast errors, ``d``
    the forecast daily demand and ``sigma_LT`` the std. dev. of lead time (days).
    Using forecast error (not raw demand spread) means that a model that
    explains trend/seasonality is rewarded with a smaller buffer.
    """
    var = lead_time_days / days_per_month * sigma_error_month**2 + (
        daily_demand * sigma_lead_time_days
    ) ** 2
    return float(z * np.sqrt(var))


def reorder_point(daily_demand: float, lead_time_days: float, safety_stock: float) -> float:
    """ROP = d * LT + SS."""
    return float(daily_demand * lead_time_days + safety_stock)


# --------------------------------------------------------------------------- classification
def abc_classification(annual_value: pd.Series, a_cut: float = 0.80, b_cut: float = 0.95) -> pd.Series:
    """ABC by cumulative share of annual consumption value (Pareto).

    The SKU that crosses a cut-off stays in the higher class, so class A is the
    smallest set of SKUs covering at least ``a_cut`` of total value.
    """
    v = annual_value.sort_values(ascending=False)
    share_before = v.cumsum() / v.sum() - v / v.sum()
    cls = np.where(share_before < a_cut, "A", np.where(share_before < b_cut, "B", "C"))
    return pd.Series(cls, index=v.index).reindex(annual_value.index)


def xyz_classification(cv: pd.Series, x_cut: float = 0.30, y_cut: float = 0.60) -> pd.Series:
    """XYZ by coefficient of variation of monthly demand."""
    return pd.Series(
        np.where(cv < x_cut, "X", np.where(cv < y_cut, "Y", "Z")), index=cv.index
    )


DEFAULT_SERVICE_LEVELS = {"A": 0.98, "B": 0.95, "C": 0.90}


def build_sku_policy_inputs(
    history: pd.DataFrame,
    sku_master: pd.DataFrame,
    window: int = 24,
    service_levels: dict[str, float] | None = None,
    order_cost: float = 40.0,
    holding_rate: float = 0.25,
) -> pd.DataFrame:
    """Static per-SKU inputs: ABC/XYZ class, target service level, costs.

    ABC uses the last 12 months of demand valued at unit cost; XYZ uses the CV of
    the last ``window`` months.
    """
    sl_map = service_levels or DEFAULT_SERVICE_LEVELS
    last12 = history.iloc[:, -12:]
    value = last12.sum(axis=1) * sku_master["unit_cost"]
    cv = history.iloc[:, -window:].std(axis=1, ddof=1) / history.iloc[:, -window:].mean(axis=1)
    out = pd.DataFrame(
        {
            "category": sku_master["category"],
            "pattern": sku_master["pattern"],
            "unit_cost": sku_master["unit_cost"],
            "annual_demand": last12.sum(axis=1),
            "annual_value": value,
            "cv": cv,
        }
    )
    out["abc"] = abc_classification(value)
    out["xyz"] = xyz_classification(cv)
    out["abc_xyz"] = out["abc"] + out["xyz"]
    out["service_level"] = out["abc"].map(sl_map)
    out["z"] = out["service_level"].map(z_score)
    out["order_cost"] = order_cost
    out["holding_cost"] = holding_rate * out["unit_cost"]
    out["lt_mean"] = sku_master["lt_mean"]
    out["lt_sd"] = sku_master["lt_sd"]
    out["eoq"] = [
        eoq(d, order_cost, h) for d, h in zip(out["annual_demand"], out["holding_cost"])
    ]
    return out

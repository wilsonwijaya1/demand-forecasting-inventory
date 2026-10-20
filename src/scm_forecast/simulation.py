"""Daily (s, Q) inventory simulation used to compare forecasting policies.

Design
------
* Horizon: the 12 held-out months (year 6), 30 days each.
* Daily demand: each month's actual demand is spread over its 30 days with a
  multinomial draw (seeded, identical across the policies being compared).
* Policy: continuous review, order ``Q`` whenever the inventory position
  (on hand + on order) is at or below the reorder point ``s``.
* Parameters are refreshed at the start of every month from that month's
  one-step-ahead forecast, so a better forecast changes ``s`` and ``Q`` directly.
* Lead times: drawn per order from a normal distribution (rounded, >= 1 day)
  using the SKU's observed mean / std. The same draw sequence is used for every
  policy (common random numbers), so differences come from the policy only.
* Unmet demand is lost (no backorders) and incurs a shortage cost.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .inventory import (
    DAYS_PER_MONTH,
    eoq,
    reorder_point,
    safety_stock_classic,
    safety_stock_forecast_error,
)


@dataclass(frozen=True)
class CostParams:
    order_cost: float = 40.0
    holding_rate: float = 0.25  # of unit cost per year
    shortage_ratio: float = 0.50  # lost-sale penalty as a fraction of unit cost


def make_daily_demand(monthly: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pieces = [
        rng.multinomial(int(m), np.full(DAYS_PER_MONTH, 1.0 / DAYS_PER_MONTH)) for m in monthly
    ]
    return np.concatenate(pieces)


def make_lead_times(lt_mean: float, lt_sd: float, n: int, rng: np.random.Generator) -> np.ndarray:
    return np.maximum(np.rint(rng.normal(lt_mean, max(lt_sd, 0.0), size=n)), 1).astype(int)


def monthly_parameters(
    policy: str,
    forecast: float,
    sigma_error: float,
    sigma_demand: float,
    z: float,
    lt_mean: float,
    lt_sd: float,
    order_cost: float,
    holding_cost: float,
) -> tuple[float, int, float]:
    """Return (reorder point s, order quantity Q, safety stock) for one SKU-month.

    policy = "classic": SS = z * sigma_demand/sqrt(30) * sqrt(LT)
    policy = "error_based": SS = z * sqrt(LT/30 * sigma_e^2 + (d * sigma_LT)^2)
    """
    f = max(float(forecast), 0.0)
    d = f / DAYS_PER_MONTH
    if policy == "classic":
        ss = safety_stock_classic(z, sigma_demand, lt_mean)
    elif policy == "error_based":
        ss = safety_stock_forecast_error(z, sigma_error, lt_mean, d, lt_sd)
    else:
        raise ValueError(policy)
    s = reorder_point(d, lt_mean, ss)
    q = max(1, int(np.ceil(eoq(12.0 * f, order_cost, holding_cost)))) if f > 0 else 1
    return float(s), q, float(ss)


def simulate_sku(
    daily_demand: np.ndarray,
    lead_times: np.ndarray,
    s_by_month: np.ndarray,
    q_by_month: np.ndarray,
    unit_cost: float,
    costs: CostParams,
) -> dict:
    """Run one SKU through the daily simulation and return KPIs."""
    n_days = len(daily_demand)
    on_hand = int(np.ceil(s_by_month[0])) + int(q_by_month[0])
    pipeline: list[tuple[int, int]] = []  # (arrival day, qty)
    n_orders = 0
    hold_unit_days = 0.0
    lost = 0
    total_demand = int(daily_demand.sum())
    stockout_days = 0
    inv_sum = 0.0
    for day in range(n_days):
        m = day // DAYS_PER_MONTH
        s, q = s_by_month[m], int(q_by_month[m])
        if pipeline:
            arrived = [p for p in pipeline if p[0] <= day]
            if arrived:
                on_hand += sum(p[1] for p in arrived)
                pipeline = [p for p in pipeline if p[0] > day]
        d = int(daily_demand[day])
        sold = min(on_hand, d)
        if sold < d:
            lost += d - sold
            stockout_days += 1
        on_hand -= sold
        position = on_hand + sum(p[1] for p in pipeline)
        while position <= s:
            lt = int(lead_times[min(n_orders, len(lead_times) - 1)])
            pipeline.append((day + lt, q))
            position += q
            n_orders += 1
        hold_unit_days += on_hand
        inv_sum += on_hand
    holding = costs.holding_rate * unit_cost * hold_unit_days / (12 * DAYS_PER_MONTH)
    ordering = costs.order_cost * n_orders
    shortage = costs.shortage_ratio * unit_cost * lost
    fill = 1.0 if total_demand == 0 else 1.0 - lost / total_demand
    return dict(
        ordering_cost=ordering,
        holding_cost=holding,
        shortage_cost=shortage,
        total_cost=ordering + holding + shortage,
        fill_rate=fill,
        stockout_days=stockout_days,
        lost_units=lost,
        demand_units=total_demand,
        orders=n_orders,
        avg_inventory_units=inv_sum / n_days,
        avg_inventory_value=inv_sum / n_days * unit_cost,
    )


def rolling_sigma(errors: pd.Series, k: int, window: int = 24) -> float:
    """RMSE of the most recent ``window`` errors that are known before month k."""
    e = errors[errors.index < k].tail(window).to_numpy(float)
    if len(e) == 0:
        return 0.0
    return float(np.sqrt(np.mean(e**2)))

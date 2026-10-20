"""Synthetic demand generator for 50 SKUs.

Everything here is produced from a seeded random number generator. No real
or course-provided data is used. The layout (SKU rows x monthly columns,
5 years of history, a separate monthly lead-time table) mirrors the shape of a
typical spreadsheet-based forecasting exercise, so results are easy to compare
with a spreadsheet workflow.

Demand patterns (counts are fixed so the portfolio is always diverse):

==================  ===========================================================
pattern             description
==================  ===========================================================
stable              flat level, low noise
trend_up            linear growth
trend_down          linear decline (product run-out)
seasonal            annual cycle, no trend
trend_seasonal      annual cycle on top of a linear trend
intermittent        many zero months, small lumpy order sizes
volatile            high noise plus random promotional spikes
==================  ===========================================================
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

N_SKU = 50
N_HISTORY = 60  # 5 years of observed history ("known" at planning time)
N_HOLDOUT = 12  # year 6, only used to evaluate forecasts and simulate inventory
START_DATE = "2020-01-01"

CATEGORIES = [
    "Grocery",
    "Household",
    "Electronics",
    "Apparel",
    "Health & Beauty",
]

PATTERN_COUNTS = {
    "stable": 8,
    "trend_up": 7,
    "trend_down": 3,
    "seasonal": 9,
    "trend_seasonal": 10,
    "intermittent": 7,
    "volatile": 6,
}

# (median unit cost, log-sd) per category
_UNIT_COST = {
    "Grocery": (4.0, 0.45),
    "Household": (12.0, 0.45),
    "Electronics": (85.0, 0.5),
    "Apparel": (25.0, 0.45),
    "Health & Beauty": (9.0, 0.45),
}

# (min, max) mean replenishment lead time in days per category
_LEAD_TIME = {
    "Grocery": (3, 10),
    "Household": (7, 18),
    "Electronics": (21, 35),
    "Apparel": (14, 30),
    "Health & Beauty": (7, 21),
}


@dataclass(frozen=True)
class SyntheticDataset:
    """Container for everything the generator produces."""

    demand: pd.DataFrame  # index: sku, columns: month start (72 columns)
    sku_master: pd.DataFrame  # index: sku
    lead_times: pd.DataFrame  # index: sku, columns M1..M12 (days)
    n_history: int
    n_holdout: int

    @property
    def history(self) -> pd.DataFrame:
        return self.demand.iloc[:, : self.n_history]

    @property
    def holdout(self) -> pd.DataFrame:
        return self.demand.iloc[:, self.n_history :]


def _seasonal_profile(t: np.ndarray, amp: float, phase: float, amp2: float) -> np.ndarray:
    base = 1.0 + amp * np.sin(2 * np.pi * (t - phase) / 12.0)
    harm = amp2 * np.sin(4 * np.pi * (t - phase) / 12.0)
    return base + harm


def _make_series(pattern: str, t: np.ndarray, rng: np.random.Generator, cat_phase: float):
    """Return (series, info) for one SKU. ``info`` holds generator parameters."""
    n = len(t)
    info: dict = {}
    if pattern == "intermittent":
        p = rng.uniform(0.35, 0.60)  # probability of a non-zero month
        lam = rng.uniform(2.0, 7.0)  # mean extra units on a demand month
        occurs = rng.random(n) < p
        sizes = rng.poisson(lam, size=n) + 1
        y = np.where(occurs, sizes, 0).astype(float)
        info.update(base_level=float(p * (lam + 1)), noise_cv=np.nan)
        return y, info

    base = float(np.exp(rng.normal(np.log(160.0), 0.65)))
    base = float(np.clip(base, 30.0, 900.0))
    trend = np.ones(n)
    season = np.ones(n)
    noise_cv = 0.10
    spikes = False

    if pattern == "stable":
        noise_cv = rng.uniform(0.05, 0.14)
    elif pattern == "trend_up":
        g = rng.uniform(0.006, 0.015)
        trend = 1.0 + g * t
        noise_cv = rng.uniform(0.06, 0.14)
        info["trend_per_month"] = float(g)
    elif pattern == "trend_down":
        g = -rng.uniform(0.004, 0.008)
        trend = 1.0 + g * t
        noise_cv = rng.uniform(0.06, 0.12)
        info["trend_per_month"] = float(g)
    elif pattern == "seasonal":
        season = _seasonal_profile(
            t, rng.uniform(0.18, 0.40), cat_phase + rng.normal(0, 0.6), rng.uniform(0.0, 0.08)
        )
        noise_cv = rng.uniform(0.05, 0.12)
    elif pattern == "trend_seasonal":
        g = rng.uniform(0.003, 0.010)
        trend = 1.0 + g * t
        season = _seasonal_profile(
            t, rng.uniform(0.15, 0.35), cat_phase + rng.normal(0, 0.6), rng.uniform(0.0, 0.06)
        )
        noise_cv = rng.uniform(0.06, 0.12)
        info["trend_per_month"] = float(g)
    elif pattern == "volatile":
        noise_cv = rng.uniform(0.25, 0.40)
        spikes = True
    else:  # pragma: no cover - guarded by PATTERN_COUNTS
        raise ValueError(f"unknown pattern {pattern!r}")

    mu = base * trend * season
    eps = np.clip(rng.normal(0.0, 1.0, size=n), -2.5, 2.5)
    y = mu * (1.0 + noise_cv * eps)
    if spikes:
        hit = rng.random(n) < 0.08
        y = np.where(hit, y * rng.uniform(1.6, 2.2, size=n), y)
    y = np.maximum(np.round(y), 0.0)
    info.update(base_level=base, noise_cv=float(noise_cv))
    return y, info


def generate_dataset(
    seed: int = 42, n_history: int = N_HISTORY, n_holdout: int = N_HOLDOUT
) -> SyntheticDataset:
    """Generate the 50-SKU synthetic dataset. Fully determined by ``seed``."""
    rng = np.random.default_rng(seed)
    n_months = n_history + n_holdout
    t = np.arange(n_months, dtype=float)

    categories = np.repeat(CATEGORIES, N_SKU // len(CATEGORIES))
    patterns = np.array(
        [p for p, c in PATTERN_COUNTS.items() for _ in range(c)], dtype=object
    )
    assert len(patterns) == N_SKU
    patterns = patterns[rng.permutation(N_SKU)]
    cat_phase = {c: rng.uniform(0, 12) for c in CATEGORIES}

    skus = [f"SKU-{i + 1:03d}" for i in range(N_SKU)]
    rows, series, lt_rows = [], [], []
    for i, sku in enumerate(skus):
        cat, pat = str(categories[i]), str(patterns[i])
        y, info = _make_series(pat, t, rng, cat_phase[cat])
        series.append(y)

        med, sd = _UNIT_COST[cat]
        unit_cost = float(np.round(np.exp(rng.normal(np.log(med), sd)), 2))
        lo, hi = _LEAD_TIME[cat]
        lt_mean = float(rng.uniform(lo, hi))
        lt_sd = float(lt_mean * rng.uniform(0.10, 0.30))
        lt_obs = np.clip(np.round(rng.normal(lt_mean, lt_sd, size=12)), 1, None)
        lt_rows.append(lt_obs)
        rows.append(
            dict(
                sku=sku,
                category=cat,
                pattern=pat,
                unit_cost=unit_cost,
                base_level=info.get("base_level"),
                noise_cv=info.get("noise_cv"),
                trend_per_month=info.get("trend_per_month", 0.0),
            )
        )

    months = pd.date_range(START_DATE, periods=n_months, freq="MS")
    demand = pd.DataFrame(np.vstack(series), index=pd.Index(skus, name="sku"), columns=months)
    demand = demand.astype(int)
    master = pd.DataFrame(rows).set_index("sku")
    lead = pd.DataFrame(
        np.vstack(lt_rows).astype(int),
        index=pd.Index(skus, name="sku"),
        columns=[f"M{k}" for k in range(1, 13)],
    )
    lt_stats = pd.DataFrame(
        {"lt_mean": lead.mean(axis=1), "lt_sd": lead.std(axis=1, ddof=1)}
    )
    master = master.join(lt_stats)
    return SyntheticDataset(demand, master, lead, n_history, n_holdout)


def save_dataset(ds: SyntheticDataset, out_dir) -> None:
    """Write the dataset to CSV files (wide format, one row per SKU)."""
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wide = ds.demand.copy()
    wide.columns = [c.strftime("%Y-%m") for c in wide.columns]
    wide.to_csv(out / "demand_monthly.csv")
    ds.sku_master.round(4).to_csv(out / "sku_master.csv")
    ds.lead_times.to_csv(out / "lead_time_obs.csv")


def demand_profile(history: pd.DataFrame) -> pd.DataFrame:
    """ADI / CV^2 demand-pattern statistics (Syntetos-Boylan style).

    ``ADI`` is the average interval between non-zero demands, ``CV2`` the squared
    coefficient of variation of the non-zero sizes.
    """
    out = []
    for sku, row in history.iterrows():
        y = row.to_numpy(float)
        nz = y[y > 0]
        adi = len(y) / max(len(nz), 1)
        cv2 = (nz.std(ddof=0) / nz.mean()) ** 2 if len(nz) > 1 else 0.0
        if adi < 1.32 and cv2 < 0.49:
            cls = "smooth"
        elif adi < 1.32:
            cls = "erratic"
        elif cv2 < 0.49:
            cls = "intermittent"
        else:
            cls = "lumpy"
        out.append((sku, adi, cv2, cls))
    return pd.DataFrame(out, columns=["sku", "adi", "cv2", "sbc_class"]).set_index("sku")


if __name__ == "__main__":  # python -m scm_forecast.data
    ds = generate_dataset()
    save_dataset(ds, "data/synthetic")
    print(ds.sku_master.pattern.value_counts())

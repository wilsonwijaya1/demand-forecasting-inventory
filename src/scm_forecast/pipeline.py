"""End-to-end pipeline: data -> backtests -> selection -> inventory simulation.

Run ``python scripts/run_pipeline.py`` (or call :func:`run_pipeline`). Results
are written as CSV files to ``reports/results`` and re-used by the notebook.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest as bt
from . import metrics as M
from .data import SyntheticDataset, demand_profile, generate_dataset, save_dataset
from .inventory import build_sku_policy_inputs
from .models import BASELINE_METHODS
from .simulation import (
    CostParams,
    make_daily_demand,
    make_lead_times,
    monthly_parameters,
    rolling_sigma,
    simulate_sku,
)

SELECTION_ORIGINS = range(36, 60)  # 24 rolling origins inside the 60-month history
TEST_ORIGINS = range(60, 72)  # one-step-ahead through the held-out year 6
HORIZON = 3
BASELINE_FORECASTER = "ma3"  # the baseline policy forecaster (3-month moving average)

RESULT_FILES = [
    "backtest_selection_scores.csv",
    "backtest_test_scores.csv",
    "best_method.csv",
    "method_summary_selection.csv",
    "method_summary_test.csv",
    "excel_style_year6.csv",
    "sku_policy_inputs.csv",
    "simulation_sku.csv",
    "simulation_summary.csv",
    "forecasts_test.csv",
    "forecasts_selection.csv",
]


def _naive_scale(Y: pd.DataFrame, n: int) -> pd.Series:
    return pd.Series({sku: M.mase_scale(Y.loc[sku].to_numpy(float)[:n]) for sku in Y.index})


def _selected_rows(fc: pd.DataFrame, best: pd.DataFrame) -> pd.DataFrame:
    m = fc.merge(best[["best_method"]], left_on="sku", right_index=True)
    out = m[m["method"] == m["best_method"]].copy()
    out["method"] = "selected"
    return out.drop(columns="best_method")


def _h1_errors(fc_all: pd.DataFrame, method_by_sku: pd.Series | str) -> dict[str, pd.Series]:
    """One-step-ahead errors by SKU, indexed by target month index."""
    h1 = fc_all[fc_all["h"] == 1]
    out = {}
    for sku, g in h1.groupby("sku"):
        meth = method_by_sku if isinstance(method_by_sku, str) else method_by_sku[sku]
        gg = g[g["method"] == meth].sort_values("target_idx")
        out[sku] = pd.Series((gg["y_pred"] - gg["y_true"]).to_numpy(), index=gg["target_idx"].to_numpy())
    return out


def run_simulation(
    ds: SyntheticDataset,
    fc_sel: pd.DataFrame,
    fc_test: pd.DataFrame,
    best: pd.DataFrame,
    seed: int = 42,
    costs: CostParams | None = None,
    service_levels: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare three policies in a daily (s, Q) simulation over year 6.

    classic_ma3   : MA(3) forecast, safety stock from std. dev. of demand (textbook)
    ma3_errorSS   : MA(3) forecast, safety stock from MA(3) forecast error
    best_errorSS  : per-SKU selected model, safety stock from its forecast error
    """
    costs = costs or CostParams()
    inputs = build_sku_policy_inputs(
        ds.history,
        ds.sku_master,
        service_levels=service_levels,
        order_cost=costs.order_cost,
        holding_rate=costs.holding_rate,
    )
    fc_all = pd.concat([fc_sel, fc_test], ignore_index=True)
    best_method = best["best_method"]
    err_ma3 = _h1_errors(fc_all, BASELINE_FORECASTER)
    err_best = _h1_errors(fc_all, best_method)

    h1_test = fc_test[fc_test["h"] == 1]

    def forecast_matrix(method_by_sku):
        mat = {}
        for sku in ds.demand.index:
            meth = method_by_sku if isinstance(method_by_sku, str) else method_by_sku[sku]
            g = h1_test[(h1_test["sku"] == sku) & (h1_test["method"] == meth)].sort_values("origin")
            mat[sku] = g["y_pred"].to_numpy()
        return mat

    fcm_ma3 = forecast_matrix(BASELINE_FORECASTER)
    fcm_best = forecast_matrix(best_method)

    policies = {
        "classic_ma3": ("classic", fcm_ma3, err_ma3),
        "ma3_errorSS": ("error_based", fcm_ma3, err_ma3),
        "best_errorSS": ("error_based", fcm_best, err_best),
    }
    n_hist = ds.n_history
    rows = []
    for i, sku in enumerate(ds.demand.index):
        rng = np.random.default_rng([seed, 7919, i])
        monthly_actual = ds.holdout.loc[sku].to_numpy()
        daily = make_daily_demand(monthly_actual, rng)
        lts = make_lead_times(inputs.at[sku, "lt_mean"], inputs.at[sku, "lt_sd"], 400, rng)
        hist_all = ds.demand.loc[sku].to_numpy(float)
        for pname, (kind, fcm, errs) in policies.items():
            s_arr, q_arr = np.zeros(12), np.zeros(12, int)
            for k in range(12):
                sigma_d = float(np.std(hist_all[n_hist + k - 12 : n_hist + k], ddof=1))
                sigma_e = rolling_sigma(errs[sku], n_hist + k)
                s, q, _ = monthly_parameters(
                    kind,
                    fcm[sku][k],
                    sigma_e,
                    sigma_d,
                    inputs.at[sku, "z"],
                    inputs.at[sku, "lt_mean"],
                    inputs.at[sku, "lt_sd"],
                    costs.order_cost,
                    inputs.at[sku, "holding_cost"],
                )
                s_arr[k], q_arr[k] = s, q
            kpi = simulate_sku(daily, lts, s_arr, q_arr, inputs.at[sku, "unit_cost"], costs)
            rows.append(
                dict(
                    sku=sku,
                    policy=pname,
                    abc=inputs.at[sku, "abc"],
                    xyz=inputs.at[sku, "xyz"],
                    target_service=inputs.at[sku, "service_level"],
                    avg_reorder_point=float(s_arr.mean()),
                    avg_order_qty=float(q_arr.mean()),
                    **kpi,
                )
            )
    sim = pd.DataFrame(rows)
    return sim, summarise_simulation(sim), inputs


def summarise_simulation(sim: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    keys = ["policy"] + (by or [])
    g = sim.groupby(keys)
    out = g.agg(
        orders=("orders", "sum"),
        ordering_cost=("ordering_cost", "sum"),
        holding_cost=("holding_cost", "sum"),
        shortage_cost=("shortage_cost", "sum"),
        total_cost=("total_cost", "sum"),
        lost_units=("lost_units", "sum"),
        demand_units=("demand_units", "sum"),
        avg_inventory_value=("avg_inventory_value", "sum"),
        mean_sku_fill_rate=("fill_rate", "mean"),
        skus=("sku", "count"),
    )
    out["fill_rate"] = 1.0 - out["lost_units"] / out["demand_units"]
    return out.reset_index()


def run_pipeline(
    seed: int = 42,
    out_dir: str | Path = "reports/results",
    data_dir: str | Path = "data/synthetic",
    use_cache: bool = True,
    n_jobs: int = -1,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run everything and return the result tables (also saved as CSV)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"
    ds = generate_dataset(seed)
    save_dataset(ds, data_dir)

    if use_cache and meta_path.exists() and all((out / f).exists() for f in RESULT_FILES):
        if json.loads(meta_path.read_text()).get("seed") == seed:
            if verbose:
                print("Loading cached results from", out)
            return load_results(out)

    log = print if verbose else (lambda *a, **k: None)
    cat_codes = pd.Categorical(ds.sku_master["category"]).codes
    Y_hist, Y_all = ds.history, ds.demand

    log("1/4 selection backtest (rolling origin, 24 origins x h=1..3, 12 methods)")
    fc_sel = bt.rolling_origin_forecasts(
        Y_hist, SELECTION_ORIGINS, HORIZON, category_codes=cat_codes, n_jobs=n_jobs
    )
    scale_sel = _naive_scale(Y_hist, SELECTION_ORIGINS[0])
    scores_sel = bt.score_forecasts(fc_sel, scale_sel)
    profile = demand_profile(Y_hist)
    best = bt.select_best(scores_sel, sbc_class=profile["sbc_class"])

    log("2/4 held-out year 6 evaluation (12 origins, one-step to 3-step)")
    fc_test = bt.rolling_origin_forecasts(
        Y_all, TEST_ORIGINS, HORIZON, category_codes=cat_codes, n_jobs=n_jobs
    )
    scale_test = _naive_scale(Y_hist, ds.n_history)
    scores_test = bt.score_forecasts(fc_test, scale_test)
    scores_sel_pick = bt.score_forecasts(_selected_rows(fc_test, best), scale_test)
    scores_test_all = pd.concat([scores_test, scores_sel_pick], ignore_index=True)

    log("3/4 single-origin 12-month forecast (spreadsheet-style)")
    fc_x = bt.rolling_origin_forecasts(
        Y_all, [ds.n_history], 12, category_codes=cat_codes, n_jobs=n_jobs
    )
    xs = []
    for (sku, method), g in fc_x.groupby(["sku", "method"]):
        a, f = g["y_true"].to_numpy(), g["y_pred"].to_numpy()
        xs.append(
            dict(
                sku=sku,
                method=method,
                annual_forecast=f.sum(),
                annual_actual=a.sum(),
                mad=M.mae(a, f),
                mape=M.mape(a, f),
                mse=M.mse(a, f),
                smape=M.smape(a, f),
                mase=M.mase(a, f, scale=scale_test[sku]),
            )
        )
    excel_style = pd.DataFrame(xs)

    log("4/4 inventory policy simulation (50 SKUs x 3 policies x 360 days)")
    sim, sim_summary, inputs = run_simulation(ds, fc_sel, fc_test, best, seed=seed)

    summary_sel = bt.aggregate_scores(scores_sel)
    summary_test = bt.aggregate_scores(scores_test_all)
    best = best.join(profile).join(ds.sku_master[["category", "pattern"]])

    results = {
        "backtest_selection_scores.csv": scores_sel,
        "backtest_test_scores.csv": scores_test_all,
        "best_method.csv": best.reset_index(),
        "method_summary_selection.csv": summary_sel.reset_index(),
        "method_summary_test.csv": summary_test.reset_index(),
        "excel_style_year6.csv": excel_style,
        "sku_policy_inputs.csv": inputs.reset_index(),
        "simulation_sku.csv": sim,
        "simulation_summary.csv": sim_summary,
        "forecasts_test.csv": fc_test,
        "forecasts_selection.csv": fc_sel,
    }
    for name, df in results.items():
        df.to_csv(out / name, index=False)
    meta_path.write_text(json.dumps({"seed": seed}))
    return load_results(out)


def load_results(out_dir: str | Path = "reports/results") -> dict[str, pd.DataFrame]:
    out = Path(out_dir)
    return {f: pd.read_csv(out / f) for f in RESULT_FILES}

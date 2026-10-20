"""Matplotlib figures used by the notebook and README (saved to reports/figures)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .models import ADVANCED_METHODS, BASELINE_METHODS, METHOD_LABELS

FIG_DIR = Path("reports/figures")

C_BASE = "#8a8f98"  # grey: baseline
C_ADV = "#1f77b4"  # blue: advanced
C_SEL = "#e4572e"  # orange-red: per-SKU selection / "best"
C_ACT = "#222222"
PALETTE = ["#1f77b4", "#e4572e", "#2ca02c", "#9467bd", "#8c564b", "#17becf", "#bcbd22"]

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
        "font.size": 9,
    }
)


def _save(fig, name: str, fig_dir=None) -> Path:
    d = Path(fig_dir) if fig_dir else FIG_DIR
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    fig.savefig(p, bbox_inches="tight")
    return p


def plot_sample_series(demand: pd.DataFrame, master: pd.DataFrame, fig_dir=None):
    patterns = ["stable", "trend_up", "trend_down", "seasonal", "trend_seasonal", "intermittent", "volatile"]
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.2), sharex=True)
    for ax, pat in zip(axes.ravel(), patterns):
        sku = master.index[master["pattern"] == pat][0]
        ax.plot(demand.columns, demand.loc[sku], color=C_ADV, lw=1.3)
        ax.axvline(demand.columns[60], color=C_SEL, ls="--", lw=1)
        ax.set_title(f"{pat}  ({sku})")
        ax.tick_params(axis="x", labelrotation=45)
    last = axes.ravel()[-1]
    last.axis("off")
    last.text(0.05, 0.5, "dashed line = start of" + chr(10) + "held-out year 6", color=C_SEL, fontsize=10, transform=last.transAxes)
    fig.suptitle("One synthetic SKU per demand pattern (monthly units)", fontweight="bold")
    fig.tight_layout()
    return _save(fig, "01_demand_patterns.png", fig_dir), fig


def plot_adi_cv2(profile: pd.DataFrame, master: pd.DataFrame, fig_dir=None):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for pat, color in zip(master["pattern"].unique(), PALETTE):
        idx = master.index[master["pattern"] == pat]
        ax.scatter(profile.loc[idx, "adi"], profile.loc[idx, "cv2"], label=pat, s=38, color=color, alpha=0.85)
    ax.axvline(1.32, color="k", ls=":", lw=1)
    ax.axhline(0.49, color="k", ls=":", lw=1)
    ax.set_xlabel("ADI (average inter-demand interval, months)")
    ax.set_ylabel("CV$^2$ of non-zero demand sizes")
    ax.set_title("Demand classification (Syntetos-Boylan cut-offs)")
    ax.legend(fontsize=8, frameon=False)
    return _save(fig, "02_adi_cv2.png", fig_dir), fig


def plot_method_mase(summary_sel: pd.DataFrame, summary_test: pd.DataFrame, fig_dir=None):
    order = [m for m in summary_test.sort_values("mean_mase")["method"] if m != "selected"]
    labels = [METHOD_LABELS[m] for m in order]
    sel = summary_sel.set_index("method")["mean_mase"]
    tst = summary_test.set_index("method")["mean_mase"]
    y = np.arange(len(order))

    def color(m):
        return C_BASE if m in BASELINE_METHODS else (C_ADV if m in ADVANCED_METHODS or m == "lightgbm" else C_SEL)

    fig, ax = plt.subplots(figsize=(8.2, 5))
    ax.barh(y + 0.2, [tst[m] for m in order], height=0.38, color=[color(m) for m in order], label="Held-out year 6")
    ax.barh(y - 0.2, [sel[m] for m in order], height=0.38, color=[color(m) for m in order], alpha=0.40,
            label="Selection backtest (lighter)")
    ax.axvline(tst["selected"], color=C_SEL, lw=1.6, ls="--")
    ax.text(tst["selected"] + 0.01, -0.75, f"per-SKU selection {tst['selected']:.2f}", color=C_SEL, va="center")
    ax.axvline(1.0, color="k", lw=0.8)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean MASE over 50 SKUs (lower is better; 1.0 = in-sample naive)")
    ax.set_title("Accuracy by method: grey = baselines, blue = advanced")
    ax.set_ylim(len(order) - 0.5, -1.1)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.93))
    return _save(fig, "03_method_mase.png", fig_dir), fig


def plot_winners(best: pd.DataFrame, fig_dir=None):
    ct = pd.crosstab(best["pattern"], best["best_method"])
    ct = ct[ct.sum().sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    bottom = np.zeros(len(ct))
    for i, col in enumerate(ct.columns):
        ax.bar(ct.index, ct[col], bottom=bottom, label=METHOD_LABELS[col], color=PALETTE[i % len(PALETTE)])
        bottom += ct[col].to_numpy()
    ax.set_ylabel("Number of SKUs")
    ax.set_title("Which method wins the selection backtest, by demand pattern")
    ax.legend(fontsize=8, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.tick_params(axis="x", labelrotation=20)
    return _save(fig, "04_winning_methods.png", fig_dir), fig


def plot_forecast_examples(demand, fc_test, best, master, fig_dir=None):
    pats = ["trend_seasonal", "seasonal", "intermittent", "volatile"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5))
    h1 = fc_test[fc_test["h"] == 1]
    for ax, pat in zip(axes.ravel(), pats):
        sku = master.index[master["pattern"] == pat][0]
        m = best.loc[sku, "best_method"]
        x = demand.columns
        ax.plot(x[36:], demand.loc[sku].iloc[36:], color=C_ACT, lw=1.4, label="Actual")
        for meth, col, ls in (("ma3", C_BASE, "-"), (m, C_SEL, "-")):
            g = h1[(h1.sku == sku) & (h1.method == meth)].sort_values("target_idx")
            ax.plot(x[g.target_idx], g.y_pred, color=col, ls=ls, lw=1.6,
                    label=f"{METHOD_LABELS[meth]}" + (" (selected)" if meth == m else ""))
        ax.axvline(x[60], color="k", ls=":", lw=1)
        ax.set_title(f"{sku}: {pat}")
        ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper left")
        ax.tick_params(axis="x", labelrotation=30)
    fig.suptitle("One-step-ahead forecasts in held-out year 6 (right of dotted line)", fontweight="bold")
    fig.tight_layout()
    return _save(fig, "05_forecast_examples.png", fig_dir), fig


def plot_pattern_improvement(scores_test: pd.DataFrame, master: pd.DataFrame, fig_dir=None):
    s = scores_test.merge(master[["pattern"]], left_on="sku", right_index=True)
    piv = s[s.method.isin(["ma3", "holt_winters_add", "lightgbm", "selected"])].pivot_table(
        index="pattern", columns="method", values="mase", aggfunc="mean"
    )
    piv = piv[["ma3", "holt_winters_add", "lightgbm", "selected"]]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    piv.plot.bar(ax=ax, color=[C_BASE, C_ADV, "#17becf", C_SEL], width=0.8)
    ax.axhline(1, color="k", lw=0.8)
    ax.set_ylabel("Mean MASE (held-out year 6)")
    ax.set_xlabel("")
    ax.set_title("Where the advanced methods help: MASE by demand pattern")
    ax.legend([METHOD_LABELS[c] if c != "selected" else "Per-SKU selection" for c in piv.columns],
              frameon=False, fontsize=8)
    ax.tick_params(axis="x", labelrotation=20)
    return _save(fig, "06_mase_by_pattern.png", fig_dir), fig


def plot_abc_xyz(inputs: pd.DataFrame, fig_dir=None):
    ct = pd.crosstab(inputs["abc"], inputs["xyz"]).reindex(index=list("ABC"), columns=list("XYZ"), fill_value=0)
    val = inputs.groupby(["abc", "xyz"])["annual_value"].sum().unstack().reindex(index=list("ABC"), columns=list("XYZ")).fillna(0)
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.imshow(val.to_numpy(), cmap="Blues")
    ax.grid(False)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{ct.iat[i, j]} SKUs\n{val.iat[i, j] / 1e3:,.0f}k", ha="center", va="center",
                    color="white" if val.iat[i, j] > val.to_numpy().max() * 0.5 else "black")
    ax.set_xticks(range(3), list("XYZ"))
    ax.set_yticks(range(3), list("ABC"))
    ax.set_xlabel("XYZ (demand variability)")
    ax.set_ylabel("ABC (annual value)")
    ax.set_title("ABC-XYZ matrix: SKU count and annual value")
    return _save(fig, "07_abc_xyz.png", fig_dir), fig


POLICY_LABELS = {
    "classic_ma3": "A. Classic\nMA(3) + SS from\ndemand std. dev.",
    "ma3_errorSS": "B. MA(3) + SS\nfrom forecast error",
    "best_errorSS": "C. Best model per SKU\n+ SS from forecast error",
}


def plot_sim_cost(sim_summary: pd.DataFrame, fig_dir=None):
    s = sim_summary.set_index("policy").loc[list(POLICY_LABELS)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    bottom = np.zeros(len(s))
    for col, name, c in (("ordering_cost", "Ordering", "#9ecae1"), ("holding_cost", "Holding", C_ADV),
                         ("shortage_cost", "Shortage (lost sales)", C_SEL)):
        ax.bar(range(len(s)), s[col] / 1e3, bottom=bottom, label=name, color=c)
        bottom += s[col].to_numpy() / 1e3
    for i, tot in enumerate(s["total_cost"] / 1e3):
        ax.text(i, tot + 1.5, f"{tot:,.1f}k", ha="center", fontweight="bold")
    ax.set_xticks(range(len(s)), [POLICY_LABELS[p] for p in s.index], fontsize=8)
    ax.set_ylabel("Total annual cost (thousands)")
    ax.set_title("Cost over the simulated year 6")
    ax.legend(frameon=False)
    ax = axes[1]
    ax.bar(range(len(s)), s["fill_rate"] * 100, color=[C_BASE, C_ADV, C_SEL])
    for i, v in enumerate(s["fill_rate"] * 100):
        ax.text(i, v + 0.05, f"{v:.2f}%", ha="center", fontweight="bold")
    ax.set_ylim(min(s["fill_rate"] * 100) - 1.0, 100.4)
    ax.set_xticks(range(len(s)), ["A", "B", "C"])
    ax.set_ylabel("Fill rate (% of demand units served)")
    ax.set_title("Service level achieved")
    fig.tight_layout()
    return _save(fig, "08_simulation_cost_service.png", fig_dir), fig


def plot_sim_by_class(sim: pd.DataFrame, fig_dir=None):
    g = sim.groupby(["abc", "policy"]).agg(
        cost=("total_cost", "sum"), lost=("lost_units", "sum"), dem=("demand_units", "sum"),
        inv=("avg_inventory_value", "sum"),
    ).reset_index()
    g["fill"] = 1 - g["lost"] / g["dem"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    w = 0.26
    for k, (p, c) in enumerate(zip(POLICY_LABELS, [C_BASE, C_ADV, C_SEL])):
        d = g[g.policy == p].set_index("abc").loc[list("ABC")]
        axes[0].bar(np.arange(3) + (k - 1) * w, d["cost"] / 1e3, w, color=c, label=p)
        axes[1].bar(np.arange(3) + (k - 1) * w, d["fill"] * 100, w, color=c, label=p)
    for ax, yl, t in zip(axes, ["Total cost (thousands)", "Fill rate (%)"], ["Cost by ABC class", "Fill rate by ABC class"]):
        ax.set_xticks(range(3), list("ABC"))
        ax.set_ylabel(yl)
        ax.set_title(t)
    axes[1].set_ylim(95, 100.3)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return _save(fig, "09_simulation_by_abc.png", fig_dir), fig


def plot_robustness(rob: pd.DataFrame, fig_dir=None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    cols = ["mase_ma3", "mase_ses", "mase_holt_winters_add", "mase_lightgbm", "mase_selected"]
    names = ["MA(3)", "SES", "HW add.", "LightGBM", "Selected"]
    axes[0].boxplot([rob[c] for c in cols], tick_labels=names, medianprops=dict(color=C_SEL))
    for i, c in enumerate(cols):
        axes[0].scatter(np.full(len(rob), i + 1), rob[c], color="k", s=14, zorder=3)
    axes[0].set_ylabel("Mean MASE, held-out year 6")
    axes[0].set_title(f"Accuracy across {len(rob)} independent datasets")
    red = (1 - rob["cost_best_errorSS"] / rob["cost_classic_ma3"]) * 100
    red_b = (1 - rob["cost_best_errorSS"] / rob["cost_ma3_errorSS"]) * 100
    axes[1].bar(np.arange(len(rob)) - 0.2, red, 0.4, color=C_SEL, label="vs A. classic")
    axes[1].bar(np.arange(len(rob)) + 0.2, red_b, 0.4, color=C_ADV, label="vs B. MA(3) + error SS")
    axes[1].set_xticks(range(len(rob)), [f"seed {s}" for s in rob["seed"]], fontsize=8)
    axes[1].set_ylabel("Total cost reduction of policy C (%)")
    axes[1].set_title("Inventory cost saving by dataset")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    return _save(fig, "10_robustness.png", fig_dir), fig


def recost(sim: pd.DataFrame, ratio: float, base_ratio: float = 0.50) -> pd.Series:
    """Total cost per policy if a lost sale cost ``ratio`` x unit cost instead of ``base_ratio``.

    Policy parameters do not depend on the shortage penalty, so the simulated
    quantities (orders, stock, lost units) are unchanged and only the shortage
    term needs rescaling.
    """
    g = sim.groupby("policy")[["ordering_cost", "holding_cost", "shortage_cost"]].sum()
    return g["ordering_cost"] + g["holding_cost"] + g["shortage_cost"] * ratio / base_ratio


def plot_shortage_sensitivity(sim: pd.DataFrame, fig_dir=None):
    ratios = np.linspace(0, 1.0, 41)
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    for p, c in zip(POLICY_LABELS, [C_BASE, C_ADV, C_SEL]):
        ax.plot(ratios, [recost(sim, r)[p] / 1e3 for r in ratios], color=c, lw=2,
                label=POLICY_LABELS[p].replace("\n", " "))
    ax.axvline(0.5, color="k", ls=":", lw=1)
    ax.text(0.505, ax.get_ylim()[0], " base case", va="bottom", fontsize=8)
    ax.set_xlabel("Lost-sale penalty (fraction of unit cost)")
    ax.set_ylabel("Total cost (thousands)")
    ax.set_title("Sensitivity of the cost comparison to the shortage penalty")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, "11_shortage_sensitivity.png", fig_dir), fig

"""Re-run the whole study on several independent synthetic datasets (seeds).

Writes reports/results/robustness_seeds.csv. Takes a few minutes per seed.
Usage: python scripts/robustness.py [seed ...]
"""
import sys
import tempfile
from pathlib import Path

import pandas as pd

from scm_forecast.pipeline import run_pipeline

SEEDS = [int(s) for s in sys.argv[1:]] or [42, 1, 2, 3, 4, 5]
rows = []
for seed in SEEDS:
    with tempfile.TemporaryDirectory() as tmp:
        res = run_pipeline(seed=seed, out_dir=Path(tmp) / "res", data_dir=Path(tmp) / "data",
                           use_cache=False, verbose=False)
    t = res["method_summary_test.csv"].set_index("method")
    s = res["simulation_summary.csv"].set_index("policy")
    row = {"seed": seed}
    for m in ["ma3", "ses", "holt_winters_add", "lightgbm", "selected"]:
        row[f"mase_{m}"] = t.at[m, "mean_mase"]
    for p in ["classic_ma3", "ma3_errorSS", "best_errorSS"]:
        row[f"cost_{p}"] = s.at[p, "total_cost"]
        row[f"fill_{p}"] = s.at[p, "fill_rate"]
        row[f"invval_{p}"] = s.at[p, "avg_inventory_value"]
    rows.append(row)
    print(seed, {k: round(v, 3) for k, v in row.items() if k != "seed"}, flush=True)
out = pd.DataFrame(rows)
out.to_csv("reports/results/robustness_seeds.csv", index=False)
print(out.round(3).to_string(index=False))

"""Run the full pipeline: python scripts/run_pipeline.py [--seed 42] [--no-cache]"""
import argparse

from scm_forecast.pipeline import run_pipeline

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    res = run_pipeline(seed=args.seed, use_cache=not args.no_cache)
    print(res["method_summary_test.csv"].round(3).to_string(index=False))
    print(res["simulation_summary.csv"].round(3).to_string(index=False))

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import MetaConfig
from evaluation import compute_equity_curves
from io_periods import load_aligned_periods_from_csv, wide_to_long_periods
from reporting import compute_core_performance_metrics
from selection import select_models_universal_v2
from sweep import run_config_sweep
from testing_utils import make_synthetic_aligned_returns, make_test_config


def _load_inputs(aligned_csv: str | None, cfg: MetaConfig):
    if aligned_csv:
        df = pd.read_csv(aligned_csv)
        df = df.drop_duplicates()
        aligned_returns, _ = load_aligned_periods_from_csv(df, cfg)
        long_df = wide_to_long_periods(df)
        return aligned_returns, long_df

    aligned_returns, long_df, _ = make_synthetic_aligned_returns()
    return aligned_returns, long_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a fast canary sweep + sanity checks.")
    parser.add_argument("--aligned-csv", type=str, default=None, help="Path to aligned period CSV.")
    parser.add_argument("--out-dir", type=str, default="canary_output", help="Output directory.")
    parser.add_argument("--n-configs", type=int, default=5, help="Number of configs to sweep.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for config grid.")
    parser.add_argument("--warmup-periods", type=int, default=5, help="Warmup periods for curves.")
    parser.add_argument("--oos-start-date", type=str, default=None, help="Optional OOS start date.")
    parser.add_argument("--enable-plots", action="store_true", help="Enable plotting in sweep.")
    parser.add_argument("--use-test-config", action="store_true", help="Use test-friendly config defaults.")
    args = parser.parse_args()

    cfg = make_test_config() if args.use_test_config or not args.aligned_csv else MetaConfig()
    aligned_returns, long_df = _load_inputs(args.aligned_csv, cfg)
    if not aligned_returns:
        raise RuntimeError("Aligned returns are empty; cannot run canary sweep.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_out = out_dir / "canary_sweep_results.csv"
    if sweep_out.exists():
        sweep_out.unlink()

    selections = select_models_universal_v2(aligned_returns, cfg)
    curves = compute_equity_curves(
        aligned_returns,
        selections,
        long_df,
        warmup_periods=args.warmup_periods,
        top_n=cfg.top_n_global,
    )
    core_metrics = compute_core_performance_metrics(curves)
    core_metrics.to_csv(out_dir / "canary_core_metrics.csv")

    mean_edge = float((curves["topN_return"] - curves["all_models_return"]).mean())
    print(f"Canary mean edge: {mean_edge:.6f}")

    run_config_sweep(
        aligned_returns,
        long_df,
        cfg,
        str(sweep_out),
        n_configs=args.n_configs,
        seed=args.seed,
        warmup_periods=args.warmup_periods,
        oos_start_date=args.oos_start_date,
        enable_plots=args.enable_plots,
    )
    print(f"Canary sweep results: {sweep_out}")


if __name__ == "__main__":
    main()

import os

import pandas as pd

from config import MetaConfig
from evaluation import (
    compute_equity_curves,
    plot_avg_trade_return_curves,
    plot_drawdown_curves,
    plot_equity_ratio_curve,
    plot_equity_curves,
    plot_return_delta_histogram,
    plot_return_histograms,
    plot_returns_scatter,
    plot_rolling_outperformance_rate,
    plot_rolling_sharpe_sortino,
    plot_trade_quality_distribution,
    plot_trade_quality_rolling_mean,
)
from io_periods import (
    aggregate_daily_periods_to_windows,
    align_to_common_periods_per_ticker,
    build_model_id,
    extract_period_indices,
    find_first_common_monday,
    infer_ticker,
    load_aligned_periods_from_csv,
    parse_date_range,
    wide_to_long_periods,
    write_aligned_periods_to_csv,
)
from scoring import (
    compute_scores_for_ticker,
    compute_scores_for_ticker_v2,
    compute_uniqueness_weights,
    downside_cvar,
    downside_cvar_matrix_v2,
    ema_smooth,
    map_z_to_alpha,
    percentile_ranks_across_models,
    percentile_ranks_across_models_v2,
    robust_std,
    rolling_zscore,
    rolling_zscore_v2,
)
from selection import select_models_universal, select_models_universal_v2
from reporting import (
    compute_core_performance_metrics,
    compute_relative_edge_metrics,
    compute_stability_robustness_metrics,
    compute_trade_quality_metrics,
    compute_significance_metrics,
)
from sweep import build_config_grid_v1, run_config_sweep
from timing_utils import _timer

__all__ = [
    "MetaConfig",
    "aggregate_daily_periods_to_windows",
    "align_to_common_periods_per_ticker",
    "build_config_grid_v1",
    "build_model_id",
    "compute_equity_curves",
    "compute_core_performance_metrics",
    "compute_relative_edge_metrics",
    "compute_stability_robustness_metrics",
    "compute_trade_quality_metrics",
    "compute_significance_metrics",
    "compute_scores_for_ticker",
    "compute_scores_for_ticker_v2",
    "compute_uniqueness_weights",
    "downside_cvar",
    "downside_cvar_matrix_v2",
    "ema_smooth",
    "extract_period_indices",
    "find_first_common_monday",
    "infer_ticker",
    "load_aligned_periods_from_csv",
    "map_z_to_alpha",
    "main",
    "parse_date_range",
    "percentile_ranks_across_models",
    "percentile_ranks_across_models_v2",
    "plot_avg_trade_return_curves",
    "plot_drawdown_curves",
    "plot_equity_ratio_curve",
    "plot_equity_curves",
    "plot_return_delta_histogram",
    "plot_return_histograms",
    "plot_returns_scatter",
    "plot_rolling_outperformance_rate",
    "plot_rolling_sharpe_sortino",
    "plot_trade_quality_distribution",
    "plot_trade_quality_rolling_mean",
    "robust_std",
    "rolling_zscore",
    "rolling_zscore_v2",
    "run_config_sweep",
    "select_models_universal",
    "select_models_universal_v2",
    "wide_to_long_periods",
    "write_aligned_periods_to_csv",
    "_timer",
]


cfg = MetaConfig(
    # You can tweak these quickly:
    top_n_global=20,
    vol_window=4,
    momentum_lookback=12,
    conf_lookback=12,
    risk_lookback=20,
    enable_uniqueness_weighting=False,  # keep causal by avoiding future-based uniqueness weights
    per_ticker_cap=10,  # prevents one ticker from dominating global topN
)

#aligned_file_path = r"C:\Users\micha\myhome\algo\artifacts\period_returns\period_returns_weeks_2_aligned.csv"
aligned_file_path = r"C:\Users\micha\myhome\algo\artifacts\period_returns\period_returns_weeks_2_aligned.csv"
if not os.path.exists(aligned_file_path):
    raise FileNotFoundError(
        f"Aligned file not found: {aligned_file_path}. Run align_period_returns.py first."
    )

aligned_df = pd.read_csv(aligned_file_path)
aligned_df = aligned_df.drop_duplicates()
print(f"Loaded aligned df: {aligned_df.shape} rows/cols")

aligned_returns, aligned_meta = load_aligned_periods_from_csv(aligned_df, cfg)
print(f"Tickers kept after alignment load: {len(aligned_returns)}")
for tkr, mat in aligned_returns.items():
    print(f"  {tkr}: models={mat.shape[0]}, common_periods={mat.shape[1]}")

if not aligned_returns:
    raise RuntimeError("No tickers survived alignment filters. Lower min_models_per_ticker or require_common_periods.")

long_df = wide_to_long_periods(aligned_df)
print(f"Aligned long df: {long_df.shape} rows (model-period records)")

out_path = r"C:\Users\micha\myhome\algo\artifacts\period_returns\meta_config_sweep_results.csv"
run_config_sweep(
    aligned_returns,
    long_df,
    cfg,
    out_path,
    n_configs=100,
    seed=42,
    warmup_periods=20,
    oos_start_date="2024-11-09",
)

selections = select_models_universal_v2(aligned_returns, cfg)
print(f"Selections: {selections.shape}")

out_path = r"C:\Users\micha\myhome\algo\artifacts\period_returns\meta_selections_universal_no_ml.csv"
selections.to_csv(out_path, index=False)
print(f"Saved selections to: {out_path}")

curves = compute_equity_curves(aligned_returns, selections, long_df, warmup_periods=20, top_n=cfg.top_n_global)
curves_out = aligned_file_path.replace(".csv", "_equity_curves.csv")
curves.to_csv(curves_out, index=False)
print(f"Saved equity curves to: {curves_out}")

plot_out = aligned_file_path.replace(".csv", "_equity_curves.png")
plot_equity_curves(curves, plot_out)
print(f"Saved equity curve plot to: {plot_out}")

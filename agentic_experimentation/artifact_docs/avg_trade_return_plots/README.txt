# avg_trade_return_plots (per-config artifacts) - Overview

This folder is produced by config sweeps (`sweep.run_config_sweep`, as driven by `adaptive_vol_momentum.py`).
It contains per-`config_id` diagnostics (plots + per-config metric tables) to help you understand *why* a
given parameter set performed well or poorly.

Directory:
- `<agentic_output_root>/run_0/avg_trade_return_plots/`

Mapping to configs:
- Each artifact filename ends with `_config_XXX` where `XXX` is the zero-padded `config_id`.
- `config_id` corresponds to the row in `<agentic_output_root>/run_0/meta_config_sweep_results.csv`.
- To inspect one parameter set, find its `config_id` row in `meta_config_sweep_results.csv`, then open the
  matching artifacts in this folder.

Below: one section per artifact type.

## avg_trade_return_config_XXX.png (PNG)
- What it stores: time-series of average return per trade for `all_models` vs `topN` over evaluation periods.
- Used for: Quick "shape" check (is top-N consistently better, or only in a few regimes?).
- How to interpret:
  - `topN_avg_return_per_trade` above `all_avg_return_per_trade` suggests improved trade outcomes.
  - Focus on stability (less whipsaw) in addition to level.
- Granularity: one PNG per `config_id`.

## trade_quality_hist_config_XXX.png (PNG)
- What it stores: histogram of `all_avg_return_per_trade` vs `topN_avg_return_per_trade` across periods.
- Used for: Distributional view of per-period trade quality (fat tails, skew, downside).
- How to interpret:
  - Right-shifted top-N distribution is generally better.
  - Watch for heavier left tail (worse downside) even if mean improves.
- Granularity: one PNG per `config_id`.

## trade_quality_rollmean_config_XXX.png (PNG)
- What it stores: rolling mean of `all_avg_return_per_trade` and `topN_avg_return_per_trade`.
- Used for: Stability/regime view of trade quality improvements.
- How to interpret: more stable and consistently higher top-N rolling mean is better.
- Granularity: one PNG per `config_id`.

## equity_ratio_config_XXX.png (PNG)
- What it stores: time-series of `equity_topN / equity_all` (Top-N equity divided by All-models equity).
- Used for: "Does Top-N pull away over time?" view; complements return-based charts.
- How to interpret:
  - Above 1.0 means Top-N has higher equity than All-models at that time.
  - Smooth, rising ratio is generally better than spiky/unstable.
- Granularity: one PNG per `config_id`.

## rolling_sharpe_sortino_config_XXX.png (PNG)
- What it stores: rolling Sharpe + Sortino for period returns of All-models vs Top-N.
- Used for: Risk-adjusted performance stability over time.
- How to interpret:
  - Higher is better, but also prefer fewer extreme collapses.
  - Sharpe uses total volatility; Sortino uses downside volatility.
- Granularity: one PNG per `config_id`.

## rolling_outperformance_config_XXX.png (PNG)
- What it stores: rolling fraction of periods where `topN_return > all_models_return`.
- Used for: Consistency metric; helps distinguish “few big wins” vs “many small wins”.
- How to interpret:
  - 0.50 is “coinflip”; >0.50 suggests top-N outperforms more often than not.
  - Prefer both high average and a healthy minimum.
- Granularity: one PNG per `config_id`.

## drawdown_curves_config_XXX.png (PNG)
- What it stores: drawdown curves for All-models and Top-N (drawdown from each series’ running equity peak).
- Used for: Downside/risk shape; complements Sharpe/Sortino.
- How to interpret: shallower drawdowns (closer to 0) and shorter drawdown episodes are better.
- Granularity: one PNG per `config_id`.

## return_hist_config_XXX.png (PNG)
- What it stores: histogram (and KDE overlay when available) of period returns for All-models vs Top-N.
- Used for: Distributional view of period returns (skew, tail risk, central tendency).
- How to interpret:
  - Right-shift is generally better; check the left tail for hidden regressions.
  - If distributions overlap heavily, improvements may be weak/unstable.
- Granularity: one PNG per `config_id`.

## return_delta_hist_config_XXX.png (PNG)
- What it stores: histogram of `topN_return - all_models_return` per period.
- Used for: Direct view of relative edge distribution.
- How to interpret:
  - More mass to the right of 0 indicates more frequent/improved outperformance.
  - Watch for a large negative tail (rare but severe underperformance).
- Granularity: one PNG per `config_id`.

## return_scatter_config_XXX.png (PNG)
- What it stores: scatter of (All-models return, Top-N return) per period with quadrant lines and y=x diagonal.
- Used for: Regime clustering and outperformance regions.
- How to interpret:
  - Points above the y=x line indicate Top-N beat All-models in that period.
  - Upper-left quadrant (baseline negative, top-N positive) is especially valuable.
- Granularity: one PNG per `config_id`.

## core_metrics_config_XXX.csv (CSV)
- What it stores: core performance metric table for All-models vs Top-N (mean/vol, Sharpe, Sortino, drawdowns, etc.).
- Used for: Compact per-config summary of "what changed" at the portfolio level.
- Column/row definitions: `agentic_experimentation/artifact_docs/avg_trade_return_plots/core_metrics_config.txt`
- Granularity: one CSV per `config_id`.

## relative_metrics_config_XXX.csv (CSV)
- What it stores: relative edge metrics describing Top-N vs All-models (deltas, capture ratios, equity ratio).
- Used for: Directly assessing the edge of Top-N relative to the baseline universe.
- Column/row definitions: `agentic_experimentation/artifact_docs/avg_trade_return_plots/relative_metrics_config.txt`
- Granularity: one CSV per `config_id`.

## stability_metrics_config_XXX.csv (CSV)
- What it stores: rolling-window stability metrics for All-models, Top-N, and the relative delta series.
- Used for: Quantifying “stability” beyond point estimates (min rolling Sharpe, losing streaks, etc.).
- Column/row definitions: `agentic_experimentation/artifact_docs/avg_trade_return_plots/stability_metrics_config.txt`
- Granularity: one CSV per `config_id`.

## trade_metrics_config_XXX.csv (CSV)
- What it stores: "trade quality" metrics computed from the *per-period average return per trade* series
  (`all_avg_return_per_trade`, `topN_avg_return_per_trade`) plus relative deltas.
- Used for: Understanding whether improvements come from better per-trade outcomes (and their consistency).
- Column/row definitions: `agentic_experimentation/artifact_docs/avg_trade_return_plots/trade_metrics_config.txt`
- Granularity: one CSV per `config_id`.

## significance_metrics_config_XXX.csv (CSV)
- What it stores: statistical tests on the delta series `topN_return - all_models_return` (t-test, sign test, bootstrap).
- Used for: Rough evidence checks (are improvements likely “real” vs noise?).
- Column/row definitions: `agentic_experimentation/artifact_docs/avg_trade_return_plots/significance_metrics_config.txt`
- Granularity: one CSV per `config_id`.

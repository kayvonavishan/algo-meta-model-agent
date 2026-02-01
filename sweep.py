import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from config import MetaConfig
from evaluation import (
    compute_equity_curves,
    plot_avg_trade_return_curves,
    plot_drawdown_curves,
    plot_equity_ratio_curve,
    plot_return_delta_histogram,
    plot_return_histograms,
    plot_returns_scatter,
    plot_rolling_outperformance_rate,
    plot_rolling_sharpe_sortino,
    plot_trade_quality_distribution,
    plot_trade_quality_rolling_mean,
)
from reporting import (
    compute_core_performance_metrics,
    compute_relative_edge_metrics,
    compute_stability_robustness_metrics,
    compute_trade_quality_metrics,
    compute_significance_metrics,
)
from selection import select_models_universal_v2


def _run_scorecard(sweep_path: str) -> None:
    """
    Invoke build_scorecard.py on the given sweep results path.
    Non-fatal: logs and continues on error.
    """
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build_scorecard.py")
    if not os.path.exists(script_path):
        print(f"[scorecard] build_scorecard.py not found at {script_path}; skipping scorecard.")
        return
    try:
        subprocess.run(
            [sys.executable, script_path, "--input", sweep_path],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"[scorecard] Failed to build scorecard: {exc}. stderr: {exc.stderr}")


def _stratified_values(values: List[Any], n: int, rng: np.random.Generator) -> List[Any]:
    if n <= 0:
        return []
    reps = int(np.ceil(n / len(values)))
    arr = (values * reps)[:n]
    rng.shuffle(arr)
    return arr


def build_config_grid_v1(n_configs: int = 100, seed: int = 42) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    vol_windows = [3, 4, 6, 8]
    alpha_pairs = [(0.25, 0.65), (0.30, 0.70), (0.35, 0.75)]
    alpha_smooths = [0.2, 0.3, 0.4]
    momentum_lookbacks = [8, 12, 16]
    delta_weights = [0.0, 0.1, 0.2, 0.3]
    conf_lookbacks = [8, 12, 16]
    risk_lookbacks = [10, 20, 30]
    cvar_alphas = [0.05, 0.10, 0.15]
    cvar_risk_aversions = [0.5, 0.75, 1.0]

    grid = {
        "vol_window": _stratified_values(vol_windows, n_configs, rng),
        "alpha_pair": _stratified_values(alpha_pairs, n_configs, rng),
        "alpha_smooth": _stratified_values(alpha_smooths, n_configs, rng),
        "momentum_lookback": _stratified_values(momentum_lookbacks, n_configs, rng),
        "delta_weight": _stratified_values(delta_weights, n_configs, rng),
        "conf_lookback": _stratified_values(conf_lookbacks, n_configs, rng),
        "risk_lookback": _stratified_values(risk_lookbacks, n_configs, rng),
        "cvar_alpha": _stratified_values(cvar_alphas, n_configs, rng),
        "cvar_risk_aversion": _stratified_values(cvar_risk_aversions, n_configs, rng),
    }

    configs: List[Dict[str, Any]] = []
    for i in range(n_configs):
        alpha_low, alpha_high = grid["alpha_pair"][i]
        configs.append({
            "config_id": i,
            "vol_window": grid["vol_window"][i],
            "alpha_low": alpha_low,
            "alpha_high": alpha_high,
            "alpha_smooth": grid["alpha_smooth"][i],
            "momentum_lookback": grid["momentum_lookback"][i],
            "delta_weight": grid["delta_weight"][i],
            "conf_lookback": grid["conf_lookback"][i],
            "risk_lookback": grid["risk_lookback"][i],
            "cvar_alpha": grid["cvar_alpha"][i],
            "cvar_risk_aversion": grid["cvar_risk_aversion"][i],
        })
    return configs


def _period_key_start_dates(curves: pd.DataFrame) -> pd.Series:
    start_str = curves["period_key"].astype(str).str.split(" to ", n=1, expand=True)[0]
    return pd.to_datetime(start_str, errors="coerce")


def run_config_sweep(
    aligned_returns: Dict[str, pd.DataFrame],
    long_df: pd.DataFrame,
    base_cfg: MetaConfig,
    out_path: str,
    n_configs: int = 100,
    config_limit: Optional[int] = None,
    seed: int = 42,
    warmup_periods: int = 20,
    oos_start_date: Optional[Union[str, pd.Timestamp]] = None,
    enable_plots: bool = True,
    scorecard_every: Optional[int] = None,
) -> None:
    if scorecard_every is None and hasattr(base_cfg, "scorecard_every"):
        scorecard_every = base_cfg.scorecard_every
    configs = build_config_grid_v1(n_configs=n_configs, seed=seed)
    if config_limit is not None:
        limit = int(config_limit)
        if limit < 0:
            raise ValueError(f"config_limit must be >= 0 (got {config_limit})")
        configs = configs[:limit]
    images_dir = os.path.join(os.path.dirname(out_path), "avg_trade_return_plots")
    selections_dir = os.path.join(os.path.dirname(out_path), "selections")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(selections_dir, exist_ok=True)
    for idx, cfg_rec in enumerate(configs):
        cfg = MetaConfig(**{
            **base_cfg.__dict__,
            "vol_window": cfg_rec["vol_window"],
            "alpha_low": cfg_rec["alpha_low"],
            "alpha_high": cfg_rec["alpha_high"],
            "alpha_smooth": cfg_rec["alpha_smooth"],
            "momentum_lookback": cfg_rec["momentum_lookback"],
            "delta_weight": cfg_rec["delta_weight"],
            "conf_lookback": cfg_rec["conf_lookback"],
            "risk_lookback": cfg_rec["risk_lookback"],
            "cvar_alpha": cfg_rec["cvar_alpha"],
            "cvar_risk_aversion": cfg_rec["cvar_risk_aversion"],
        })
        
        row = dict(cfg_rec)
        row["seed"] = seed
        row["top_n_global"] = cfg.top_n_global
        row["include_n_top_tickers"] = cfg.include_n_top_tickers
        row["per_ticker_cap"] = cfg.per_ticker_cap
        row["per_symbol_outer_trial_cap"] = cfg.per_symbol_outer_trial_cap
        row["min_ticker_score"] = cfg.min_ticker_score
        row["oos_start_date"] = "" if oos_start_date is None else str(oos_start_date)
        row["status"] = "ok"
        row["error"] = ""
        row["selections_path"] = ""
        try:
            selections = select_models_universal_v2(aligned_returns, cfg)
            selections_path = os.path.join(
                selections_dir,
                f"meta_selections_config_{cfg_rec['config_id']:03d}.csv.gz",
            )
            selections.to_csv(selections_path, index=False, compression="gzip")
            row["selections_path"] = selections_path
            curves = compute_equity_curves(
                aligned_returns,
                selections,
                long_df,
                warmup_periods=warmup_periods,
                top_n=cfg.top_n_global,
            )
            if enable_plots:
                plot_path = os.path.join(
                    images_dir,
                    f"avg_trade_return_config_{cfg_rec['config_id']:03d}.png",
                )
                plot_avg_trade_return_curves(
                    curves,
                    plot_path,
                    title=f"Config {cfg_rec['config_id']}",
                )
                plot_trade_quality_distribution(
                    curves,
                    os.path.join(
                        images_dir,
                        f"trade_quality_hist_config_{cfg_rec['config_id']:03d}.png",
                    ),
                )
                plot_trade_quality_rolling_mean(
                    curves,
                    os.path.join(
                        images_dir,
                        f"trade_quality_rollmean_config_{cfg_rec['config_id']:03d}.png",
                    ),
                )
                plot_equity_ratio_curve(
                    curves,
                    os.path.join(
                        images_dir,
                        f"equity_ratio_config_{cfg_rec['config_id']:03d}.png",
                    ),
                )
                plot_rolling_sharpe_sortino(
                    curves,
                    os.path.join(
                        images_dir,
                        f"rolling_sharpe_sortino_config_{cfg_rec['config_id']:03d}.png",
                    ),
                )
                plot_rolling_outperformance_rate(
                    curves,
                    os.path.join(
                        images_dir,
                        f"rolling_outperformance_config_{cfg_rec['config_id']:03d}.png",
                    ),
                )
                plot_drawdown_curves(
                    curves,
                    os.path.join(
                        images_dir,
                        f"drawdown_curves_config_{cfg_rec['config_id']:03d}.png",
                    ),
                )
                plot_return_histograms(
                    curves,
                    os.path.join(
                        images_dir,
                        f"return_hist_config_{cfg_rec['config_id']:03d}.png",
                    ),
                )
                plot_return_delta_histogram(
                    curves,
                    os.path.join(
                        images_dir,
                        f"return_delta_hist_config_{cfg_rec['config_id']:03d}.png",
                    ),
                )
                plot_returns_scatter(
                    curves,
                    os.path.join(
                        images_dir,
                        f"return_scatter_config_{cfg_rec['config_id']:03d}.png",
                    ),
                )
            metrics = compute_core_performance_metrics(curves)
            metrics_path = os.path.join(
                images_dir,
                f"core_metrics_config_{cfg_rec['config_id']:03d}.csv",
            )
            metrics.to_csv(metrics_path)
            for scope in metrics.columns:
                series = metrics[scope]
                for key, val in series.to_dict().items():
                    row[f"core_{scope}_{key}"] = val
            rel_metrics = compute_relative_edge_metrics(curves)
            rel_metrics_path = os.path.join(
                images_dir,
                f"relative_metrics_config_{cfg_rec['config_id']:03d}.csv",
            )
            rel_metrics.to_csv(rel_metrics_path)
            rel_series = rel_metrics["relative"] if "relative" in rel_metrics.columns else pd.Series(dtype=float)
            for key, val in rel_series.to_dict().items():
                row[f"rel_{key}"] = val
            stab_metrics = compute_stability_robustness_metrics(curves)
            stab_metrics_path = os.path.join(
                images_dir,
                f"stability_metrics_config_{cfg_rec['config_id']:03d}.csv",
            )
            stab_metrics.to_csv(stab_metrics_path)
            for scope in ["all_models", "topN", "relative"]:
                if scope not in stab_metrics.columns:
                    continue
                series = stab_metrics[scope]
                for key, val in series.to_dict().items():
                    row[f"stab_{scope}_{key}"] = val
            trade_metrics = compute_trade_quality_metrics(curves)
            trade_metrics_path = os.path.join(
                images_dir,
                f"trade_metrics_config_{cfg_rec['config_id']:03d}.csv",
            )
            trade_metrics.to_csv(trade_metrics_path)
            for scope in ["all_models", "topN", "relative"]:
                if scope not in trade_metrics.columns:
                    continue
                series = trade_metrics[scope]
                for key, val in series.to_dict().items():
                    row[f"trade_{scope}_{key}"] = val
            sig_metrics = compute_significance_metrics(curves, seed=seed)
            sig_metrics_path = os.path.join(
                images_dir,
                f"significance_metrics_config_{cfg_rec['config_id']:03d}.csv",
            )
            sig_metrics.to_csv(sig_metrics_path)
            sig_series = sig_metrics["significance"] if "significance" in sig_metrics.columns else pd.Series(dtype=float)
            for key, val in sig_series.to_dict().items():
                row[f"sig_{key}"] = val
            row["mean_all_avg_return_per_trade_pct"] = float(curves["all_avg_return_per_trade"].mean() * 100.0)
            row["mean_topN_avg_return_per_trade_pct"] = float(curves["topN_avg_return_per_trade"].mean() * 100.0)
            if oos_start_date is None:
                oos_curves = curves.tail(21)
            else:
                oos_start = pd.to_datetime(oos_start_date, errors="coerce")
                if pd.isna(oos_start):
                    raise ValueError(f"Invalid oos_start_date: {oos_start_date}")
                starts = _period_key_start_dates(curves)
                oos_curves = curves[starts >= oos_start]
            row["n_oos_periods"] = int(oos_curves.shape[0])
            row["mean_all_avg_return_per_trade_pct_oos"] = float(
                oos_curves["all_avg_return_per_trade"].mean() * 100.0
            )
            row["mean_topN_avg_return_per_trade_pct_oos"] = float(
                oos_curves["topN_avg_return_per_trade"].mean() * 100.0
            )
            row["n_selections"] = int(selections.shape[0])
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
            row["mean_all_avg_return_per_trade_pct"] = np.nan
            row["mean_topN_avg_return_per_trade_pct"] = np.nan
            row["mean_all_avg_return_per_trade_pct_oos"] = np.nan
            row["mean_topN_avg_return_per_trade_pct_oos"] = np.nan
            row["n_oos_periods"] = 0
            row["n_selections"] = 0
        
        out_df = pd.DataFrame([row])
        header = not os.path.exists(out_path)
        out_df.to_csv(out_path, mode="a", header=header, index=False)

        # Periodic scorecard generation
        if scorecard_every is not None and scorecard_every > 0:
            if (idx + 1) % scorecard_every == 0:
                _run_scorecard(out_path)

    # Final scorecard at end (if last batch not aligned to interval)
    if scorecard_every is not None and scorecard_every > 0:
        if len(configs) % scorecard_every != 0:
            _run_scorecard(out_path)

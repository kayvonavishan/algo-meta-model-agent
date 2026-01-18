import argparse
import os
import sys
from typing import Dict, List, Optional

import numpy as np
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
from io_periods import load_aligned_periods_from_csv, wide_to_long_periods
from reporting import (
    compute_core_performance_metrics,
    compute_relative_edge_metrics,
    compute_significance_metrics,
    compute_stability_robustness_metrics,
    compute_trade_quality_metrics,
)
from scoring import compute_scores_for_ticker_v2, percentile_ranks_across_models_v2


DEFAULT_ALIGNED_FILE = r"C:\Users\micha\myhome\algo\artifacts\period_returns\period_returns_weeks_2_aligned.csv"
DEFAULT_SWEEP_RESULTS = r"C:\Users\micha\myhome\algo\artifacts\period_returns\meta_config_sweep_results.csv"
DEFAULT_METRIC = "mean_topN_avg_return_per_trade_pct"
DEFAULT_WARMUP_PERIODS = 20
DEFAULT_OOS_START_DATE = "2024-11-09"


def _config_from_row(row: pd.Series, base_cfg: MetaConfig) -> MetaConfig:
    data = dict(base_cfg.__dict__)
    for key in [
        "vol_window",
        "momentum_lookback",
        "conf_lookback",
        "risk_lookback",
        "top_n_global",
        "per_ticker_cap",
    ]:
        if key in row and pd.notna(row[key]):
            data[key] = int(row[key])
    for key in [
        "alpha_low",
        "alpha_high",
        "alpha_smooth",
        "delta_weight",
        "cvar_alpha",
        "cvar_risk_aversion",
        "min_ticker_score",
    ]:
        if key in row and pd.notna(row[key]):
            data[key] = float(row[key])
    return MetaConfig(**data)


def _normalize_scores(scores_df: pd.DataFrame, method: str) -> pd.DataFrame:
    if method == "percentile":
        arr = percentile_ranks_across_models_v2(scores_df.to_numpy(dtype=float), axis=0)
        return pd.DataFrame(arr, index=scores_df.index, columns=scores_df.columns)
    if method == "none":
        return scores_df
    raise ValueError(f"Unsupported normalization method: {method}")


def _ensemble_scores(
    aligned_returns: Dict[str, pd.DataFrame],
    cfgs: List[MetaConfig],
    normalize: str,
) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for ticker, mat in aligned_returns.items():
        models = mat.index
        periods = None
        sum_arr = None
        count_arr = None
        for cfg in cfgs:
            scores_df, _ = compute_scores_for_ticker_v2(mat, cfg)
            if scores_df.empty:
                continue
            scores_df = scores_df.reindex(index=models)
            scores_df = _normalize_scores(scores_df, normalize)
            if periods is None:
                periods = list(scores_df.columns)
                sum_arr = np.zeros((len(models), len(periods)), dtype=float)
                count_arr = np.zeros_like(sum_arr)
            scores_df = scores_df.reindex(columns=periods)
            arr = scores_df.to_numpy(dtype=float)
            mask = np.isfinite(arr)
            sum_arr[mask] += arr[mask]
            count_arr[mask] += 1.0
        if sum_arr is None:
            continue
        avg = np.full_like(sum_arr, np.nan, dtype=float)
        valid = count_arr > 0
        avg[valid] = sum_arr[valid] / count_arr[valid]
        out[ticker] = pd.DataFrame(avg, index=models, columns=periods)
    return out


def _ticker_score(scores_df: pd.DataFrame, top_m: int) -> pd.Series:
    vals = []
    for col in scores_df.columns:
        col_vals = scores_df[col].to_numpy(dtype=float)
        col_vals = col_vals[np.isfinite(col_vals)]
        if col_vals.size == 0:
            vals.append(np.nan)
            continue
        top = np.sort(col_vals)[::-1][: min(top_m, col_vals.size)]
        vals.append(float(np.median(top)))
    return pd.Series(vals, index=scores_df.columns, name="ticker_score")


def _select_from_scores(
    scores_by_ticker: Dict[str, pd.DataFrame],
    cfg: MetaConfig,
) -> pd.DataFrame:
    score_frames = []
    gate_frames = []
    for ticker, scores_df in scores_by_ticker.items():
        if scores_df.empty:
            continue
        s_long = scores_df.stack(dropna=True).rename("score").reset_index()
        s_long = s_long.rename(columns={"level_0": "model_id", "level_1": "period_key"})
        s_long["ticker"] = ticker
        score_frames.append(s_long)
        gate = _ticker_score(scores_df, max(1, cfg.top_m_for_ticker_gate))
        gate = gate.dropna().rename("ticker_score").reset_index()
        gate = gate.rename(columns={"index": "period_key"})
        gate["ticker"] = ticker
        gate_frames.append(gate)
    if not score_frames or not gate_frames:
        return pd.DataFrame()
    scores_long = pd.concat(score_frames, ignore_index=True)
    ticker_gate = pd.concat(gate_frames, ignore_index=True)
    if cfg.min_ticker_score is not None:
        ticker_gate = ticker_gate[ticker_gate["ticker_score"] >= cfg.min_ticker_score]
    if ticker_gate.empty:
        return pd.DataFrame()
    ticker_gate = ticker_gate.sort_values(
        ["period_key", "ticker_score", "ticker"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    ticker_gate["ticker_rank"] = ticker_gate.groupby("period_key").cumcount() + 1
    ticker_counts = ticker_gate.groupby("period_key")["ticker"].transform("count")
    k_tickers = np.maximum(1, np.floor(np.sqrt(ticker_counts)).astype(int))
    chosen_tickers = ticker_gate[ticker_gate["ticker_rank"] <= k_tickers]
    candidates = scores_long.merge(
        chosen_tickers[["period_key", "ticker", "ticker_score", "ticker_rank"]],
        on=["period_key", "ticker"],
        how="inner",
    )
    if candidates.empty:
        return pd.DataFrame()
    candidates["score"] = pd.to_numeric(candidates["score"], errors="coerce")
    candidates = candidates.dropna(subset=["score"])
    if candidates.empty:
        return pd.DataFrame()
    candidates = candidates.sort_values(
        ["ticker", "period_key", "score", "model_id"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    candidates["rank_in_ticker"] = candidates.groupby(["ticker", "period_key"]).cumcount() + 1
    if cfg.per_ticker_cap is not None:
        candidates = candidates[candidates["rank_in_ticker"] <= cfg.per_ticker_cap]
        if candidates.empty:
            return pd.DataFrame()
    candidates = candidates.sort_values(
        ["period_key", "score", "ticker_rank", "rank_in_ticker", "ticker", "model_id"],
        ascending=[True, False, True, True, True, True],
        kind="mergesort",
    )
    candidates["rank_global"] = candidates.groupby("period_key").cumcount() + 1
    candidates = candidates[candidates["rank_global"] <= cfg.top_n_global]
    out = candidates.rename(columns={"period_key": "period_end"})
    out = out[
        ["period_end", "ticker", "model_id", "score", "rank_global", "rank_in_ticker", "ticker_score"]
    ].sort_values(["period_end", "rank_global"]).reset_index(drop=True)
    return out


def _period_key_start_dates(curves: pd.DataFrame) -> pd.Series:
    start_str = curves["period_key"].astype(str).str.split(" to ", n=1, expand=True)[0]
    return pd.to_datetime(start_str, errors="coerce")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ensemble meta-model selection using top configs from a sweep.",
    )
    parser.add_argument("--aligned-file", default=DEFAULT_ALIGNED_FILE)
    parser.add_argument("--sweep-results", default=DEFAULT_SWEEP_RESULTS)
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument("--top-n-configs", type=int, default=10)
    parser.add_argument("--normalize", choices=["percentile", "none"], default="percentile")
    parser.add_argument("--output-selections", default=None)
    parser.add_argument("--output-configs", default=None)
    parser.add_argument("--output-scores", default=None)
    parser.add_argument("--analysis-dir", default=None)
    parser.add_argument("--warmup-periods", type=int, default=DEFAULT_WARMUP_PERIODS)
    parser.add_argument("--oos-start-date", default=DEFAULT_OOS_START_DATE)
    parser.add_argument("--skip-analysis", action="store_true")
    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument("--enable-plots", action="store_true", dest="enable_plots")
    plot_group.add_argument("--disable-plots", action="store_false", dest="enable_plots")
    parser.set_defaults(enable_plots=True)
    parser.add_argument("--top-n-global", type=int, default=None)
    parser.add_argument("--per-ticker-cap", type=int, default=None)
    parser.add_argument("--min-ticker-score", type=float, default=None)
    parser.add_argument("--top-m-for-ticker-gate", type=int, default=None)
    parser.add_argument("--min-models-per-ticker", type=int, default=None)
    parser.add_argument("--require-common-periods", type=int, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.aligned_file):
        print(f"Aligned file not found: {args.aligned_file}", file=sys.stderr)
        return 1
    if not os.path.exists(args.sweep_results):
        print(f"Sweep results file not found: {args.sweep_results}", file=sys.stderr)
        return 1

    base_cfg = MetaConfig()
    if args.min_models_per_ticker is not None:
        base_cfg.min_models_per_ticker = args.min_models_per_ticker
    if args.require_common_periods is not None:
        base_cfg.require_common_periods = args.require_common_periods

    aligned_df = pd.read_csv(args.aligned_file).drop_duplicates()
    aligned_returns, _ = load_aligned_periods_from_csv(aligned_df, base_cfg)
    if not aligned_returns:
        print("No tickers survived alignment filters.", file=sys.stderr)
        return 1

    sweep_df = pd.read_csv(args.sweep_results)
    if args.metric not in sweep_df.columns:
        print(f"Metric column not found: {args.metric}", file=sys.stderr)
        return 1
    if "status" in sweep_df.columns:
        sweep_df = sweep_df[sweep_df["status"] == "ok"]
    sweep_df = sweep_df.copy()
    sweep_df[args.metric] = pd.to_numeric(sweep_df[args.metric], errors="coerce")
    sweep_df = sweep_df.dropna(subset=[args.metric])
    if sweep_df.empty:
        print("No usable configs found after filtering.", file=sys.stderr)
        return 1

    sweep_df = sweep_df.sort_values(args.metric, ascending=False).head(args.top_n_configs)
    if sweep_df.empty:
        print("No configs selected after sorting.", file=sys.stderr)
        return 1

    selection_cfg = _config_from_row(sweep_df.iloc[0], base_cfg)
    if args.top_n_global is not None:
        selection_cfg.top_n_global = args.top_n_global
    if args.per_ticker_cap is not None:
        selection_cfg.per_ticker_cap = args.per_ticker_cap
    if args.min_ticker_score is not None:
        selection_cfg.min_ticker_score = args.min_ticker_score
    if args.top_m_for_ticker_gate is not None:
        selection_cfg.top_m_for_ticker_gate = args.top_m_for_ticker_gate

    cfgs = [_config_from_row(row, selection_cfg) for _, row in sweep_df.iterrows()]
    scores_by_ticker = _ensemble_scores(aligned_returns, cfgs, args.normalize)
    selections = _select_from_scores(scores_by_ticker, selection_cfg)
    if selections.empty:
        print("No selections produced by ensemble.", file=sys.stderr)
        return 1

    if args.output_selections is None:
        base_dir = os.path.dirname(args.sweep_results)
        args.output_selections = os.path.join(base_dir, "meta_ensemble_selections.csv")
    selections.to_csv(args.output_selections, index=False)
    print(f"Saved selections to: {args.output_selections}")

    if args.output_configs is None:
        base_dir = os.path.dirname(args.sweep_results)
        args.output_configs = os.path.join(base_dir, "meta_ensemble_selected_configs.csv")
    sweep_df.to_csv(args.output_configs, index=False)
    print(f"Saved selected configs to: {args.output_configs}")

    if args.output_scores is not None:
        score_frames = []
        for ticker, scores_df in scores_by_ticker.items():
            if scores_df.empty:
                continue
            s_long = scores_df.stack(dropna=True).rename("score").reset_index()
            s_long = s_long.rename(columns={"level_0": "model_id", "level_1": "period_key"})
            s_long["ticker"] = ticker
            score_frames.append(s_long)
        if score_frames:
            scores_long = pd.concat(score_frames, ignore_index=True)
            scores_long.to_csv(args.output_scores, index=False)
            print(f"Saved ensemble scores to: {args.output_scores}")

    if not args.skip_analysis:
        base_dir = os.path.dirname(args.sweep_results)
        analysis_dir = args.analysis_dir or base_dir
        plots_dir = os.path.join(analysis_dir, "meta_ensemble_plots")
        os.makedirs(plots_dir, exist_ok=True)

        long_df = wide_to_long_periods(aligned_df)
        curves = compute_equity_curves(
            aligned_returns,
            selections,
            long_df,
            warmup_periods=args.warmup_periods,
            top_n=selection_cfg.top_n_global,
        )
        curves_path = os.path.join(analysis_dir, "meta_ensemble_equity_curves.csv")
        curves.to_csv(curves_path, index=False)
        print(f"Saved ensemble equity curves to: {curves_path}")

        if args.enable_plots:
            plot_equity_curves(curves, os.path.join(plots_dir, "equity_curves.png"))
            plot_avg_trade_return_curves(curves, os.path.join(plots_dir, "avg_trade_return_curves.png"))
            plot_trade_quality_distribution(curves, os.path.join(plots_dir, "trade_quality_hist.png"))
            plot_trade_quality_rolling_mean(curves, os.path.join(plots_dir, "trade_quality_rollmean.png"))
            plot_equity_ratio_curve(curves, os.path.join(plots_dir, "equity_ratio.png"))
            plot_rolling_sharpe_sortino(curves, os.path.join(plots_dir, "rolling_sharpe_sortino.png"))
            plot_rolling_outperformance_rate(curves, os.path.join(plots_dir, "rolling_outperformance.png"))
            plot_drawdown_curves(curves, os.path.join(plots_dir, "drawdown_curves.png"))
            plot_return_histograms(curves, os.path.join(plots_dir, "return_hist.png"))
            plot_return_delta_histogram(curves, os.path.join(plots_dir, "return_delta_hist.png"))
            plot_returns_scatter(curves, os.path.join(plots_dir, "return_scatter.png"))
            print(f"Saved ensemble plots to: {plots_dir}")

        core_metrics = compute_core_performance_metrics(curves)
        core_path = os.path.join(analysis_dir, "meta_ensemble_core_metrics.csv")
        core_metrics.to_csv(core_path)
        rel_metrics = compute_relative_edge_metrics(curves)
        rel_path = os.path.join(analysis_dir, "meta_ensemble_relative_metrics.csv")
        rel_metrics.to_csv(rel_path)
        stab_metrics = compute_stability_robustness_metrics(curves)
        stab_path = os.path.join(analysis_dir, "meta_ensemble_stability_metrics.csv")
        stab_metrics.to_csv(stab_path)
        trade_metrics = compute_trade_quality_metrics(curves)
        trade_path = os.path.join(analysis_dir, "meta_ensemble_trade_metrics.csv")
        trade_metrics.to_csv(trade_path)
        sig_metrics = compute_significance_metrics(curves, seed=42)
        sig_path = os.path.join(analysis_dir, "meta_ensemble_significance_metrics.csv")
        sig_metrics.to_csv(sig_path)
        print(f"Saved ensemble metrics to: {analysis_dir}")

        summary = {
            "ensemble_metric": args.metric,
            "top_n_configs": int(sweep_df.shape[0]),
            "normalize": args.normalize,
            "top_n_global": selection_cfg.top_n_global,
            "per_ticker_cap": selection_cfg.per_ticker_cap,
            "min_ticker_score": selection_cfg.min_ticker_score,
            "oos_start_date": "" if args.oos_start_date is None else str(args.oos_start_date),
            "n_selections": int(selections.shape[0]),
        }
        if "config_id" in sweep_df.columns:
            summary["selected_config_ids"] = ",".join(str(v) for v in sweep_df["config_id"].tolist())

        rel_series = rel_metrics["relative"] if "relative" in rel_metrics.columns else pd.Series(dtype=float)
        for key, val in rel_series.to_dict().items():
            summary[f"rel_{key}"] = val
        for scope in ["all_models", "topN", "relative"]:
            if scope in stab_metrics.columns:
                for key, val in stab_metrics[scope].to_dict().items():
                    summary[f"stab_{scope}_{key}"] = val
            if scope in trade_metrics.columns:
                for key, val in trade_metrics[scope].to_dict().items():
                    summary[f"trade_{scope}_{key}"] = val
        sig_series = sig_metrics["significance"] if "significance" in sig_metrics.columns else pd.Series(dtype=float)
        for key, val in sig_series.to_dict().items():
            summary[f"sig_{key}"] = val

        summary["mean_all_avg_return_per_trade_pct"] = float(curves["all_avg_return_per_trade"].mean() * 100.0)
        summary["mean_topN_avg_return_per_trade_pct"] = float(curves["topN_avg_return_per_trade"].mean() * 100.0)
        if args.oos_start_date is None:
            oos_curves = curves.tail(21)
        else:
            oos_start = pd.to_datetime(args.oos_start_date, errors="coerce")
            if pd.isna(oos_start):
                raise ValueError(f"Invalid oos_start_date: {args.oos_start_date}")
            starts = _period_key_start_dates(curves)
            oos_curves = curves[starts >= oos_start]
        summary["n_oos_periods"] = int(oos_curves.shape[0])
        summary["mean_all_avg_return_per_trade_pct_oos"] = float(
            oos_curves["all_avg_return_per_trade"].mean() * 100.0
        )
        summary["mean_topN_avg_return_per_trade_pct_oos"] = float(
            oos_curves["topN_avg_return_per_trade"].mean() * 100.0
        )

        summary_df = pd.DataFrame([summary])
        summary_path = os.path.join(analysis_dir, "meta_ensemble_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved ensemble summary to: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import os
from typing import Iterable, List

import numpy as np
import pandas as pd


def _series(df: pd.DataFrame, name: str) -> pd.Series:
    """Return numeric series or NaNs if missing."""
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def _first_existing(df: pd.DataFrame, names: Iterable[str]) -> pd.Series:
    for n in names:
        if n in df.columns:
            return pd.to_numeric(df[n], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def _period_returns_root() -> str:
    return os.path.join(
        os.path.expanduser("~"),
        "myhome",
        "algo",
        "artifacts",
        "period_returns",
    )


def _find_latest_sweep(base_dir: str) -> str | None:
    """Return the newest sweep results path from run_* dirs; fallback to root file."""
    if not os.path.isdir(base_dir):
        return None

    candidates: List[tuple[str, str]] = []
    for name in os.listdir(base_dir):
        run_dir = os.path.join(base_dir, name)
        if not os.path.isdir(run_dir):
            continue
        if not name.startswith("run_"):
            continue
        sweep_path = os.path.join(run_dir, "meta_config_sweep_results.csv")
        if os.path.exists(sweep_path):
            candidates.append((name, sweep_path))

    if candidates:
        candidates.sort(key=lambda t: t[0], reverse=True)
        return candidates[0][1]

    fallback = os.path.join(base_dir, "meta_config_sweep_results.csv")
    if os.path.exists(fallback):
        return fallback
    return None


def _build_scorecard(df: pd.DataFrame) -> pd.DataFrame:
    """Construct a compact scorecard per config."""
    base_cols: List[str] = [
        "config_id",
        "vol_window",
        "momentum_lookback",
        "conf_lookback",
        "risk_lookback",
        "alpha_low",
        "alpha_high",
        "alpha_smooth",
        "delta_weight",
        "cvar_alpha",
        "cvar_risk_aversion",
        "top_n_global",
        "per_ticker_cap",
    ]
    scorecard = pd.DataFrame(index=df.index)
    for c in base_cols:
        if c in df.columns:
            scorecard[c] = df[c]

    # Relative edge
    scorecard["edge_mean_pct"] = _series(df, "rel_mean_return_delta") * 100.0
    scorecard["edge_median_pct"] = _series(df, "rel_median_return_delta") * 100.0
    scorecard["edge_hit_rate"] = _series(df, "rel_pct_outperform")
    scorecard["edge_win_loss_ratio"] = _series(df, "rel_win_loss_ratio")
    scorecard["edge_equity_ratio_end"] = _series(df, "rel_equity_ratio_end")
    scorecard["edge_equity_ratio_cagr"] = _series(df, "rel_equity_ratio_cagr")

    # Sharpe/Sortino and volatility lifts (if core metrics already merged into sweep results)
    scorecard["sharpe_topN"] = _series(df, "core_topN_sharpe")
    scorecard["sharpe_all"] = _series(df, "core_all_models_sharpe")
    scorecard["sharpe_lift"] = scorecard["sharpe_topN"] - scorecard["sharpe_all"]
    scorecard["sortino_topN"] = _series(df, "core_topN_sortino")
    scorecard["sortino_all"] = _series(df, "core_all_models_sortino")
    scorecard["sortino_lift"] = scorecard["sortino_topN"] - scorecard["sortino_all"]
    scorecard["vol_topN"] = _series(df, "core_topN_volatility")
    scorecard["vol_all"] = _series(df, "core_all_models_volatility")

    # Drawdowns
    scorecard["drawdown_topN"] = _series(df, "core_topN_max_drawdown")
    scorecard["drawdown_all"] = _series(df, "core_all_models_max_drawdown")
    scorecard["drawdown_reduction"] = scorecard["drawdown_all"] - scorecard["drawdown_topN"]

    # Trade quality deltas
    scorecard["trade_quality_lift"] = _series(
        df, "trade_relative_mean_delta_avg_return_per_trade"
    )
    scorecard["trade_quality_hit_rate"] = _series(
        df, "trade_relative_pct_outperform_avg_return_per_trade"
    )

    # Stability
    scorecard["stab_outperf_rate_mean"] = _series(
        df, "stab_relative_rolling_outperformance_rate_mean"
    )
    scorecard["stab_outperf_rate_min"] = _series(
        df, "stab_relative_rolling_outperformance_rate_min"
    )
    scorecard["stab_max_outperf_streak"] = _series(
        df, "stab_relative_max_outperformance_streak"
    )
    scorecard["stab_max_underperf_streak"] = _series(
        df, "stab_relative_max_underperformance_streak"
    )

    # Significance
    scorecard["t_p_value"] = _series(df, "sig_t_p_value")
    scorecard["sign_test_p_value"] = _series(df, "sig_sign_test_p_value")
    scorecard["bootstrap_prob_gt0"] = _series(df, "sig_bootstrap_prob_mean_gt0")

    # OOS summary (if present)
    scorecard["oos_periods"] = _series(df, "n_oos_periods")
    scorecard["oos_edge_topN_avg_trade_pct"] = _series(
        df, "mean_topN_avg_return_per_trade_pct_oos"
    )
    scorecard["oos_edge_all_avg_trade_pct"] = _series(
        df, "mean_all_avg_return_per_trade_pct_oos"
    )

    # Counts
    scorecard["n_periods"] = _first_existing(df, ["rel_n_periods", "sig_n_periods"])
    scorecard["n_selections"] = _series(df, "n_selections")

    return scorecard


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a condensed scorecard per config from meta_config_sweep_results.csv"
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to meta_config_sweep_results.csv (default: latest run_*/meta_config_sweep_results.csv under period_returns)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: same directory, meta_config_scorecard.csv)",
    )
    parser.add_argument(
        "--sort",
        default="edge_mean_pct",
        help="Column to sort by (default: edge_mean_pct)",
    )
    args = parser.parse_args()

    if args.input is None:
        default_input = _find_latest_sweep(_period_returns_root())
        if default_input is None:
            raise FileNotFoundError(
                "Could not find meta_config_sweep_results.csv. "
                "Provide --input explicitly or ensure a sweep has been run."
            )
        input_path = default_input
    else:
        input_path = args.input

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
    if df.empty:
        raise RuntimeError("No rows to process after filtering status == 'ok'.")

    # Drop duplicate config_id if any, keeping first occurrence.
    if "config_id" in df.columns:
        df = df.drop_duplicates(subset=["config_id"], keep="first")

    scorecard = _build_scorecard(df)

    sort_col = args.sort
    if sort_col in scorecard.columns:
        scorecard = scorecard.sort_values(by=sort_col, ascending=False, na_position="last")

    out_path = (
        args.output
        if args.output
        else os.path.join(os.path.dirname(os.path.abspath(input_path)), "meta_config_scorecard.csv")
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    scorecard.to_csv(out_path, index=False)
    print(f"Wrote scorecard to: {out_path} (rows={scorecard.shape[0]}, cols={scorecard.shape[1]})")


if __name__ == "__main__":
    main()

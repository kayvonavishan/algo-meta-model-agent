from typing import Dict, Optional

import numpy as np
import pandas as pd


def _infer_periods_per_year(curves: pd.DataFrame) -> float:
    if "period_key" not in curves.columns:
        return 52.0
    parts = curves["period_key"].astype(str).str.split(" to ", n=1, expand=True)
    start = pd.to_datetime(parts[0], errors="coerce")
    end = pd.to_datetime(parts[1], errors="coerce")
    days = (end - start).dt.days + 1
    median_days = days.dropna().median()
    if pd.isna(median_days) or median_days <= 0:
        return 52.0
    return float(365.25 / median_days)


def _equity_from_returns(returns: pd.Series) -> pd.Series:
    return (1.0 + returns).cumprod()


def _drawdown_stats(equity: pd.Series) -> Dict[str, float]:
    eq = equity.dropna()
    if eq.empty:
        return {
            "max_drawdown": np.nan,
            "avg_drawdown_depth": np.nan,
            "avg_drawdown_duration": np.nan,
            "max_drawdown_duration": np.nan,
        }
    running_max = eq.cummax()
    drawdown = eq / running_max - 1.0
    max_dd = float(drawdown.min())
    dd_depth = drawdown[drawdown < 0.0]
    avg_dd = float(dd_depth.mean()) if not dd_depth.empty else 0.0

    durations = []
    cur = 0
    for in_dd in (drawdown < 0.0).to_numpy():
        if in_dd:
            cur += 1
        elif cur > 0:
            durations.append(cur)
            cur = 0
    if cur > 0:
        durations.append(cur)
    if durations:
        avg_dur = float(np.mean(durations))
        max_dur = float(np.max(durations))
    else:
        avg_dur = 0.0
        max_dur = 0.0

    return {
        "max_drawdown": max_dd,
        "avg_drawdown_depth": avg_dd,
        "avg_drawdown_duration": avg_dur,
        "max_drawdown_duration": max_dur,
    }


def _core_metrics_for_series(
    returns: pd.Series,
    equity: Optional[pd.Series],
    periods_per_year: float,
) -> Dict[str, float]:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if r.empty:
        return {
            "n_periods": 0,
            "mean_return": np.nan,
            "median_return": np.nan,
            "volatility": np.nan,
            "sharpe": np.nan,
            "sortino": np.nan,
            "hit_rate": np.nan,
            "cagr": np.nan,
            "calmar": np.nan,
            "p05_return": np.nan,
            "cvar_05": np.nan,
            "max_drawdown": np.nan,
            "avg_drawdown_depth": np.nan,
            "avg_drawdown_duration": np.nan,
            "max_drawdown_duration": np.nan,
        }

    if equity is None:
        eq = _equity_from_returns(r)
    else:
        eq = pd.to_numeric(equity, errors="coerce")
        eq = eq.loc[r.index].dropna()
        if eq.empty:
            eq = _equity_from_returns(r)

    n = int(r.shape[0])
    mean_ret = float(r.mean())
    med_ret = float(r.median())
    vol = float(r.std(ddof=1))
    sharpe = (mean_ret / vol) * np.sqrt(periods_per_year) if vol > 0 else np.nan
    downside = r[r < 0.0]
    down_std = float(downside.std(ddof=1)) if downside.shape[0] > 1 else np.nan
    sortino = (mean_ret / down_std) * np.sqrt(periods_per_year) if down_std and down_std > 0 else np.nan
    hit_rate = float((r > 0.0).mean())
    if eq.iloc[-1] <= 0:
        cagr = np.nan
    else:
        cagr = float(eq.iloc[-1] ** (periods_per_year / n) - 1.0)
    p05 = float(r.quantile(0.05))
    tail = r[r <= p05]
    cvar = float(tail.mean()) if not tail.empty else np.nan

    dd_stats = _drawdown_stats(eq)
    max_dd = dd_stats["max_drawdown"]
    calmar = float(cagr / abs(max_dd)) if max_dd is not None and max_dd < 0 else np.nan

    return {
        "n_periods": n,
        "mean_return": mean_ret,
        "median_return": med_ret,
        "volatility": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "hit_rate": hit_rate,
        "cagr": cagr,
        "calmar": calmar,
        "p05_return": p05,
        "cvar_05": cvar,
        **dd_stats,
    }


def compute_core_performance_metrics(
    curves: pd.DataFrame,
    periods_per_year: Optional[float] = None,
) -> pd.DataFrame:
    """
    Compute core performance metrics for all-models vs top-N portfolios.
    """
    if periods_per_year is None:
        periods_per_year = _infer_periods_per_year(curves)

    all_metrics = _core_metrics_for_series(
        curves["all_models_return"],
        curves.get("equity_all"),
        periods_per_year,
    )
    top_metrics = _core_metrics_for_series(
        curves["topN_return"],
        curves.get("equity_topN"),
        periods_per_year,
    )

    return pd.DataFrame({
        "all_models": all_metrics,
        "topN": top_metrics,
    })


def compute_relative_edge_metrics(curves: pd.DataFrame) -> pd.DataFrame:
    """
    Compute relative edge metrics for top-N vs all-models.
    """
    base = pd.to_numeric(curves["all_models_return"], errors="coerce")
    top = pd.to_numeric(curves["topN_return"], errors="coerce")
    mask = base.notna() & top.notna()
    if not mask.any():
        return pd.DataFrame({"relative": {}})

    base = base[mask]
    top = top[mask]
    delta = top - base

    mean_delta = float(delta.mean())
    median_delta = float(delta.median())
    std_delta = float(delta.std(ddof=1))
    pct_outperform = float((delta > 0.0).mean())
    pct_underperform = float((delta < 0.0).mean())
    mean_win = float(delta[delta > 0.0].mean()) if (delta > 0.0).any() else np.nan
    mean_loss = float(delta[delta < 0.0].mean()) if (delta < 0.0).any() else np.nan
    win_loss_ratio = float(mean_win / abs(mean_loss)) if np.isfinite(mean_win) and np.isfinite(mean_loss) and mean_loss != 0 else np.nan

    base_neg = base < 0.0
    base_pos = base > 0.0
    pct_base_neg = float(base_neg.mean())
    pct_base_pos = float(base_pos.mean())
    pct_base_neg_top_pos = float((top[base_neg] > 0.0).mean()) if base_neg.any() else np.nan
    pct_base_pos_top_neg = float((top[base_pos] < 0.0).mean()) if base_pos.any() else np.nan
    if base_neg.any():
        mean_base_neg = float(base[base_neg].mean())
        mean_top_on_base_neg = float(top[base_neg].mean())
        downside_capture = mean_top_on_base_neg / mean_base_neg if mean_base_neg != 0 else np.nan
    else:
        downside_capture = np.nan
    if base_pos.any():
        mean_base_pos = float(base[base_pos].mean())
        mean_top_on_base_pos = float(top[base_pos].mean())
        upside_capture = mean_top_on_base_pos / mean_base_pos if mean_base_pos != 0 else np.nan
    else:
        upside_capture = np.nan

    if "equity_all" in curves.columns and "equity_topN" in curves.columns:
        equity_all = pd.to_numeric(curves["equity_all"], errors="coerce")
        equity_top = pd.to_numeric(curves["equity_topN"], errors="coerce")
        eq_mask = equity_all.notna() & equity_top.notna()
        if eq_mask.any():
            eq_ratio = (equity_top[eq_mask] / equity_all[eq_mask]).replace([np.inf, -np.inf], np.nan).dropna()
            if not eq_ratio.empty:
                ratio_end = float(eq_ratio.iloc[-1])
                periods_per_year = _infer_periods_per_year(curves.loc[eq_mask])
                ratio_cagr = float(eq_ratio.iloc[-1] ** (periods_per_year / len(eq_ratio)) - 1.0)
            else:
                ratio_end = np.nan
                ratio_cagr = np.nan
        else:
            ratio_end = np.nan
            ratio_cagr = np.nan
    else:
        ratio_end = np.nan
        ratio_cagr = np.nan

    metrics = {
        "n_periods": int(mask.sum()),
        "mean_return_delta": mean_delta,
        "median_return_delta": median_delta,
        "volatility_delta": std_delta,
        "pct_outperform": pct_outperform,
        "pct_underperform": pct_underperform,
        "mean_delta_win": mean_win,
        "mean_delta_loss": mean_loss,
        "win_loss_ratio": win_loss_ratio,
        "pct_baseline_negative": pct_base_neg,
        "pct_baseline_positive": pct_base_pos,
        "pct_baseline_negative_meta_positive": pct_base_neg_top_pos,
        "pct_baseline_positive_meta_negative": pct_base_pos_top_neg,
        "downside_capture": downside_capture,
        "upside_capture": upside_capture,
        "equity_ratio_end": ratio_end,
        "equity_ratio_cagr": ratio_cagr,
    }

    return pd.DataFrame({"relative": metrics})


def _max_streak(mask: pd.Series) -> int:
    max_len = 0
    cur = 0
    for val in mask.to_numpy():
        if val:
            cur += 1
            if cur > max_len:
                max_len = cur
        else:
            cur = 0
    return int(max_len)


def _rolling_max_drawdown(returns: pd.Series, window: int) -> pd.Series:
    def _dd(x: np.ndarray) -> float:
        eq = np.cumprod(1.0 + x)
        run_max = np.maximum.accumulate(eq)
        dd = eq / run_max - 1.0
        return float(np.min(dd))
    return returns.rolling(window=window, min_periods=window).apply(_dd, raw=True)


def compute_trade_quality_metrics(curves: pd.DataFrame) -> pd.DataFrame:
    """
    Compute trade-quality metrics using avg return per trade series.
    """
    base = pd.to_numeric(curves["all_avg_return_per_trade"], errors="coerce").dropna()
    top = pd.to_numeric(curves["topN_avg_return_per_trade"], errors="coerce").dropna()
    n = min(len(base), len(top))
    base = base.iloc[:n]
    top = top.iloc[:n]

    def _series_metrics(s: pd.Series) -> Dict[str, float]:
        if s.empty:
            return {
                "n_periods": 0,
                "mean_avg_return_per_trade": np.nan,
                "median_avg_return_per_trade": np.nan,
                "volatility_avg_return_per_trade": np.nan,
                "hit_rate_avg_return_per_trade": np.nan,
                "neg_rate_avg_return_per_trade": np.nan,
                "p05_avg_return_per_trade": np.nan,
                "cvar_05_avg_return_per_trade": np.nan,
                "max_negative_streak": np.nan,
            }
        mean_val = float(s.mean())
        med_val = float(s.median())
        vol_val = float(s.std(ddof=1))
        hit_rate = float((s > 0.0).mean())
        neg_rate = float((s < 0.0).mean())
        p05 = float(s.quantile(0.05))
        tail = s[s <= p05]
        cvar = float(tail.mean()) if not tail.empty else np.nan
        return {
            "n_periods": int(s.shape[0]),
            "mean_avg_return_per_trade": mean_val,
            "median_avg_return_per_trade": med_val,
            "volatility_avg_return_per_trade": vol_val,
            "hit_rate_avg_return_per_trade": hit_rate,
            "neg_rate_avg_return_per_trade": neg_rate,
            "p05_avg_return_per_trade": p05,
            "cvar_05_avg_return_per_trade": cvar,
            "max_negative_streak": _max_streak(s < 0.0),
        }

    base_metrics = _series_metrics(base)
    top_metrics = _series_metrics(top)

    delta = top - base
    rel_metrics = {
        "mean_delta_avg_return_per_trade": float(delta.mean()) if not delta.empty else np.nan,
        "median_delta_avg_return_per_trade": float(delta.median()) if not delta.empty else np.nan,
        "pct_outperform_avg_return_per_trade": float((delta > 0.0).mean()) if not delta.empty else np.nan,
        "pct_underperform_avg_return_per_trade": float((delta < 0.0).mean()) if not delta.empty else np.nan,
        "max_outperformance_streak": _max_streak(delta > 0.0) if not delta.empty else np.nan,
        "max_underperformance_streak": _max_streak(delta < 0.0) if not delta.empty else np.nan,
    }

    return pd.DataFrame({
        "all_models": base_metrics,
        "topN": top_metrics,
        "relative": rel_metrics,
    })


def _binom_cdf(k: int, n: int, p: float = 0.5) -> float:
    from math import comb
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    probs = [comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(k + 1)]
    return float(sum(probs))


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + np.math.erf(x / np.sqrt(2.0)))


def compute_significance_metrics(
    curves: pd.DataFrame,
    n_bootstrap: int = 1000,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Compute significance/confidence metrics for return deltas.
    """
    base = pd.to_numeric(curves["all_models_return"], errors="coerce")
    top = pd.to_numeric(curves["topN_return"], errors="coerce")
    mask = base.notna() & top.notna()
    if not mask.any():
        return pd.DataFrame({"significance": {}})

    delta = (top - base)[mask].to_numpy(dtype=float)
    n = int(delta.shape[0])
    mean_delta = float(np.mean(delta))
    std_delta = float(np.std(delta, ddof=1)) if n > 1 else np.nan
    t_stat = float(mean_delta / (std_delta / np.sqrt(n))) if std_delta and std_delta > 0 else np.nan
    p_value = float(2.0 * (1.0 - _normal_cdf(abs(t_stat)))) if np.isfinite(t_stat) else np.nan

    pos = int(np.sum(delta > 0.0))
    neg = int(np.sum(delta < 0.0))
    n_eff = pos + neg
    if n_eff > 0:
        cdf = _binom_cdf(min(pos, neg), n_eff, 0.5)
        sign_p = float(2.0 * cdf)
        sign_p = min(1.0, sign_p)
    else:
        sign_p = np.nan

    rng = np.random.default_rng(seed)
    boot_means = rng.choice(delta, size=(n_bootstrap, n), replace=True).mean(axis=1)
    ci_low, ci_high = np.quantile(boot_means, [0.025, 0.975])
    prob_pos = float(np.mean(boot_means > 0.0))

    metrics = {
        "n_periods": n,
        "mean_delta": mean_delta,
        "std_delta": std_delta,
        "t_stat": t_stat,
        "t_p_value": p_value,
        "sign_test_p_value": sign_p,
        "sign_test_pos_rate": float(pos / n_eff) if n_eff > 0 else np.nan,
        "bootstrap_mean_ci_low": float(ci_low),
        "bootstrap_mean_ci_high": float(ci_high),
        "bootstrap_prob_mean_gt0": prob_pos,
    }

    return pd.DataFrame({"significance": metrics})


def compute_stability_robustness_metrics(
    curves: pd.DataFrame,
    window: int = 12,
    periods_per_year: Optional[float] = None,
) -> pd.DataFrame:
    """
    Compute rolling stability/robustness metrics for top-N vs all-models.
    """
    if periods_per_year is None:
        periods_per_year = _infer_periods_per_year(curves)

    base = pd.to_numeric(curves["all_models_return"], errors="coerce").dropna()
    top = pd.to_numeric(curves["topN_return"], errors="coerce").dropna()
    n = min(len(base), len(top))
    base = base.iloc[:n]
    top = top.iloc[:n]

    def _rolling_stats(r: pd.Series) -> Dict[str, float]:
        roll_mean = r.rolling(window=window, min_periods=window).mean()
        roll_vol = r.rolling(window=window, min_periods=window).std(ddof=1)
        roll_sharpe = (roll_mean / roll_vol) * np.sqrt(periods_per_year)
        roll_hit = (r > 0.0).astype(float).rolling(window=window, min_periods=window).mean()
        roll_dd = _rolling_max_drawdown(r, window)

        return {
            "rolling_mean_return_mean": float(roll_mean.mean()),
            "rolling_mean_return_min": float(roll_mean.min()),
            "rolling_vol_mean": float(roll_vol.mean()),
            "rolling_vol_max": float(roll_vol.max()),
            "rolling_sharpe_mean": float(roll_sharpe.mean()),
            "rolling_sharpe_min": float(roll_sharpe.min()),
            "rolling_hit_rate_mean": float(roll_hit.mean()),
            "rolling_hit_rate_min": float(roll_hit.min()),
            "rolling_max_drawdown_mean": float(roll_dd.mean()),
            "rolling_max_drawdown_min": float(roll_dd.min()),
            "max_losing_streak": _max_streak(r < 0.0),
        }

    base_stats = _rolling_stats(base)
    top_stats = _rolling_stats(top)

    delta = top - base
    outperf = delta > 0.0
    roll_outperf = outperf.astype(float).rolling(window=window, min_periods=window).mean()
    roll_delta = delta.rolling(window=window, min_periods=window).mean()

    rel_stats = {
        "rolling_outperformance_rate_mean": float(roll_outperf.mean()),
        "rolling_outperformance_rate_min": float(roll_outperf.min()),
        "rolling_delta_mean_mean": float(roll_delta.mean()),
        "rolling_delta_mean_min": float(roll_delta.min()),
        "max_outperformance_streak": _max_streak(outperf),
        "max_underperformance_streak": _max_streak(~outperf),
    }

    return pd.DataFrame({
        "all_models": base_stats,
        "topN": top_stats,
        "relative": rel_stats,
    })

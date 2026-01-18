import numpy as np
import pandas as pd


def compute_equity_curves(
    aligned: dict,
    selections: pd.DataFrame,
    long_df: pd.DataFrame,
    warmup_periods: int = 20,
    top_n: int = 50,
) -> pd.DataFrame:
    """
    Build equity curves for:
      - all models equally weighted
      - top-N selected models equally weighted
    Starts after warmup_periods to ensure sufficient history.
    Also returns weighted average return per trade when trade data is available in long_df.
    """
    if not aligned:
        raise ValueError("Aligned input is empty.")
    period_keys = list(next(iter(aligned.values())).columns)
    if len(period_keys) <= warmup_periods:
        raise ValueError("Not enough periods to apply warmup.")
    keep_keys = period_keys[warmup_periods:]
    
    # Stack aligned returns once
    ret_frames = []
    for tkr, mat in aligned.items():
        s = mat.stack(dropna=True)
        if s.empty:
            continue
        ret_frames.append(
            s.rename("period_return")
             .reset_index()
             .rename(columns={"level_0": "model_id", "level_1": "period_key"})
             .assign(ticker=tkr)
        )
    returns_df = pd.concat(ret_frames, ignore_index=True)
    
    # All-model equal-weight return per period
    all_models_ret = returns_df.groupby("period_key")["period_return"].mean()
    
    # Prepare trade info lookup from long_df
    trade_cols_present = {"num_trades", "avg_return_per_trade"}.issubset(set(long_df.columns))
    if "period_key" not in long_df.columns:
        long_df = long_df.copy()
        if "date_range" in long_df.columns:
            long_df["period_key"] = long_df["date_range"]
        else:
            long_df["period_key"] = (
                pd.to_datetime(long_df["period_start"]).dt.strftime("%Y-%m-%d")
                + " to "
                + pd.to_datetime(long_df["period_end"]).dt.strftime("%Y-%m-%d")
            )
    if trade_cols_present:
        trade_df = long_df[["ticker", "model_id", "period_key", "num_trades", "avg_return_per_trade"]].copy()
        trade_df["num_trades"] = pd.to_numeric(trade_df["num_trades"], errors="coerce").fillna(0.0)
        trade_df["avg_return_per_trade"] = pd.to_numeric(trade_df["avg_return_per_trade"], errors="coerce").fillna(0.0)
        # Weighted avg return per trade per period (all models)
        agg_all = trade_df.groupby("period_key").apply(
            lambda g: (g["avg_return_per_trade"] * g["num_trades"]).sum() / g["num_trades"].sum()
            if g["num_trades"].sum() > 0 else np.nan
        )
    else:
        trade_df = None
        agg_all = pd.Series(np.nan, index=period_keys, name="all_avg_trade_return")
    
    # Top-N returns per period using selections (post-warmup)
    sel = selections[selections["period_end"].isin(keep_keys)].copy()
    sel = sel.sort_values(["period_end", "rank_global"])
    sel_top = sel.groupby("period_end").head(top_n)
    sel_top = sel_top.merge(
        returns_df,
        left_on=["ticker", "model_id", "period_end"],
        right_on=["ticker", "model_id", "period_key"],
        how="left",
    )
    top_ret = sel_top.groupby("period_end")["period_return"].mean()
    
    if trade_cols_present:
        sel_top_trade = sel_top.merge(
            trade_df,
            left_on=["ticker", "model_id", "period_end"],
            right_on=["ticker", "model_id", "period_key"],
            how="left",
            suffixes=("", "_trade"),
        )
        def _weighted_avg(g):
            w = g["num_trades"]
            if w.sum() <= 0:
                return np.nan
            return float((g["avg_return_per_trade"] * w).sum() / w.sum())
        top_avg = sel_top_trade.groupby("period_end").apply(_weighted_avg)
    else:
        top_avg = pd.Series(np.nan, index=keep_keys, name="topN_avg_trade_return")
    
    df = pd.DataFrame({
        "period_key": keep_keys,
        "all_models_return": all_models_ret.reindex(keep_keys).values,
        "topN_return": top_ret.reindex(keep_keys).values,
    })
    df["all_avg_return_per_trade"] = agg_all.reindex(keep_keys).values
    df["topN_avg_return_per_trade"] = top_avg.reindex(keep_keys).values
    df["equity_all"] = (1.0 + df["all_models_return"]).cumprod()
    df["equity_topN"] = (1.0 + df["topN_return"]).cumprod()
    return df


def plot_equity_curves(curves_df: pd.DataFrame, out_path: str) -> None:
    """Simple line plot for equity curves."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib is required to plot equity curves.") from e
    plt.figure(figsize=(10, 5))
    plt.plot(curves_df["period_key"], curves_df["equity_all"], label="All models EW")
    plt.plot(curves_df["period_key"], curves_df["equity_topN"], label="Top-N EW")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Period (date range)")
    plt.ylabel("Equity (start=1.0)")
    plt.legend()
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path)
    plt.close()


def _period_ticks(curves_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x_vals = np.arange(len(curves_df))
    step = max(1, len(x_vals) // 12)
    tick_idx = np.arange(0, len(x_vals), step)
    return x_vals, tick_idx


def _infer_periods_per_year(curves_df: pd.DataFrame) -> float:
    if "period_key" not in curves_df.columns:
        return 52.0
    parts = curves_df["period_key"].astype(str).str.split(" to ", n=1, expand=True)
    start = pd.to_datetime(parts[0], errors="coerce")
    end = pd.to_datetime(parts[1], errors="coerce")
    days = (end - start).dt.days + 1
    median_days = days.dropna().median()
    if pd.isna(median_days) or median_days <= 0:
        return 52.0
    return float(365.25 / median_days)


def plot_equity_ratio_curve(curves_df: pd.DataFrame, out_path: str) -> None:
    """Plot equity ratio (top-N / all-models)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib is required to plot equity ratio curves.") from e
    equity_all = pd.to_numeric(curves_df["equity_all"], errors="coerce")
    equity_top = pd.to_numeric(curves_df["equity_topN"], errors="coerce")
    ratio = (equity_top / equity_all).replace([np.inf, -np.inf], np.nan)
    x_vals, tick_idx = _period_ticks(curves_df)
    plt.figure(figsize=(10, 5))
    plt.plot(x_vals, ratio, label="Equity ratio (Top-N / All)")
    plt.axhline(1.0, color="black", linewidth=1.0, alpha=0.6)
    plt.xticks(tick_idx, curves_df["period_key"].iloc[tick_idx], rotation=45, ha="right")
    plt.xlabel("Period (date range)")
    plt.ylabel("Equity ratio")
    plt.legend()
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path)
    plt.close()


def plot_rolling_sharpe_sortino(
    curves_df: pd.DataFrame,
    out_path: str,
    window: int = 12,
) -> None:
    """Rolling Sharpe and Sortino for all-models vs top-N."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib is required to plot rolling Sharpe/Sortino.") from e
    r_all = pd.to_numeric(curves_df["all_models_return"], errors="coerce")
    r_top = pd.to_numeric(curves_df["topN_return"], errors="coerce")
    periods_per_year = _infer_periods_per_year(curves_df)

    roll_mean_all = r_all.rolling(window=window, min_periods=window).mean()
    roll_vol_all = r_all.rolling(window=window, min_periods=window).std(ddof=1)
    sharpe_all = (roll_mean_all / roll_vol_all) * np.sqrt(periods_per_year)

    roll_mean_top = r_top.rolling(window=window, min_periods=window).mean()
    roll_vol_top = r_top.rolling(window=window, min_periods=window).std(ddof=1)
    sharpe_top = (roll_mean_top / roll_vol_top) * np.sqrt(periods_per_year)

    def _downside_std(x: np.ndarray) -> float:
        neg = x[x < 0.0]
        if neg.size < 2:
            return np.nan
        return float(np.std(neg, ddof=1))

    down_all = r_all.rolling(window=window, min_periods=window).apply(_downside_std, raw=True)
    sortino_all = (roll_mean_all / down_all) * np.sqrt(periods_per_year)
    down_top = r_top.rolling(window=window, min_periods=window).apply(_downside_std, raw=True)
    sortino_top = (roll_mean_top / down_top) * np.sqrt(periods_per_year)

    x_vals, tick_idx = _period_ticks(curves_df)
    plt.figure(figsize=(10, 5))
    plt.plot(x_vals, sharpe_all, label="Sharpe (All)")
    plt.plot(x_vals, sharpe_top, label="Sharpe (Top-N)")
    plt.plot(x_vals, sortino_all, label="Sortino (All)", linestyle="--")
    plt.plot(x_vals, sortino_top, label="Sortino (Top-N)", linestyle="--")
    plt.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    plt.xticks(tick_idx, curves_df["period_key"].iloc[tick_idx], rotation=45, ha="right")
    plt.xlabel("Period (date range)")
    plt.ylabel(f"Rolling ({window}) Sharpe / Sortino")
    plt.legend()
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path)
    plt.close()


def plot_rolling_outperformance_rate(
    curves_df: pd.DataFrame,
    out_path: str,
    window: int = 12,
) -> None:
    """Rolling outperformance rate: pct periods Top-N > All over window."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib is required to plot rolling outperformance rate.") from e
    r_all = pd.to_numeric(curves_df["all_models_return"], errors="coerce")
    r_top = pd.to_numeric(curves_df["topN_return"], errors="coerce")
    outperf = (r_top > r_all).astype(float)
    roll = outperf.rolling(window=window, min_periods=window).mean()
    x_vals, tick_idx = _period_ticks(curves_df)
    plt.figure(figsize=(10, 5))
    plt.plot(x_vals, roll, label="Rolling outperformance rate")
    plt.axhline(0.5, color="black", linewidth=1.0, alpha=0.6)
    plt.xticks(tick_idx, curves_df["period_key"].iloc[tick_idx], rotation=45, ha="right")
    plt.xlabel("Period (date range)")
    plt.ylabel(f"Outperformance rate (window={window})")
    plt.legend()
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path)
    plt.close()


def plot_drawdown_curves(curves_df: pd.DataFrame, out_path: str) -> None:
    """Drawdown curves for all-models and top-N."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib is required to plot drawdown curves.") from e
    equity_all = pd.to_numeric(curves_df["equity_all"], errors="coerce")
    equity_top = pd.to_numeric(curves_df["equity_topN"], errors="coerce")
    dd_all = equity_all / equity_all.cummax() - 1.0
    dd_top = equity_top / equity_top.cummax() - 1.0
    x_vals, tick_idx = _period_ticks(curves_df)
    plt.figure(figsize=(10, 5))
    plt.plot(x_vals, dd_all, label="Drawdown (All)")
    plt.plot(x_vals, dd_top, label="Drawdown (Top-N)")
    plt.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    plt.xticks(tick_idx, curves_df["period_key"].iloc[tick_idx], rotation=45, ha="right")
    plt.xlabel("Period (date range)")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path)
    plt.close()


def plot_return_histograms(curves_df: pd.DataFrame, out_path: str, bins: int = 50) -> None:
    """Histogram/KDE overlay of period returns."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib is required to plot return histograms.") from e
    all_series = pd.to_numeric(curves_df["all_models_return"], errors="coerce").dropna()
    top_series = pd.to_numeric(curves_df["topN_return"], errors="coerce").dropna()
    plt.figure(figsize=(10, 5))
    plt.hist(all_series, bins=bins, alpha=0.6, label="All models", density=True)
    plt.hist(top_series, bins=bins, alpha=0.6, label="Top-N", density=True)
    try:
        from scipy.stats import gaussian_kde
    except Exception:
        gaussian_kde = None
    if gaussian_kde is not None and not all_series.empty and not top_series.empty:
        xs = np.linspace(min(all_series.min(), top_series.min()), max(all_series.max(), top_series.max()), 200)
        plt.plot(xs, gaussian_kde(all_series)(xs), label="All models KDE")
        plt.plot(xs, gaussian_kde(top_series)(xs), label="Top-N KDE")
    plt.xlabel("Period return")
    plt.ylabel("Density")
    plt.title("Period Returns Distribution")
    plt.legend()
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path)
    plt.close()


def plot_return_delta_histogram(curves_df: pd.DataFrame, out_path: str, bins: int = 50) -> None:
    """Histogram of return delta (Top-N - All)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib is required to plot return delta histogram.") from e
    delta = pd.to_numeric(curves_df["topN_return"], errors="coerce") - pd.to_numeric(
        curves_df["all_models_return"],
        errors="coerce",
    )
    delta = delta.dropna()
    plt.figure(figsize=(10, 5))
    plt.hist(delta, bins=bins, alpha=0.75, color="steelblue", density=True)
    plt.axvline(0.0, color="black", linewidth=1.0, alpha=0.6)
    plt.xlabel("Return delta (Top-N - All)")
    plt.ylabel("Density")
    plt.title("Return Delta Distribution")
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path)
    plt.close()


def plot_returns_scatter(curves_df: pd.DataFrame, out_path: str) -> None:
    """Scatter of baseline vs top-N returns with quadrant lines."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib is required to plot return scatter.") from e
    base = pd.to_numeric(curves_df["all_models_return"], errors="coerce")
    top = pd.to_numeric(curves_df["topN_return"], errors="coerce")
    mask = base.notna() & top.notna()
    base = base[mask]
    top = top[mask]
    if base.empty or top.empty:
        return
    plt.figure(figsize=(6, 6))
    plt.scatter(base, top, alpha=0.5, s=12)
    plt.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    plt.axvline(0.0, color="black", linewidth=1.0, alpha=0.6)
    lim = max(abs(base).max(), abs(top).max())
    plt.plot([-lim, lim], [-lim, lim], linestyle="--", color="gray", alpha=0.6)
    plt.xlabel("All models return")
    plt.ylabel("Top-N return")
    plt.title("Baseline vs Top-N Returns")
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path)
    plt.close()


def plot_avg_trade_return_curves(curves_df: pd.DataFrame, out_path: str, title: str | None = None) -> None:
    """Plot average return per trade for all models vs top-N."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib is required to plot average trade return curves.") from e
    x_vals = np.arange(len(curves_df))
    step = max(1, len(x_vals) // 12)
    tick_idx = np.arange(0, len(x_vals), step)
    plt.figure(figsize=(10, 5))
    plt.plot(x_vals, curves_df["all_avg_return_per_trade"], label="All models avg return per trade")
    plt.plot(x_vals, curves_df["topN_avg_return_per_trade"], label="Top-N avg return per trade")
    plt.xticks(tick_idx, curves_df["period_key"].iloc[tick_idx], rotation=45, ha="right")
    plt.xlabel("Period (date range)")
    plt.ylabel("Avg return per trade")
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path)
    plt.close()


def plot_trade_quality_distribution(curves_df: pd.DataFrame, out_path: str, bins: int = 50) -> None:
    """Histogram of avg return per trade for all models vs top-N."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib is required to plot trade quality distributions.") from e
    all_series = pd.to_numeric(curves_df["all_avg_return_per_trade"], errors="coerce").dropna()
    top_series = pd.to_numeric(curves_df["topN_avg_return_per_trade"], errors="coerce").dropna()
    plt.figure(figsize=(10, 5))
    plt.hist(all_series, bins=bins, alpha=0.6, label="All models", density=True)
    plt.hist(top_series, bins=bins, alpha=0.6, label="Top-N", density=True)
    plt.xlabel("Avg return per trade")
    plt.ylabel("Density")
    plt.title("Avg Return per Trade Distribution")
    plt.legend()
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path)
    plt.close()


def plot_trade_quality_rolling_mean(
    curves_df: pd.DataFrame,
    out_path: str,
    window: int = 12,
) -> None:
    """Rolling mean of avg return per trade for all models vs top-N."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib is required to plot trade quality rolling mean.") from e
    all_series = pd.to_numeric(curves_df["all_avg_return_per_trade"], errors="coerce")
    top_series = pd.to_numeric(curves_df["topN_avg_return_per_trade"], errors="coerce")
    roll_all = all_series.rolling(window=window, min_periods=window).mean()
    roll_top = top_series.rolling(window=window, min_periods=window).mean()
    x_vals = np.arange(len(curves_df))
    step = max(1, len(x_vals) // 12)
    tick_idx = np.arange(0, len(x_vals), step)
    plt.figure(figsize=(10, 5))
    plt.plot(x_vals, roll_all, label=f"All models ({window}-period MA)")
    plt.plot(x_vals, roll_top, label=f"Top-N ({window}-period MA)")
    plt.xticks(tick_idx, curves_df["period_key"].iloc[tick_idx], rotation=45, ha="right")
    plt.xlabel("Period (date range)")
    plt.ylabel("Rolling avg return per trade")
    plt.title("Rolling Avg Return per Trade")
    plt.legend()
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path)
    plt.close()

import re
import math
import os
import time
from contextlib import contextmanager
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


# =========================
# Config
# =========================

@dataclass
class MetaConfig:
    # Alignment / parsing
    min_models_per_ticker: int = 5   # skip tickers with too few models
    require_common_periods: int = 8  # skip tickers with too few common periods after alignment
    
    # Vol -> alpha (adaptive memory)
    vol_window: int = 4              # 3–5 typical
    alpha_low: float = 0.30          # 0.25–0.35 typical
    alpha_high: float = 0.70         # 0.60–0.75 typical
    z_low: float = -1.0              # clip mapping range
    z_high: float = 1.0
    alpha_smooth: float = 0.30       # EMA smoothing for alpha series (0.1–0.4 typical)
    
    # Momentum / delta
    momentum_lookback: int = 12
    delta_weight: float = 0.20       # small (0.05–0.3)
    
    # Confidence (training-free)
    conf_lookback: int = 12
    conf_eps: float = 1e-8
    
    # Risk penalty (training-free)
    risk_lookback: int = 20
    cvar_alpha: float = 0.10         # tail depth
    cvar_risk_aversion: float = 0.75 # penalty strength
    cvar_window_stride: int = 1      # downsample within CVaR lookback window (1 = exact)
    
    # Baseline
    baseline_method: str = "median"  # "median" or "mean"
    
    # Redundancy control (optional)
    enable_uniqueness_weighting: bool = True
    corr_cluster_threshold: float = 0.95  # high correlation => duplicates
    uniqueness_floor: float = 0.25        # prevent weights from going too small
    
    # Selection
    top_n_global: int = 20           # total selected per period across all tickers
    top_m_for_ticker_gate: int = 5   # use top M per ticker to compute ticker score
    per_ticker_cap: Optional[int] = None  # cap selected models per ticker (None = no cap)
    min_ticker_score: Optional[float] = None  # abstain tickers below this score (None = no abstain)


# =========================
# Helpers: period parsing / long-format conversion
# =========================

@contextmanager
def _timer(label: str):
    start_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    start = time.time()
    print(f"[TIMER] {label} start={start_ts}")
    try:
        yield
    finally:
        end = time.time()
        end_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[TIMER] {label} end={end_ts} elapsed_s={end - start:.3f}")

PERIOD_RE = re.compile(r"^period_(\d+)_(return|avg_return_per_trade|num_trades|date_range)$")

def extract_period_indices(df: pd.DataFrame) -> List[int]:
    periods = set()
    for c in df.columns:
        m = PERIOD_RE.match(c)
        if m:
            periods.add(int(m.group(1)))
    return sorted(periods)

def parse_date_range(dr: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    # Expect: "YYYY-MM-DD to YYYY-MM-DD"
    a, b = dr.split(" to ")
    return pd.to_datetime(a), pd.to_datetime(b)

def infer_ticker(row: pd.Series) -> str:
    """
    Try to infer ticker. Prefer optuna_ticker_list if present.
    Falls back to source_experiment_name prefix.
    """
    if "optuna_ticker_list" in row and pd.notna(row["optuna_ticker_list"]):
        v = str(row["optuna_ticker_list"]).strip()
        # common forms: "TQQQ", "['TQQQ']", "TQQQ,SOXL"
        v = v.strip("[]()")
        v = v.replace("'", "").replace('"', "")
        parts = [p.strip() for p in v.split(",") if p.strip()]
        if parts:
            return parts[0].lower()
    
    # fallback: source_experiment_name like "tqqq-long-YYYYMMDD..."
    if "source_experiment_name" in row and pd.notna(row["source_experiment_name"]):
        name = str(row["source_experiment_name"]).lower()
        # take prefix up to first '-'
        if "-" in name:
            return name.split("-")[0]
        return name
    return "unknown"

def build_model_id(df: pd.DataFrame) -> pd.Series:
    """
    Create a stable model identifier. Adjust if you prefer different keys.
    """
    cols = []
    for c in ["source_experiment_name","source_run_id", "optuna_ticker_list","outer_trial_number", "inner_trial_number", "buy_sell_signal_strategy", "model_type"]:
        if c in df.columns:
            cols.append(c)
    if not cols:
        # fallback to index
        return df.index.astype(str)
    
    # combine
    return df[cols].astype(str).agg("|".join, axis=1)

def wide_to_long_periods(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fast wide -> long conversion using wide_to_long + vectorized date parsing.
    Output columns:
      [model_id, ticker, period, date_range, period_return, avg_return_per_trade, num_trades, period_start, period_end]
    """
    period_idxs = extract_period_indices(df)
    if not period_idxs:
        raise ValueError("No period_N_* columns found.")
    
    df = df.copy()
    
    # These are expensive row-wise operations; do them once, vectorized where possible
    if "ticker" not in df.columns:
        # If you already created ticker earlier, skip this.
        df["ticker"] = df.apply(infer_ticker, axis=1)  # can be optimized further if needed
    if "model_id" not in df.columns:
        df["model_id"] = build_model_id(df)
    
    # Keep only needed columns to reduce memory traffic
    id_cols = ["model_id", "ticker"]
    period_cols = [c for c in df.columns if c.startswith("period_")]
    base = df[id_cols + period_cols]
    
    # wide_to_long expects stubs like "period_return" with suffix "1", "2", ...
    # Your columns are period_1_return, so we rename to period_return_1, etc.
    rename_map = {}
    for c in period_cols:
        m = PERIOD_RE.match(c)
        if not m:
            continue
        k = m.group(1)
        field = m.group(2)
        rename_map[c] = f"period_{field}_{k}"
    
    base = base.rename(columns=rename_map)
    
    # Now we can wide_to_long on stubs: period_return, period_date_range, period_avg_return_per_trade, period_num_trades
    stubnames = ["period_return", "period_date_range", "period_avg_return_per_trade", "period_num_trades"]
    # Some files might not have all stubs; keep only those present
    present_stubs = []
    for s in stubnames:
        if any(col.startswith(s + "_") for col in base.columns):
            present_stubs.append(s)
    
    long_df = (
        pd.wide_to_long(
            base,
            stubnames=present_stubs,
            i=id_cols,
            j="period",
            sep="_",
            suffix=r"\d+",
        )
        .reset_index()
        .rename(columns={
            "period_date_range": "date_range",
            "period_avg_return_per_trade": "avg_return_per_trade",
            "period_num_trades": "num_trades",
            "period_return": "period_return",
        })
    )
    
    # Drop rows with no date_range (missing period)
    if "date_range" in long_df.columns:
        long_df = long_df.dropna(subset=["date_range"])
    else:
        raise ValueError("Missing date_range after reshape; ensure period_*_date_range columns exist.")
    
    # Vectorized date parsing: "YYYY-MM-DD to YYYY-MM-DD"
    # Much faster than Python loops
    dr = long_df["date_range"].astype(str)
    parts = dr.str.split(" to ", n=1, expand=True)
    long_df["period_start"] = pd.to_datetime(parts[0], errors="coerce")
    long_df["period_end"] = pd.to_datetime(parts[1], errors="coerce")
    
    # Numeric conversion (vectorized)
    long_df["period_return"] = pd.to_numeric(long_df["period_return"], errors="coerce")
    if "avg_return_per_trade" in long_df.columns:
        long_df["avg_return_per_trade"] = pd.to_numeric(long_df["avg_return_per_trade"], errors="coerce")
    if "num_trades" in long_df.columns:
        long_df["num_trades"] = pd.to_numeric(long_df["num_trades"], errors="coerce")
    
    # Drop rows without usable return/end date
    long_df = long_df.dropna(subset=["period_end", "period_return"])
    
    # Optional: ensure period is int
    long_df["period"] = long_df["period"].astype(int)
    
    return long_df


# =========================
# Helpers: aggregation from daily to multi-week periods
# =========================

def find_first_common_monday(df: pd.DataFrame) -> pd.Timestamp:
    """
    Find the first Monday on or after every model's test_data_start_timestamp.
    This date is used as the shared anchor so that all aggregated periods align.
    """
    if "test_data_start_timestamp" not in df.columns:
        raise ValueError("Column 'test_data_start_timestamp' is required to anchor periods.")
    
    starts = pd.to_datetime(df["test_data_start_timestamp"], errors="coerce", utc=True)
    starts = starts.dt.tz_convert(None).dt.normalize()
    if starts.isna().all():
        raise ValueError("Could not parse any test_data_start_timestamp values.")
    
    # For each model, roll forward to the next Monday (or the same day if already Monday)
    next_mondays = starts + pd.to_timedelta((7 - starts.dt.weekday) % 7, unit="D")
    anchor = next_mondays.max()
    if pd.isna(anchor):
        raise ValueError("Failed to determine a common Monday anchor date.")
    return anchor

def aggregate_daily_periods_to_windows(
    daily_long_df: pd.DataFrame,
    anchor_monday: pd.Timestamp,
    window_days: int = 14,
) -> pd.DataFrame:
    """
    Aggregate daily period rows into fixed-length windows (default: 14 days).
    Period 1 starts at anchor_monday for every model, ensuring aligned date ranges.
    """
    if window_days <= 0:
        raise ValueError("window_days must be positive.")
    
    df = daily_long_df.copy()
    for col in ["period_start", "period_end"]:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    df["period_return"] = pd.to_numeric(df["period_return"], errors="coerce")
    has_trades = "num_trades" in df.columns
    has_avg = "avg_return_per_trade" in df.columns
    if has_trades:
        df["num_trades"] = pd.to_numeric(df["num_trades"], errors="coerce")
    if has_avg:
        df["avg_return_per_trade"] = pd.to_numeric(df["avg_return_per_trade"], errors="coerce")
    
    # Detect and rescale percentage-like returns to decimal if needed
    abs_q99 = df["period_return"].abs().quantile(0.99)
    needs_scale = (abs_q99 > 1.0) or (df["period_return"].max() > 1.0) or (df["period_return"].min() < -1.0)
    if needs_scale:
        df["period_return"] = df["period_return"] / 100.0
        if has_avg:
            df["avg_return_per_trade"] = df["avg_return_per_trade"] / 100.0
    
    df = df.dropna(subset=["model_id", "ticker", "period_start", "period_return"])
    df = df[df["period_start"] >= anchor_monday]
    if df.empty:
        raise ValueError("No daily periods found on or after the anchor Monday.")
    
    df["window_idx"] = ((df["period_start"] - anchor_monday).dt.days // window_days).astype(int)
    df["window_start"] = anchor_monday + pd.to_timedelta(df["window_idx"] * window_days, unit="D")
    df["window_end"] = df["window_start"] + pd.Timedelta(days=window_days - 1)
    
    # Use log space to compound quickly: sum(log(1+r)) then expm1
    df["period_return"] = df["period_return"].clip(lower=-0.999999)  # guard against < -100% returns
    df["log_ret"] = np.log1p(df["period_return"])
    
    if has_trades and has_avg:
        df["weighted_avg_ret"] = df["avg_return_per_trade"].fillna(0.0) * df["num_trades"].fillna(0.0)
    
    group_keys = ["model_id", "ticker", "window_idx", "window_start", "window_end"]
    agg_dict = {
        "log_ret": "sum",
    }
    if has_trades:
        agg_dict["num_trades"] = "sum"
    if has_trades and has_avg:
        agg_dict["weighted_avg_ret"] = "sum"
    
    grouped = df.groupby(group_keys, sort=False).agg(agg_dict).reset_index()
    if grouped.empty:
        raise ValueError("No aggregated windows were produced from the daily periods.")
    
    grouped["period"] = grouped["window_idx"].astype(int) + 1
    grouped["period_return"] = np.expm1(grouped["log_ret"])
    
    if has_trades:
        grouped["num_trades"] = grouped["num_trades"].astype(float)
    if has_trades and has_avg:
        trades_sum = grouped["num_trades"].replace(0, np.nan)
        grouped["avg_return_per_trade"] = grouped["weighted_avg_ret"] / trades_sum
    
    grouped["date_range"] = (
        grouped["window_start"].dt.strftime("%Y-%m-%d")
        + " to "
        + grouped["window_end"].dt.strftime("%Y-%m-%d")
    )
    
    cols = [
        "model_id",
        "ticker",
        "period",
        "date_range",
        "period_return",
        "avg_return_per_trade" if has_trades and has_avg else None,
        "num_trades" if has_trades else None,
        "window_start",
        "window_end",
    ]
    cols = [c for c in cols if c is not None]
    out = grouped[cols].rename(columns={"window_start": "period_start", "window_end": "period_end"})
    out = out.sort_values(["model_id", "period"]).reset_index(drop=True)
    return out


# =========================
# Alignment per ticker
# =========================

def align_to_common_periods_per_ticker(long_df: pd.DataFrame, cfg: MetaConfig) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, pd.DataFrame]]]:
    out_returns: Dict[str, pd.DataFrame] = {}
    out_meta: Dict[str, Dict[str, pd.DataFrame]] = {}
    
    gdf = long_df.copy()
    
    # Ensure clean datetimes (and kill time-of-day / timezone drift)
    for c in ["period_start", "period_end"]:
        gdf[c] = pd.to_datetime(gdf[c], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    
    gdf = gdf.dropna(subset=["ticker", "model_id", "period_start", "period_end", "period_return"])
    
    # Canonical period key so all equality comparisons are exact and stable
    gdf["period_key"] = (
        gdf["period_start"].dt.strftime("%Y-%m-%d")
        + " to "
        + gdf["period_end"].dt.strftime("%Y-%m-%d")
    )
    
    # Optional sanity filter: drop clearly bogus timestamps (e.g., 1972 from numeric parsing)
    gdf = gdf[gdf["period_end"] >= pd.Timestamp("2000-01-01")]
    
    def _longest_continuous_block(periods: pd.DataFrame) -> List[str]:
        """
        periods: unique rows with columns [period_key, period_start, period_end], sorted by period_end
        Returns list of period_key in the longest contiguous run.
        Continuity is defined by the most common step between consecutive period_end values.
        """
        if periods.empty:
            return []
        
        periods = periods.sort_values("period_end").reset_index(drop=True)
        ends = periods["period_end"]
        
        if len(ends) == 1:
            return periods["period_key"].tolist()
        
        diffs = ends.diff().dropna()
        # choose the most common diff as the "step" (falls back to median)
        mode = diffs.mode()
        step = mode.iloc[0] if not mode.empty else diffs.median()
        
        best_start = 0
        best_len = 1
        cur_start = 0
        cur_len = 1
        
        for i in range(1, len(ends)):
            if (ends.iloc[i] - ends.iloc[i - 1]) == step:
                cur_len += 1
            else:
                if cur_len > best_len:
                    best_len = cur_len
                    best_start = cur_start
                cur_start = i
                cur_len = 1
        
        if cur_len > best_len:
            best_len = cur_len
            best_start = cur_start
        
        block = periods.iloc[best_start : best_start + best_len]
        return block["period_key"].tolist()
    
    # Filter tickers that meet the minimum model requirement
    ticker_counts = gdf.groupby("ticker")["model_id"].nunique()
    eligible_tickers = ticker_counts.index[ticker_counts >= cfg.min_models_per_ticker]
    gdf = gdf[gdf["ticker"].isin(eligible_tickers)]
    if gdf.empty:
        return {}
    
    total_models = gdf["model_id"].nunique()
    
    # Global common periods: must appear in every model across all tickers
    key_counts = gdf.groupby("period_key")["model_id"].nunique()
    global_common_keys = key_counts.index[key_counts == total_models]
    print(f"Global: {total_models} models, {len(global_common_keys)} common periods across all models")
    if len(global_common_keys) == 0:
        return {}
    
    # Continuity over globally common periods
    common_periods = (
        gdf[gdf["period_key"].isin(global_common_keys)][["period_key", "period_start", "period_end"]]
        .drop_duplicates()
    )
    keep_keys = _longest_continuous_block(common_periods)
    print(f"Global: keeping {len(keep_keys)} periods in longest continuous block")
    if len(keep_keys) < cfg.require_common_periods:
        return {}
    
    # Build per-ticker pivots on the globally aligned period set
    for ticker, g in gdf.groupby("ticker", sort=False):
        gg = g[g["period_key"].isin(keep_keys)].copy()
        if gg.empty:
            continue
        
        pivot_ret = (
            gg.pivot_table(
                index="model_id",
                columns="period_key",
                values="period_return",
                aggfunc="first",
            )
            .reindex(columns=keep_keys)
            .dropna(axis=0, how="any")
        )
        
        if pivot_ret.shape[1] < cfg.require_common_periods or pivot_ret.empty:
            continue
        
        # Optional extra metrics: avg_return_per_trade and num_trades
        meta_dict: Dict[str, pd.DataFrame] = {}
        if "avg_return_per_trade" in gg.columns:
            pivot_avg = (
                gg.pivot_table(
                    index="model_id",
                    columns="period_key",
                    values="avg_return_per_trade",
                    aggfunc="first",
                )
                .reindex(columns=keep_keys)
            )
            meta_dict["avg_return_per_trade"] = pivot_avg.reindex(index=pivot_ret.index)
        if "num_trades" in gg.columns:
            pivot_trades = (
                gg.pivot_table(
                    index="model_id",
                    columns="period_key",
                    values="num_trades",
                    aggfunc="first",
                )
                .reindex(columns=keep_keys)
            )
            meta_dict["num_trades"] = pivot_trades.reindex(index=pivot_ret.index)
        
        out_returns[ticker] = pivot_ret
        out_meta[ticker] = meta_dict
    
    return out_returns, out_meta




# =========================
# Import helpers (aligned CSV)
# =========================

def load_aligned_periods_from_csv(
    aligned_df: pd.DataFrame,
    cfg: Optional[MetaConfig] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, pd.DataFrame]]]:
    """
    Parse a wide aligned CSV (period_N_* columns) into per-ticker matrices.
    """
    with _timer("load_aligned_periods_from_csv"):
        df = aligned_df.copy()
        if "model_id" not in df.columns:
            df["model_id"] = build_model_id(df)
        if "ticker" not in df.columns:
            df["ticker"] = df.apply(infer_ticker, axis=1)
        
        period_idxs = extract_period_indices(df)
        if not period_idxs:
            raise ValueError("No period_N_* columns found in aligned CSV.")
        
        # Use date_range columns to preserve the aligned period order.
        period_keys: List[str] = []
        for i in period_idxs:
            date_col = f"period_{i}_date_range"
            if date_col not in df.columns:
                raise ValueError(f"Missing {date_col} in aligned CSV.")
            uniq = df[date_col].dropna().unique()
            if len(uniq) == 0:
                raise ValueError(f"No date ranges found in {date_col}.")
            if len(uniq) > 1:
                raise ValueError(f"Aligned CSV has multiple date ranges for {date_col} (count={len(uniq)}).")
            period_keys.append(str(uniq[0]))
        
        if len(set(period_keys)) != len(period_keys):
            raise ValueError("Aligned CSV has duplicate date_range values across period columns.")
        
        if cfg is not None and len(period_keys) < cfg.require_common_periods:
            raise ValueError("Aligned CSV has fewer common periods than required.")
        
        return_cols = []
        return_rename = {}
        for idx, i in enumerate(period_idxs):
            col = f"period_{i}_return"
            if col not in df.columns:
                raise ValueError(f"Missing {col} in aligned CSV.")
            return_cols.append(col)
            return_rename[col] = period_keys[idx]
        
        returns_wide = df[["model_id", "ticker"] + return_cols].rename(columns=return_rename)
        
        avg_wide = None
        avg_cols = []
        avg_rename = {}
        for idx, i in enumerate(period_idxs):
            col = f"period_{i}_avg_return_per_trade"
            if col in df.columns:
                avg_cols.append(col)
                avg_rename[col] = period_keys[idx]
        if avg_cols:
            avg_wide = df[["model_id", "ticker"] + avg_cols].rename(columns=avg_rename)
        
        num_wide = None
        num_cols = []
        num_rename = {}
        for idx, i in enumerate(period_idxs):
            col = f"period_{i}_num_trades"
            if col in df.columns:
                num_cols.append(col)
                num_rename[col] = period_keys[idx]
        if num_cols:
            num_wide = df[["model_id", "ticker"] + num_cols].rename(columns=num_rename)
        
        aligned_returns: Dict[str, pd.DataFrame] = {}
        aligned_meta: Dict[str, Dict[str, pd.DataFrame]] = {}
        
        for ticker, g in returns_wide.groupby("ticker", sort=False):
            mat = g.set_index("model_id")[period_keys]
            mat = mat.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
            if cfg is not None and mat.shape[0] < cfg.min_models_per_ticker:
                continue
            if cfg is not None and mat.shape[1] < cfg.require_common_periods:
                continue
            
            aligned_returns[ticker] = mat
            meta_dict: Dict[str, pd.DataFrame] = {}
            
            if avg_wide is not None:
                avg_mat = avg_wide[avg_wide["ticker"] == ticker].set_index("model_id")[period_keys]
                avg_mat = avg_mat.apply(pd.to_numeric, errors="coerce").reindex(index=mat.index)
                meta_dict["avg_return_per_trade"] = avg_mat
            if num_wide is not None:
                num_mat = num_wide[num_wide["ticker"] == ticker].set_index("model_id")[period_keys]
                num_mat = num_mat.apply(pd.to_numeric, errors="coerce").reindex(index=mat.index)
                meta_dict["num_trades"] = num_mat
            
            aligned_meta[ticker] = meta_dict
        
        return aligned_returns, aligned_meta


# =========================
# Export helpers
# =========================

def write_aligned_periods_to_csv(
    aligned_returns: Dict[str, pd.DataFrame],
    aligned_meta: Dict[str, Dict[str, pd.DataFrame]],
    original_df: pd.DataFrame,
    out_path: str,
) -> None:
    """
    Flatten the aligned per-ticker matrices back to a wide CSV.
    Output columns mirror the input non-period columns, followed by period_X_return/date_range pairs.
    """
    if not aligned_returns:
        raise ValueError("Aligned input is empty; nothing to write.")
    
    first = next(iter(aligned_returns.values()))
    period_keys = list(first.columns)
    
    meta_df = original_df.copy()
    if "model_id" not in meta_df.columns:
        meta_df["model_id"] = build_model_id(meta_df)
    if "ticker" not in meta_df.columns:
        meta_df["ticker"] = meta_df.apply(infer_ticker, axis=1)
    # Keep all non-period columns from the original file (requested list included)
    metadata_cols = [c for c in meta_df.columns if not c.startswith("period_") and c not in ["model_id", "ticker"]]
    meta_df = meta_df[metadata_cols + ["model_id", "ticker"]]
    
    rows = []
    for ticker, mat in aligned_returns.items():
        if list(mat.columns) != period_keys:
            raise ValueError(f"Ticker {ticker} has different period columns than the first ticker; cannot write unified file.")
        meta_dict = aligned_meta.get(ticker, {})
        avg_mat = meta_dict.get("avg_return_per_trade")
        num_mat = meta_dict.get("num_trades")
        for model_id, row in mat.iterrows():
            rec = {}
            meta = meta_df[meta_df["model_id"] == model_id]
            if not meta.empty:
                meta_row = meta.iloc[0].to_dict()
            else:
                meta_row = {c: np.nan for c in metadata_cols}
                meta_row["model_id"] = model_id
                meta_row["ticker"] = ticker
            for c in metadata_cols:
                rec[c] = meta_row.get(c, np.nan)
            for i, pk in enumerate(period_keys, start=1):
                rec[f"period_{i}_return"] = row[pk]
                rec[f"period_{i}_date_range"] = pk
                rec[f"period_{i}_avg_return_per_trade"] = avg_mat.loc[model_id, pk] if avg_mat is not None and model_id in avg_mat.index else np.nan
                rec[f"period_{i}_num_trades"] = num_mat.loc[model_id, pk] if num_mat is not None and model_id in num_mat.index else np.nan
            rows.append(rec)
    
    out_df = pd.DataFrame(rows)
    ordered_cols = metadata_cols + [
        col
        for i in range(1, len(period_keys) + 1)
        for col in [
            f"period_{i}_return",
            f"period_{i}_avg_return_per_trade",
            f"period_{i}_num_trades",
            f"period_{i}_date_range",
        ]
    ]
    ordered_cols = list(dict.fromkeys(ordered_cols))  # drop duplicate labels if any
    out_df = out_df.reindex(columns=ordered_cols)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote aligned periods to: {out_path} (rows={out_df.shape[0]}, periods={len(period_keys)})")


def compute_equity_curves(
    aligned: Dict[str, pd.DataFrame],
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


# =========================
# Meta model core calculations
# =========================

def robust_std(x: np.ndarray) -> float:
    """MAD-based robust std estimate."""
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad  # approx std under normality

def rolling_zscore(series: np.ndarray, window: int) -> np.ndarray:
    z = np.full_like(series, np.nan, dtype=float)
    for t in range(len(series)):
        lo = max(0, t - window + 1)
        w = series[lo:t+1]
        w = w[~np.isnan(w)]
        if len(w) < 2:
            continue
        mu = np.mean(w)
        sd = np.std(w, ddof=1)
        if sd < 1e-12:
            continue
        z[t] = (series[t] - mu) / sd
    return z

def rolling_zscore_v2(series: np.ndarray, window: int) -> np.ndarray:
    """
    Vectorized rolling z-score using pandas rolling stats.
    """
    with _timer("rolling_zscore_v2"):
        s = pd.Series(series, dtype=float)
        roll = s.rolling(window=window, min_periods=2)
        mu = roll.mean()
        sd = roll.std(ddof=1)
        z = (s - mu) / sd
        z = z.to_numpy()
        z[sd.to_numpy() < 1e-12] = np.nan
        return z

def map_z_to_alpha(z: float, cfg: MetaConfig) -> float:
    if np.isnan(z):
        return 0.5 * (cfg.alpha_low + cfg.alpha_high)
    zc = float(np.clip(z, cfg.z_low, cfg.z_high))
    # linear map
    frac = (zc - cfg.z_low) / (cfg.z_high - cfg.z_low + 1e-12)
    return cfg.alpha_low + frac * (cfg.alpha_high - cfg.alpha_low)

def ema_smooth(values: np.ndarray, alpha: float) -> np.ndarray:
    with _timer("ema_smooth"):
        out = np.full_like(values, np.nan, dtype=float)
        prev = np.nan
        for i, v in enumerate(values):
            if np.isnan(v):
                out[i] = prev
                continue
            if np.isnan(prev):
                prev = v
            else:
                prev = alpha * v + (1 - alpha) * prev
            out[i] = prev
        return out

def percentile_ranks_across_models(x: np.ndarray) -> np.ndarray:
    """
    Compute percentile ranks of x across models (vector length n_models), output in [0,1].
    NaNs remain NaN.
    """
    out = np.full_like(x, np.nan, dtype=float)
    mask = ~np.isnan(x)
    vals = x[mask]
    if len(vals) == 0:
        return out
    # rankdata without scipy
    order = np.argsort(vals)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(vals) + 1, dtype=float)
    out[mask] = (ranks - 1) / max(1, (len(vals) - 1))  # 0..1
    return out

def percentile_ranks_across_models_v2(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Vectorized percentile ranks along an axis; NaNs remain NaN.
    """
    with _timer("percentile_ranks_across_models_v2"):
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 1:
            mask = ~np.isnan(arr)
            if not np.any(mask):
                return np.full_like(arr, np.nan, dtype=float)
            vals = np.where(mask, arr, np.inf)
            order = np.argsort(vals, kind="quicksort")
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(1, arr.size + 1, dtype=float)
            denom = max(1, int(mask.sum()) - 1)
            out = np.full_like(arr, np.nan, dtype=float)
            out[mask] = (ranks[mask] - 1) / denom
            return out
        
        if axis not in (0, 1):
            raise ValueError("axis must be 0 or 1")
        
        if axis == 1:
            arr = arr.T
        
        mask = ~np.isnan(arr)
        vals = np.where(mask, arr, np.inf)
        order = np.argsort(vals, axis=0, kind="quicksort")
        ranks = np.empty_like(order, dtype=float)
        col_idx = np.arange(arr.shape[1])
        ranks[order, col_idx] = np.arange(1, arr.shape[0] + 1, dtype=float)[:, None]
        counts = mask.sum(axis=0)
        denom = np.maximum(1, counts - 1)
        out = (ranks - 1) / denom
        out[~mask] = np.nan
        if axis == 1:
            out = out.T
        return out

def compute_uniqueness_weights(returns_matrix: pd.DataFrame, cfg: MetaConfig) -> pd.Series:
    """
    Simple duplicate control:
      - compute correlation matrix of model return series
      - greedy clustering: any model with corr>=threshold joins existing cluster
      - weight = 1/sqrt(cluster_size), clipped by uniqueness_floor
    """
    with _timer("compute_uniqueness_weights"):
        # To keep scoring causal, default to neutral weights (no future leakage).
        return pd.Series(1.0, index=returns_matrix.index)
        
    X = returns_matrix.to_numpy(dtype=float)
    # Correlation across models; handle constant series
    with np.errstate(invalid="ignore"):
        corr = np.corrcoef(X)
    n = corr.shape[0]
    assigned = np.full(n, False)
    cluster_ids = np.full(n, -1, dtype=int)
    clusters: List[List[int]] = []
    
    for i in range(n):
        if assigned[i]:
            continue
        # start new cluster
        cid = len(clusters)
        members = [i]
        assigned[i] = True
        cluster_ids[i] = cid
        
        # add all j highly correlated to i (single-link-ish greedy)
        for j in range(i + 1, n):
            if assigned[j]:
                continue
            c = corr[i, j]
            if np.isnan(c):
                continue
            if c >= cfg.corr_cluster_threshold:
                assigned[j] = True
                cluster_ids[j] = cid
                members.append(j)
        
        clusters.append(members)
    
    cluster_sizes = np.array([len(cl) for cl in clusters], dtype=float)
    sizes_per_model = np.array([cluster_sizes[cluster_ids[i]] for i in range(n)], dtype=float)
    w = 1.0 / np.sqrt(sizes_per_model)
    w = np.clip(w, cfg.uniqueness_floor, 1.0)
    return pd.Series(w, index=returns_matrix.index)

def downside_cvar(values: np.ndarray, alpha: float) -> float:
    """
    CVaR of negative tail for values (already something like residuals).
    We compute CVaR on the *negative side*:
      tail = values <= quantile(values, alpha)
      return -mean(tail) if tail is negative, else 0
    """
    v = values[~np.isnan(values)]
    if len(v) < 5:
        return 0.0
    q = np.quantile(v, alpha)
    tail = v[v <= q]
    if len(tail) == 0:
        return 0.0
    m = float(np.mean(tail))
    return max(0.0, -m)

def downside_cvar_matrix_v2(
    resid: np.ndarray,
    lookback: int,
    alpha: float,
    min_count: int = 5,
    max_chunk_elems: int = 20_000_000,
    window_stride: int = 1,
) -> np.ndarray:
    """
    Vectorized downside CVaR over rolling windows (per model, per time).
    """
    with _timer("downside_cvar_matrix_v2"):
        if lookback <= 0:
            raise ValueError("lookback must be positive.")
        if window_stride <= 0:
            raise ValueError("window_stride must be positive.")
        resid = np.asarray(resid, dtype=float)
        n_models, T = resid.shape
        pad = np.full((n_models, lookback - 1), np.nan, dtype=float)
        padded = np.concatenate([pad, resid], axis=1)
        
        if T == 0:
            return np.zeros_like(resid, dtype=float)
        
        chunk_models = max(1, int(max_chunk_elems // max(1, (T * lookback))))
        chunk_models  = chunk_models * 2
        out = np.zeros((n_models, T), dtype=float)
        
        for start in range(0, n_models, chunk_models):
            end = min(n_models, start + chunk_models)
            chunk = padded[start:end]
            windows = sliding_window_view(chunk, window_shape=lookback, axis=1)
            if window_stride > 1:
                windows = windows[..., ::window_stride]
            valid = ~np.isnan(windows)
            counts = valid.sum(axis=2)
            q = np.nanquantile(windows, alpha, axis=2)
            tail_mask = valid & (windows <= q[:, :, None])
            tail_counts = tail_mask.sum(axis=2)
            tail_sums = np.where(tail_mask, windows, 0.0).sum(axis=2)
            tail_mean = np.divide(
                tail_sums,
                tail_counts,
                out=np.full_like(tail_sums, np.nan, dtype=float),
                where=tail_counts > 0,
            )
            out[start:end] = np.where(
                (counts >= min_count) & np.isfinite(tail_mean),
                np.maximum(0.0, -tail_mean),
                0.0,
            )
        
        return out

def compute_scores_for_ticker(returns_matrix: pd.DataFrame, cfg: MetaConfig) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Given returns_matrix (index=model_id, columns=ordered common date_ranges),
    compute per-period scores (same shape) and ticker_score per period.
    Returns:
      scores_df: index=model_id, columns=period columns (date_range strings)
      ticker_score: series indexed by period column
    """
    models = returns_matrix.index
    periods = list(returns_matrix.columns)
    R = returns_matrix.to_numpy(dtype=float)  # shape: (n_models, T)
    n_models, T = R.shape
    
    # 1) Per-period dispersion (within ticker)
    disp = np.array([robust_std(R[:, t]) for t in range(T)], dtype=float)
    z = rolling_zscore(disp, cfg.vol_window)
    alpha_raw = np.array([map_z_to_alpha(z[t], cfg) for t in range(T)], dtype=float)
    alpha_t = ema_smooth(alpha_raw, cfg.alpha_smooth)
    
    # 2) Percentile ranks per period
    Q = np.vstack([percentile_ranks_across_models(R[:, t]) for t in range(T)]).T  # (n_models, T)
    
    # 3) Adaptive EWMA momentum on Q (time-varying alpha)
    M = np.full_like(Q, np.nan, dtype=float)
    for t in range(T):
        a = alpha_t[t]
        if t == 0:
            M[:, t] = Q[:, t]
        else:
            M[:, t] = a * Q[:, t] + (1 - a) * M[:, t - 1]
    
    # Optional: limit to momentum_lookback by “resetting” far history effect (approx)
    # If you want strict lookback, you can compute weighted sums; recursive EWMA is usually fine.
    
    # 4) Empirical delta (no ML)
    D = np.full_like(Q, 0.0, dtype=float)
    D[:, 1:] = Q[:, 1:] - Q[:, :-1]
    base_forecast = M + cfg.delta_weight * D
    
    # 5) Ticker-local baseline
    if cfg.baseline_method == "mean":
        baseline = np.nanmean(base_forecast, axis=0)
    else:
        baseline = np.nanmedian(base_forecast, axis=0)
    rel = base_forecast - baseline  # (n_models, T)
    
    # 6) Confidence (training-free): stability of Q over conf_lookback, plus participation
    CONF = np.full_like(Q, np.nan, dtype=float)
    for t in range(T):
        lo = max(0, t - cfg.conf_lookback + 1)
        window = Q[:, lo:t+1]
        participation = np.mean(~np.isnan(window), axis=1)
        std = np.nanstd(window, axis=1, ddof=0)
        raw = 1.0 / (std + cfg.conf_eps)
        raw = raw * np.sqrt(participation)
        # Normalize to percentile within ticker (pool-safe)
        CONF[:, t] = percentile_ranks_across_models(raw)
    
    # 7) Risk penalty: downside CVaR on rank residuals (Q - 0.5)
    risk = np.zeros((n_models, T), dtype=float)
    resid = Q - 0.5
    for t in range(T):
        lo = max(0, t - cfg.risk_lookback + 1)
        for i in range(n_models):
            risk[i, t] = downside_cvar(resid[i, lo:t+1], cfg.cvar_alpha)
    risk_pen = cfg.cvar_risk_aversion * risk
    
    # 8) Uniqueness weighting (constant per model within ticker)
    uniq_w = compute_uniqueness_weights(returns_matrix, cfg).to_numpy(dtype=float)  # (n_models,)
    
    # Final score
    SCORE = (rel * CONF) - risk_pen
    SCORE = (uniq_w[:, None] * SCORE)
    
    scores_df = pd.DataFrame(SCORE, index=models, columns=periods)
    
    # Ticker gate score per period: median of TopM
    ticker_score = []
    Mgate = max(1, cfg.top_m_for_ticker_gate)
    for t, p in enumerate(periods):
        vals = scores_df[p].to_numpy()
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            ticker_score.append(np.nan)
            continue
        top = np.sort(vals)[::-1][:min(Mgate, len(vals))]
        ticker_score.append(np.median(top))
    ticker_score = pd.Series(ticker_score, index=periods, name="ticker_score")
    
    # Causal shift: score for period t is based on returns up to period t-1
    scores_df = scores_df.shift(axis=1)
    ticker_score = ticker_score.shift(1)
    # Drop leading all-NaN column from shift
    scores_df = scores_df.loc[:, scores_df.columns[scores_df.notna().any()]]
    ticker_score = ticker_score.loc[scores_df.columns]
    
    return scores_df, ticker_score


def compute_scores_for_ticker_v2(returns_matrix: pd.DataFrame, cfg: MetaConfig) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Vectorized version of compute_scores_for_ticker for faster scoring.
    """
    with _timer("compute_scores_for_ticker_v2"):
        models = returns_matrix.index
        periods = list(returns_matrix.columns)
        R = returns_matrix.to_numpy(dtype=float)
        n_models, T = R.shape
        
        # 1) Per-period dispersion (within ticker) using MAD
        counts = np.sum(~np.isnan(R), axis=0)
        med = np.nanmedian(R, axis=0)
        mad = np.nanmedian(np.abs(R - med), axis=0)
        disp = 1.4826 * mad
        disp[counts < 2] = np.nan
        
        # 2) Rolling z-score over dispersion
        z = rolling_zscore_v2(disp, cfg.vol_window)
        
        # 3) Map z to alpha (vectorized)
        alpha_raw = np.full_like(z, 0.5 * (cfg.alpha_low + cfg.alpha_high), dtype=float)
        mask = ~np.isnan(z)
        if np.any(mask):
            zc = np.clip(z[mask], cfg.z_low, cfg.z_high)
            frac = (zc - cfg.z_low) / (cfg.z_high - cfg.z_low + 1e-12)
            alpha_raw[mask] = cfg.alpha_low + frac * (cfg.alpha_high - cfg.alpha_low)
        alpha_t = ema_smooth(alpha_raw, cfg.alpha_smooth)
        
        # 4) Percentile ranks per period across models
        Q = percentile_ranks_across_models_v2(R, axis=0)
        
        # 5) Adaptive EWMA momentum on Q (time-varying alpha)
        M = np.full_like(Q, np.nan, dtype=float)
        for t in range(T):
            a = alpha_t[t]
            if t == 0:
                M[:, t] = Q[:, t]
            else:
                M[:, t] = a * Q[:, t] + (1 - a) * M[:, t - 1]
        
        # 6) Empirical delta (no ML)
        D = np.zeros_like(Q, dtype=float)
        D[:, 1:] = Q[:, 1:] - Q[:, :-1]
        base_forecast = M + cfg.delta_weight * D
        
        # 7) Ticker-local baseline
        if cfg.baseline_method == "mean":
            baseline = np.nanmean(base_forecast, axis=0)
        else:
            baseline = np.nanmedian(base_forecast, axis=0)
        rel = base_forecast - baseline
        
        # 8) Confidence (training-free)
        with _timer("compute_scores_for_ticker_v2.confidence"):
            Q_df = pd.DataFrame(Q)
            roll_std = Q_df.rolling(window=cfg.conf_lookback, axis=1, min_periods=1).std(ddof=0)
            participation = (
                Q_df.notna()
                .astype(float)
                .rolling(window=cfg.conf_lookback, axis=1, min_periods=1)
                .mean()
            )
            raw = (1.0 / (roll_std + cfg.conf_eps)) * np.sqrt(participation)
            CONF = percentile_ranks_across_models_v2(raw.to_numpy(), axis=0)
        
        # 9) Risk penalty: downside CVaR on rank residuals (Q - 0.5)
        with _timer("compute_scores_for_ticker_v2.cvar"):
            resid = Q - 0.5
            risk = downside_cvar_matrix_v2(
                resid,
                cfg.risk_lookback,
                cfg.cvar_alpha,
                window_stride=cfg.cvar_window_stride,
            )
            risk_pen = cfg.cvar_risk_aversion * risk
        
        # 10) Uniqueness weighting
        uniq_w = compute_uniqueness_weights(returns_matrix, cfg).to_numpy(dtype=float)
        
        SCORE = (rel * CONF) - risk_pen
        SCORE = (uniq_w[:, None] * SCORE)
        scores_df = pd.DataFrame(SCORE, index=models, columns=periods)
        
        # Ticker gate score per period: median of TopM
        ticker_score = []
        Mgate = max(1, cfg.top_m_for_ticker_gate)
        for p in periods:
            vals = scores_df[p].to_numpy()
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                ticker_score.append(np.nan)
                continue
            top = np.sort(vals)[::-1][:min(Mgate, len(vals))]
            ticker_score.append(np.median(top))
        ticker_score = pd.Series(ticker_score, index=periods, name="ticker_score")
        
        # Causal shift: score for period t is based on returns up to period t-1
        scores_df = scores_df.shift(axis=1)
        ticker_score = ticker_score.shift(1)
        scores_df = scores_df.loc[:, scores_df.columns[scores_df.notna().any()]]
        ticker_score = ticker_score.loc[scores_df.columns]
        
        return scores_df, ticker_score

def select_models_universal(
    aligned_by_ticker: Dict[str, pd.DataFrame],
    cfg: MetaConfig
) -> pd.DataFrame:
    """
    Runs meta-model for each ticker, then selects global Top-N each period using two-stage selection:
      - pick tickers by ticker_score
      - pick models within selected tickers by score (optionally per_ticker_cap)
    Returns a long dataframe of selections:
      [period_date_range, ticker, model_id, score, rank_global, rank_in_ticker, ticker_score]
    """
    # Compute per-ticker scores
    ticker_scores: Dict[str, pd.DataFrame] = {}
    ticker_gate: Dict[str, pd.Series] = {}
    common_periods_global: Optional[List[str]] = None
    
    for ticker, mat in aligned_by_ticker.items():
        scores_df, ticker_score = compute_scores_for_ticker(mat, cfg)
        ticker_scores[ticker] = scores_df
        ticker_gate[ticker] = ticker_score
        
        # Note: different tickers can have different common periods sets.
        # For universal selection, we select per period by period_end alignment via date_range strings.
        # We'll treat each date_range as its own decision point; union over all tickers.
        if common_periods_global is None:
            common_periods_global = list(scores_df.columns)
        else:
            # union later; do nothing here
            pass
    
    # Build union of all period keys (date_range strings)
    all_periods = sorted({p for tkr, s in ticker_scores.items() for p in s.columns})
    
    rows = []
    for p in all_periods:
        # 1) build ticker gate list for this period
        ticker_list = []
        for tkr, gate in ticker_gate.items():
            if p in gate.index and pd.notna(gate[p]):
                ts = float(gate[p])
                if cfg.min_ticker_score is not None and ts < cfg.min_ticker_score:
                    continue
                ticker_list.append((tkr, ts))
        if not ticker_list:
            continue
        
        # Choose tickers: if you want fixed K tickers, set it via top_n_global heuristic.
        # We'll pick all tickers, then model-level topN handles it; but two-stage helps fairness.
        # Here: keep tickers whose ticker score is within top ~sqrt(#tickers) (reasonable default)
        ticker_list.sort(key=lambda x: x[1], reverse=True)
        k_tickers = max(1, int(math.sqrt(len(ticker_list))))
        chosen_tickers = ticker_list[:k_tickers]
        
        # 2) gather candidate models from chosen tickers for this period
        candidates = []
        for tkr, ts in chosen_tickers:
            sdf = ticker_scores[tkr]
            if p not in sdf.columns:
                continue
            col = sdf[p].dropna()
            if col.empty:
                continue
            # rank within ticker
            col_sorted = col.sort_values(ascending=False)
            if cfg.per_ticker_cap is not None:
                col_sorted = col_sorted.iloc[:cfg.per_ticker_cap]
            for rank_in_ticker, (mid, sc) in enumerate(col_sorted.items(), start=1):
                candidates.append((tkr, mid, float(sc), rank_in_ticker, ts))
        
        if not candidates:
            continue
            
        # 3) global top-N
        candidates.sort(key=lambda x: x[2], reverse=True)
        topN = candidates[:min(cfg.top_n_global, len(candidates))]
        
        for rank_global, (tkr, mid, sc, rank_in_ticker, ts) in enumerate(topN, start=1):
            rows.append({
                "period_end": p,            # p is a pd.Timestamp
                "ticker": tkr,
                "model_id": mid,
                "score": sc,
                "rank_global": rank_global,
                "rank_in_ticker": rank_in_ticker,
                "ticker_score": ts,
            })
    
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["period_end", "rank_global"]).reset_index(drop=True)
    return out


def select_models_universal_v2(
    aligned_by_ticker: Union[Dict[str, pd.DataFrame], pd.DataFrame],
    cfg: MetaConfig
) -> pd.DataFrame:
    """
    Vectorized version of select_models_universal for faster selection.
    """
    with _timer("select_models_universal_v2"):
        if isinstance(aligned_by_ticker, pd.DataFrame):
            aligned_by_ticker, _ = load_aligned_periods_from_csv(aligned_by_ticker, cfg)
        elif isinstance(aligned_by_ticker, dict):
            sample = next(iter(aligned_by_ticker.values()), None)
            if isinstance(sample, pd.Series):
                aligned_by_ticker, _ = load_aligned_periods_from_csv(
                    pd.DataFrame(aligned_by_ticker),
                    cfg,
                )
        else:
            raise TypeError("aligned_by_ticker must be a dict of DataFrames or an aligned wide DataFrame.")
        
        score_frames = []
        gate_frames = []
        
        for ticker, mat in aligned_by_ticker.items():
            scores_df, ticker_score = compute_scores_for_ticker_v2(mat, cfg)
            if scores_df.empty:
                continue
            
            s_long = scores_df.stack(dropna=True).rename("score").reset_index()
            s_long = s_long.rename(columns={"level_0": "model_id", "level_1": "period_key"})
            s_long["ticker"] = ticker
            score_frames.append(s_long)
            
            gate = ticker_score.dropna().rename("ticker_score").reset_index()
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
        
        # Rank tickers per period and keep top sqrt(n_tickers)
        ticker_gate["ticker_rank"] = ticker_gate.groupby("period_key")["ticker_score"].rank(
            method="first",
            ascending=False,
        )
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
        
        candidates["rank_in_ticker"] = candidates.groupby(["ticker", "period_key"])["score"].rank(
            method="first",
            ascending=False,
        )
        if cfg.per_ticker_cap is not None:
            candidates = candidates[candidates["rank_in_ticker"] <= cfg.per_ticker_cap]
            if candidates.empty:
                return pd.DataFrame()
        
        candidates = candidates.sort_values(
            ["period_key", "score", "ticker_rank", "rank_in_ticker"],
            ascending=[True, False, True, True],
        )
        candidates["rank_global"] = candidates.groupby("period_key").cumcount() + 1
        candidates = candidates[candidates["rank_global"] <= cfg.top_n_global]
        
        out = candidates.rename(columns={"period_key": "period_end"})
        out = out[
            ["period_end", "ticker", "model_id", "score", "rank_global", "rank_in_ticker", "ticker_score"]
        ].sort_values(["period_end", "rank_global"]).reset_index(drop=True)
        return out


# =========================
# Main
# =========================


cfg = MetaConfig(
    # You can tweak these quickly:
    top_n_global=20,
    vol_window=4,
    momentum_lookback=12,
    conf_lookback=12,#12,
    risk_lookback=20,
    enable_uniqueness_weighting=False,  # keep causal by avoiding future-based uniqueness weights
    per_ticker_cap=10#25,  # prevents one ticker from dominating global topN
)

aligned_file_path = r"C:\Users\micha\myhome\algo\artifacts\period_returns\period_returns_weeks_1_aligned.csv"
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

selections = select_models_universal(aligned_returns, cfg)
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


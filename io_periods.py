import re
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from config import MetaConfig
from timing_utils import _timer


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
    for c in [
        "source_experiment_name",
        "source_run_id",
        "optuna_ticker_list",
        "outer_trial_number",
        "inner_trial_number",
        "buy_sell_signal_strategy",
        "model_type",
    ]:
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


def align_to_common_periods_per_ticker(
    long_df: pd.DataFrame,
    cfg: MetaConfig,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, pd.DataFrame]]]:
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

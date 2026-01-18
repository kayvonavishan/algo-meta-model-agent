from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from config import MetaConfig
from io_periods import load_aligned_periods_from_csv
from scoring import compute_scores_for_ticker, compute_scores_for_ticker_v2


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
    common_periods_global: Optional[list] = None
    
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
        k_tickers = max(1, int(np.sqrt(len(ticker_list))))
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
    
    # Rank tickers per period with deterministic tie-breakers.
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
    
    # Deterministic rank within ticker.
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

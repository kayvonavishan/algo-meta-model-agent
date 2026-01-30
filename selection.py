from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from config import MetaConfig
from io_periods import load_aligned_periods_from_csv
from scoring import compute_scores_for_ticker_v2


def select_models_universal_v2(
    aligned_by_ticker: Union[Dict[str, pd.DataFrame], pd.DataFrame],
    cfg: MetaConfig,
) -> pd.DataFrame:
    """
    Vectorized version of select_models_universal for faster selection.
    """
    if cfg.per_symbol_outer_trial_cap is not None and cfg.per_symbol_outer_trial_cap <= 0:
        raise ValueError("per_symbol_outer_trial_cap must be a positive int or None.")
    if cfg.include_n_top_tickers is not None and cfg.include_n_top_tickers <= 0:
        raise ValueError("include_n_top_tickers must be a positive int or None.")
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

    if cfg.include_n_top_tickers is None:
        chosen_tickers = ticker_gate
    else:
        chosen_tickers = ticker_gate[ticker_gate["ticker_rank"] <= cfg.include_n_top_tickers]
    
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

    if cfg.per_symbol_outer_trial_cap is not None:
        model_parts = candidates["model_id"].astype(str).str.split("|")
        inferred = []
        for parts in model_parts:
            nums = []
            for p in parts:
                if p.isdigit():
                    nums.append(p)
            outer = None
            # If we see consecutive numeric fields, assume (outer, inner, ...).
            for i in range(len(parts) - 1):
                if parts[i].isdigit() and parts[i + 1].isdigit():
                    outer = parts[i]
                    break
            if outer is None and nums:
                outer = nums[0]
            inferred.append(outer)
        outer_num = pd.to_numeric(pd.Series(inferred, index=candidates.index, dtype="object"), errors="coerce")

        # If we can't infer an outer trial, use a unique key per model_id so we don't unexpectedly apply the cap.
        candidates["_outer_trial_key"] = outer_num.where(~pd.isna(outer_num), candidates["model_id"].astype(str))
        candidates["_outer_trial_key"] = candidates["_outer_trial_key"].astype(str)
    
    # Deterministic rank within ticker.
    candidates = candidates.sort_values(
        ["ticker", "period_key", "score", "model_id"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )

    if cfg.per_symbol_outer_trial_cap is not None:
        candidates["rank_in_outer_trial"] = candidates.groupby(
            ["ticker", "period_key", "_outer_trial_key"],
            dropna=False,
        ).cumcount() + 1
        candidates = candidates[candidates["rank_in_outer_trial"] <= cfg.per_symbol_outer_trial_cap]
        if candidates.empty:
            return pd.DataFrame()

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

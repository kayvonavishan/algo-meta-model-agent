from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import MetaConfig


def make_test_config() -> MetaConfig:
    return MetaConfig(
        min_models_per_ticker=2,
        require_common_periods=8,
        vol_window=3,
        alpha_low=0.3,
        alpha_high=0.7,
        alpha_smooth=0.3,
        momentum_lookback=6,
        enable_momentum_lookback=False,
        delta_weight=0.2,
        conf_lookback=6,
        risk_lookback=6,
        cvar_alpha=0.1,
        cvar_risk_aversion=0.5,
        cvar_window_stride=1,
        baseline_method="median",
        enable_uniqueness_weighting=False,
        top_n_global=4,
        top_m_for_ticker_gate=3,
        per_ticker_cap=3,
        min_ticker_score=None,
    )


def _period_keys(
    start_date: str,
    n_periods: int,
    window_days: int,
) -> List[str]:
    starts = pd.date_range(start_date, periods=n_periods, freq=f"{window_days}D")
    ends = starts + pd.Timedelta(days=window_days - 1)
    return [f"{s.date()} to {e.date()}" for s, e in zip(starts, ends)]


def make_synthetic_aligned_returns(
    n_tickers: int = 2,
    n_models: int = 6,
    n_periods: int = 24,
    seed: int = 7,
    start_date: str = "2024-01-01",
    window_days: int = 7,
    include_trade_metrics: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, List[str]]:
    rng = np.random.default_rng(seed)
    period_keys = _period_keys(start_date, n_periods, window_days)
    aligned_returns: Dict[str, pd.DataFrame] = {}
    rows = []

    base_time = 0.002 * np.sin(np.linspace(0, 2 * np.pi, n_periods))

    for t_idx in range(n_tickers):
        ticker = f"tkr{t_idx}"
        model_ids = [f"{ticker}_m{m}" for m in range(n_models)]
        skills = np.linspace(-0.015, 0.02, n_models) + 0.002 * t_idx
        noise = rng.normal(0.0, 0.0015, size=(n_models, n_periods))
        epsilon = (
            1e-6
            * (np.arange(n_models)[:, None] + 1)
            * (np.arange(n_periods)[None, :] + 1)
        )
        mat = skills[:, None] + base_time + noise + epsilon + (0.0001 * t_idx)
        df = pd.DataFrame(mat, index=model_ids, columns=period_keys)
        df.index.name = "model_id"
        aligned_returns[ticker] = df

        if include_trade_metrics:
            for m_idx, model_id in enumerate(model_ids):
                for p_idx, key in enumerate(period_keys):
                    start_str, end_str = key.split(" to ")
                    period_start = pd.to_datetime(start_str)
                    period_end = pd.to_datetime(end_str)
                    num_trades = 10 + ((m_idx + p_idx) % 5)
                    avg_ret = float(df.iloc[m_idx, p_idx]) / max(1.0, num_trades)
                    rows.append({
                        "model_id": model_id,
                        "ticker": ticker,
                        "period": p_idx + 1,
                        "date_range": key,
                        "period_return": float(df.iloc[m_idx, p_idx]),
                        "avg_return_per_trade": avg_ret,
                        "num_trades": float(num_trades),
                        "period_start": period_start,
                        "period_end": period_end,
                    })

    long_df = pd.DataFrame(rows) if include_trade_metrics else pd.DataFrame()
    return aligned_returns, long_df, period_keys


def aligned_returns_to_wide_df(aligned_returns: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not aligned_returns:
        return pd.DataFrame()
    period_keys = list(next(iter(aligned_returns.values())).columns)
    rows = []
    for ticker, mat in aligned_returns.items():
        for model_id, row in mat.iterrows():
            rec = {"model_id": model_id, "ticker": ticker}
            for i, key in enumerate(period_keys, start=1):
                rec[f"period_{i}_return"] = float(row[key])
                rec[f"period_{i}_date_range"] = key
            rows.append(rec)
    return pd.DataFrame(rows)

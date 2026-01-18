from typing import List, Tuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from config import MetaConfig
from timing_utils import _timer


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


def _adaptive_momentum_recursive(Q: np.ndarray, alpha_t: np.ndarray) -> np.ndarray:
    n_models, T = Q.shape
    M = np.full_like(Q, np.nan, dtype=float)
    for t in range(T):
        a = alpha_t[t]
        if t == 0:
            M[:, t] = Q[:, t]
        else:
            M[:, t] = a * Q[:, t] + (1 - a) * M[:, t - 1]
    return M


def _adaptive_window_weights(alpha_t: np.ndarray, lo: int, hi: int) -> np.ndarray:
    length = hi - lo + 1
    weights = np.empty(length, dtype=float)
    decay = 1.0
    for offset in range(length):
        idx = hi - offset
        a = alpha_t[idx]
        weights[length - 1 - offset] = a * decay
        decay *= (1.0 - a)
    weight_sum = weights.sum()
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        return np.full(length, 1.0 / length, dtype=float)
    return weights / weight_sum


def _adaptive_momentum_window(
    Q: np.ndarray,
    alpha_t: np.ndarray,
    lookback: int,
) -> np.ndarray:
    if lookback <= 0:
        raise ValueError("lookback must be positive.")
    n_models, T = Q.shape
    M = np.full_like(Q, np.nan, dtype=float)
    for t in range(T):
        lo = max(0, t - lookback + 1)
        weights = _adaptive_window_weights(alpha_t, lo, t)
        window = Q[:, lo:t + 1]
        M[:, t] = window @ weights
        if np.isnan(window).any():
            invalid = np.isnan(window).any(axis=1)
            M[invalid, t] = np.nan
    return M


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


def compute_scores_for_ticker(
    returns_matrix: pd.DataFrame,
    cfg: MetaConfig,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Given returns_matrix (index=model_id, columns=ordered common date_ranges),
    compute per-period scores (same shape) and ticker_score per period.
    Returns:
      scores_df: index=model_id, columns=period columns (date_range strings)
      ticker_score: series indexed by period column
    """
    orig_models = returns_matrix.index
    sorted_matrix = returns_matrix.sort_index()
    models = sorted_matrix.index
    periods = list(sorted_matrix.columns)
    R = sorted_matrix.to_numpy(dtype=float)  # shape: (n_models, T)
    n_models, T = R.shape
    
    # 1) Per-period dispersion (within ticker)
    disp = np.array([robust_std(R[:, t]) for t in range(T)], dtype=float)
    z = rolling_zscore(disp, cfg.vol_window)
    alpha_raw = np.array([map_z_to_alpha(z[t], cfg) for t in range(T)], dtype=float)
    alpha_t = ema_smooth(alpha_raw, cfg.alpha_smooth)
    
    # 2) Percentile ranks per period
    Q = np.vstack([percentile_ranks_across_models(R[:, t]) for t in range(T)]).T  # (n_models, T)
    
    # 3) Adaptive EWMA momentum on Q (time-varying alpha)
    if cfg.enable_momentum_lookback:
        if cfg.momentum_lookback <= 0:
            raise ValueError("momentum_lookback must be positive when enable_momentum_lookback is True.")
        M = _adaptive_momentum_window(Q, alpha_t, cfg.momentum_lookback)
    else:
        M = _adaptive_momentum_recursive(Q, alpha_t)
    
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
    scores_df = scores_df.reindex(index=orig_models)
    ticker_score = ticker_score.loc[scores_df.columns]
    
    return scores_df, ticker_score


def compute_scores_for_ticker_v2(
    returns_matrix: pd.DataFrame,
    cfg: MetaConfig,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Vectorized version of compute_scores_for_ticker for faster scoring.
    """
    with _timer("compute_scores_for_ticker_v2"):
        orig_models = returns_matrix.index
        sorted_matrix = returns_matrix.sort_index()
        models = sorted_matrix.index
        periods = list(sorted_matrix.columns)
        R = sorted_matrix.to_numpy(dtype=float)
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
        if cfg.enable_momentum_lookback:
            if cfg.momentum_lookback <= 0:
                raise ValueError("momentum_lookback must be positive when enable_momentum_lookback is True.")
            with _timer("adaptive_momentum_window"):
                M = _adaptive_momentum_window(Q, alpha_t, cfg.momentum_lookback)
        else:
            M = _adaptive_momentum_recursive(Q, alpha_t)
    
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
        scores_df = scores_df.reindex(index=orig_models)
        ticker_score = ticker_score.loc[scores_df.columns]
    
        return scores_df, ticker_score

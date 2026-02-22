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


def compute_efficiency_ratio(returns: pd.DataFrame) -> pd.Series:
    """
    Efficiency ratio: net return / sum(abs(returns)) per strategy.
    Values are clipped to [-1, 1]; empty or flat series map to 0.
    """
    net = returns.sum(axis=0, skipna=True)
    path = returns.abs().sum(axis=0, skipna=True)
    counts = returns.count(axis=0)
    eff = net / path
    eff = eff.where(path != 0.0, 0.0)
    eff = eff.where(counts > 0, 0.0)
    eff = eff.fillna(0.0)
    return eff.clip(-1.0, 1.0)


def compute_win_rate(returns: pd.DataFrame, lookback: int) -> np.ndarray:
    """
    Compute rolling win-rate (fraction of positive returns) over lookback window.
    Returns matrix of shape (n_models, T) with values in [0, 1].
    """
    if lookback <= 0:
        raise ValueError("lookback must be positive.")
    positive = returns.gt(0).astype(float)
    positive = positive.where(returns.notna())
    win_rate = positive.rolling(window=lookback, min_periods=1).mean()
    win_rate = win_rate.fillna(0.0)
    return win_rate.to_numpy().T


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


def compute_rank_durability_raw(Q: np.ndarray, cap: int) -> np.ndarray:
    """
    Compute consecutive above-median tenure ending at t-1 for each period t.
    """
    Q = np.asarray(Q, dtype=float)
    cap = int(max(1, cap))
    n_models, T = Q.shape
    if T == 0:
        return np.zeros_like(Q, dtype=float)
    is_above = (Q > 0.5) & np.isfinite(Q)
    tenure_end = np.zeros((n_models, T), dtype=int)
    tenure_end[:, 0] = is_above[:, 0].astype(int)
    for t in range(1, T):
        inc = tenure_end[:, t - 1] + 1
        tenure_end[:, t] = np.where(is_above[:, t], np.minimum(inc, cap), 0)
    durability_raw = np.zeros((n_models, T), dtype=float)
    if T > 1:
        durability_raw[:, 1:] = tenure_end[:, :-1]
    return durability_raw


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


def _compute_downside_volatility_penalty(
    R: np.ndarray,
    lookback: int,
    threshold_z: float,
    weight: float,
) -> np.ndarray | None:
    if weight <= 0:
        return None
    if lookback <= 0:
        return None
    T = R.shape[1]
    L = min(lookback, T)
    if L < 2:
        return None
    R_negative = np.where(R < 0, R, np.nan)
    R_neg_df = pd.DataFrame(R_negative.T)
    downside_std = R_neg_df.rolling(window=L, min_periods=2).std(ddof=1).to_numpy().T
    ds_series = pd.DataFrame(downside_std.T)
    roll_mean = ds_series.rolling(window=L, min_periods=2).mean().to_numpy().T
    roll_std_ds = ds_series.rolling(window=L, min_periods=2).std(ddof=1).to_numpy().T
    z_downside = (downside_std - roll_mean) / (roll_std_ds + 1e-8)
    excess_z = np.maximum(0, z_downside - threshold_z)
    penalty = weight * excess_z
    penalty = np.where(np.isfinite(penalty), penalty, 0.0)
    return penalty


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
        downside_vol_pen = _compute_downside_volatility_penalty(
            R,
            cfg.downside_vol_lookback,
            cfg.downside_vol_threshold_z,
            cfg.downside_vol_cap_weight,
        )
    
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
        # Q shape: (n_models, T) with time axis last (most recent at end).
        Q = percentile_ranks_across_models_v2(R, axis=0)

        durability_norm = None
        if cfg.rank_durability_weight != 0:
            durability_raw = compute_rank_durability_raw(Q, cfg.rank_durability_cap)
            durability_norm = percentile_ranks_across_models_v2(durability_raw, axis=0)

        breakout_norm = None
        if cfg.breakout_weight != 0:
            L = cfg.breakout_lookback
            thr = cfg.breakout_threshold
            breakout_raw = np.zeros_like(Q, dtype=float)
            for t in range(T):
                start = max(0, t - L)
                if start >= t:
                    continue
                q_t = Q[:, t]
                q_prev = Q[:, start:t]
                q_prev_nan = np.isnan(q_prev).any(axis=1)
                was_below_recently = np.any(q_prev <= thr, axis=1) & ~q_prev_nan
                is_above = (q_t > thr) & np.isfinite(q_t)

                q_win = Q[:, start:t + 1]
                if q_win.shape[1] >= 2:
                    q_win_filled = np.where(np.isfinite(q_win), q_win, -np.inf)
                    above_int = (q_win_filled > thr).astype(int)
                    crossings = np.sum(np.abs(np.diff(above_int, axis=1)), axis=1)
                else:
                    crossings = np.zeros(n_models, dtype=int)

                eligible = is_above & was_below_recently
                if np.any(eligible):
                    penalized = 1.0 - cfg.breakout_oscillation_penalty
                    breakout_raw[eligible & (crossings <= 2), t] = 1.0
                    breakout_raw[eligible & (crossings > 2), t] = penalized
                breakout_raw[~np.isfinite(q_t), t] = 0.0

            breakout_norm = np.zeros_like(breakout_raw, dtype=float)
            for t in range(T):
                col = breakout_raw[:, t]
                finite = np.isfinite(col)
                if finite.sum() < 2:
                    continue
                col_vals = col[finite]
                if np.min(col_vals) == np.max(col_vals):
                    continue
                ranks = percentile_ranks_across_models_v2(col)
                ranks = np.where(np.isfinite(ranks), ranks, 0.0)
                breakout_norm[:, t] = ranks
    
        # 5) Adaptive EWMA momentum on Q (time-varying alpha)
        if cfg.enable_momentum_lookback:
            if cfg.momentum_lookback <= 0:
                raise ValueError("momentum_lookback must be positive when enable_momentum_lookback is True.")
            with _timer("adaptive_momentum_window"):
                M = _adaptive_momentum_window(Q, alpha_t, cfg.momentum_lookback)
        else:
            M = _adaptive_momentum_recursive(Q, alpha_t)

        # M shape: (n_models, T) adaptive momentum per period.

        rank_persist_norm = None
        if cfg.rank_persistence_weight != 0:
            rp_L = min(cfg.rank_persistence_lookback, T)
            if rp_L >= 2:
                above_median = (Q > 0.5).astype(float)
                above_median = np.where(np.isnan(Q), np.nan, above_median)
                Q_above_df = pd.DataFrame(above_median.T)
                rank_persist = Q_above_df.rolling(window=rp_L, min_periods=2).mean().to_numpy().T
                rank_persist_norm = percentile_ranks_across_models_v2(rank_persist, axis=0)
    
        mom_sharpe_norm = None
        if cfg.momentum_sharpe_weight != 0:
            L = min(cfg.momentum_sharpe_lookback, Q.shape[1])
            if L >= 2:
                Q_df = pd.DataFrame(Q)
                mom_vol = Q_df.rolling(
                    window=L,
                    axis=1,
                    min_periods=2,
                ).std(ddof=1).to_numpy()
                mom_vol[~np.isfinite(mom_vol)] = np.inf
                mom_sharpe = M / (mom_vol + 1e-6)
                mom_sharpe_norm = percentile_ranks_across_models_v2(mom_sharpe, axis=0)

        hit_asym_norm = None
        if cfg.hit_asymmetry_weight != 0:
            ha_L = min(cfg.hit_asymmetry_lookback, T)
            if ha_L >= 4:
                Q_df = pd.DataFrame(Q.T)
                R_df = pd.DataFrame(R.T)
                cs_median = R_df.median(axis=1)
                is_bad_period = cs_median < 0
                is_good_period = cs_median >= 0
                is_top = Q_df > cfg.hit_asymmetry_threshold

                top_and_bad = (is_top.T & is_bad_period).T.astype(float)
                bad_count = is_bad_period.astype(float)
                roll_top_bad = top_and_bad.rolling(window=ha_L, min_periods=2).sum()
                roll_bad = bad_count.rolling(window=ha_L, min_periods=2).sum()
                hit_rate_bad = roll_top_bad.div(roll_bad.where(roll_bad > 0), axis=0).fillna(0.0)

                top_and_good = (is_top.T & is_good_period).T.astype(float)
                good_count = is_good_period.astype(float)
                roll_top_good = top_and_good.rolling(window=ha_L, min_periods=2).sum()
                roll_good = good_count.rolling(window=ha_L, min_periods=2).sum()
                hit_rate_good = roll_top_good.div(roll_good.where(roll_good > 0), axis=0).fillna(0.0)

                asymmetry = hit_rate_bad / (hit_rate_good + 0.01)
                asymmetry = asymmetry.clip(0.0, 5.0)
                hit_asym_norm = percentile_ranks_across_models_v2(asymmetry.to_numpy().T, axis=0)

        # 6) Empirical delta (no ML)
        D = np.zeros_like(Q, dtype=float)
        D[:, 1:] = Q[:, 1:] - Q[:, :-1]

        # 6b) Efficiency ratio (momentum smoothness) over a rolling window
        returns_t = None
        eff_norm = None
        win_norm = None
        if cfg.efficiency_weight != 0 or cfg.win_rate_weight != 0:
            returns_t = sorted_matrix.T
        if cfg.efficiency_weight != 0:
            eff_lookback = cfg.momentum_lookback if cfg.enable_momentum_lookback else T
            if eff_lookback <= 0:
                eff_lookback = max(1, T)
            net = returns_t.rolling(window=eff_lookback, min_periods=1).sum()
            path = returns_t.abs().rolling(window=eff_lookback, min_periods=1).sum()
            counts = returns_t.rolling(window=eff_lookback, min_periods=1).count()
            eff = net.divide(path)
            eff = eff.where(path != 0.0, 0.0)
            eff = eff.where(counts > 0, 0.0)
            eff = eff.fillna(0.0).clip(-1.0, 1.0)
            eff_norm = ((eff + 1.0) * 0.5).clip(0.0, 1.0).to_numpy().T
        if cfg.win_rate_weight != 0:
            wr_lookback = cfg.momentum_lookback if cfg.enable_momentum_lookback else T
            if wr_lookback <= 0:
                wr_lookback = max(1, T)
            win_norm = compute_win_rate(returns_t, wr_lookback)

        base_forecast = M + cfg.delta_weight * D
        if eff_norm is not None:
            base_forecast = base_forecast + cfg.efficiency_weight * eff_norm
        if win_norm is not None:
            base_forecast = base_forecast + cfg.win_rate_weight * win_norm
        if mom_sharpe_norm is not None:
            base_forecast = base_forecast + cfg.momentum_sharpe_weight * mom_sharpe_norm
        if rank_persist_norm is not None:
            base_forecast = base_forecast + cfg.rank_persistence_weight * rank_persist_norm
        if cfg.rank_durability_weight != 0 and durability_norm is not None:
            base_forecast = base_forecast + cfg.rank_durability_weight * durability_norm
        if hit_asym_norm is not None:
            base_forecast = base_forecast + cfg.hit_asymmetry_weight * hit_asym_norm
        if breakout_norm is not None:
            base_forecast = base_forecast + cfg.breakout_weight * breakout_norm
    
        # 7) Ticker-local baseline
        if cfg.baseline_method == "mean":
            baseline = np.nanmean(base_forecast, axis=0)
        else:
            baseline = np.nanmedian(base_forecast, axis=0)
        if cfg.regime_baseline_adjust > 0:
            forecast_std = np.nanstd(base_forecast, axis=0)
            L = min(cfg.regime_dispersion_lookback, len(forecast_std))
            if L >= 2:
                fs_series = pd.Series(forecast_std, dtype=float)
                roll = fs_series.rolling(window=L, min_periods=2)
                roll_mean = roll.mean().to_numpy()
                roll_std = roll.std(ddof=1).to_numpy()
                z_regime = (forecast_std - roll_mean) / (roll_std + 1e-8)
                baseline_shift = cfg.regime_baseline_adjust * np.clip(z_regime, -1.5, 1.5)
                baseline_shift = np.where(np.isfinite(baseline_shift), baseline_shift, 0.0)
                if isinstance(baseline, pd.Series):
                    baseline = baseline + pd.Series(baseline_shift, index=baseline.index)
                else:
                    baseline = baseline + baseline_shift
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
        if downside_vol_pen is not None:
            SCORE = SCORE - downside_vol_pen
        SCORE = (uniq_w[:, None] * SCORE)

        if cfg.score_spread_boost_weight != 0:
            score_spread = np.full(T, np.nan, dtype=float)
            for t in range(T):
                col = SCORE[:, t]
                if np.isfinite(col).sum() < 2:
                    continue
                q75 = np.nanpercentile(col, 75)
                q25 = np.nanpercentile(col, 25)
                score_spread[t] = q75 - q25
            spread_series = pd.Series(score_spread, dtype=float)
            roll = spread_series.shift(1).rolling(
                window=cfg.score_spread_lookback,
                min_periods=2,
            )
            roll_mean = roll.mean().to_numpy()
            roll_std = roll.std(ddof=1).to_numpy()
            eps = 1e-12
            spread_z = (score_spread - roll_mean) / (roll_std + eps)
            excess = spread_z - cfg.score_spread_z_threshold
            excess = np.where(np.isfinite(excess), np.maximum(0.0, excess), 0.0)
            gate = 1.0 + cfg.score_spread_boost_weight * excess
            SCORE = SCORE * gate[None, :]
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

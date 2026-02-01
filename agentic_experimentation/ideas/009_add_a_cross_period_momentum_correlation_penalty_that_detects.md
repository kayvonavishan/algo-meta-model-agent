IDEA: Add a "cross-period momentum correlation penalty" that detects when a model's momentum signal is highly correlated with the ticker-wide average momentum, and downweights such models in favor of those with more independent momentum trajectories.

RATIONALE: The current scoring pipeline rewards models with high adaptive momentum (`M_{i,t}`) and stability (confidence), but doesn't consider whether a model's momentum trajectory is just riding the overall ticker trend vs. showing independent alpha. When most models in a ticker move together (e.g., all improving during a favorable regime), selecting the "top" models may just be picking the ones that happened to load most heavily on the common factor rather than those with genuine idiosyncratic edge. By penalizing models whose momentum series is highly correlated with the cross-sectional mean momentum, we favor models that demonstrate independent performance patterns—these are more likely to provide diversification benefit and to have captured something beyond beta-to-the-ticker. This is training-free (computed from the already-available momentum series), causal (uses only historical momentum), and addresses a form of redundancy that the current all-ones uniqueness weights don't capture (correlation of returns vs. correlation of momentum trajectories are different).

REQUIRED_CHANGES:

**1. Add new config parameters to `MetaConfig` (in `config.py`):**
```python
enable_momentum_independence: bool = True
momentum_corr_lookback: int = 8  # periods over which to measure momentum correlation
momentum_corr_penalty_weight: float = 0.15  # how much to penalize high correlation
momentum_corr_threshold: float = 0.7  # correlation above this triggers penalty
```

**2. Add momentum independence computation function in `scoring.py`:**

After computing adaptive momentum `M` in `compute_scores_for_ticker_v2`, compute the independence penalty:

```python
def compute_momentum_independence_penalty(momentum_df, lookback=8, threshold=0.7):
    """
    Compute a penalty for models whose momentum is highly correlated with the 
    ticker-wide average momentum.
    
    Parameters:
    - momentum_df: DataFrame of adaptive momentum (models x periods)
    - lookback: number of periods for rolling correlation
    - threshold: correlation above which penalty kicks in
    
    Returns:
    - penalty_df: DataFrame of penalties in [0, 1] (models x periods)
                  0 = independent momentum, 1 = perfectly correlated with average
    """
    penalty_df = pd.DataFrame(0.0, index=momentum_df.index, columns=momentum_df.columns)
    
    # Compute cross-sectional mean momentum per period
    mean_momentum = momentum_df.mean(axis=0)  # Series, one value per period
    
    for t_idx, period in enumerate(momentum_df.columns):
        if t_idx < lookback:
            # Not enough history for reliable correlation
            continue
        
        # Get lookback window
        start_idx = t_idx - lookback + 1
        window_cols = momentum_df.columns[start_idx:t_idx + 1]
        
        # Mean momentum over window
        mean_mom_window = mean_momentum[window_cols].values
        
        for i, model in enumerate(momentum_df.index):
            model_mom_window = momentum_df.loc[model, window_cols].values
            
            # Skip if insufficient valid data
            valid_mask = ~(np.isnan(model_mom_window) | np.isnan(mean_mom_window))
            if valid_mask.sum() < 3:
                continue
            
            # Compute correlation
            corr = np.corrcoef(
                model_mom_window[valid_mask], 
                mean_mom_window[valid_mask]
            )[0, 1]
            
            if np.isnan(corr):
                continue
            
            # Convert to penalty: 0 if corr <= threshold, scales up to 1 if corr = 1
            if corr > threshold:
                penalty = (corr - threshold) / (1.0 - threshold)
            else:
                penalty = 0.0
            
            penalty_df.iloc[i, t_idx] = penalty
    
    return penalty_df
```

**3. Integrate into `compute_scores_for_ticker_v2` in `scoring.py`:**

After computing adaptive momentum `M` and before final score assembly:

```python
# After: M = _adaptive_momentum_window(Q, alpha_per_period, lookback)

# Compute momentum independence penalty if enabled
if cfg.enable_momentum_independence:
    momentum_corr_penalty = compute_momentum_independence_penalty(
        M,
        lookback=cfg.momentum_corr_lookback,
        threshold=cfg.momentum_corr_threshold
    )
else:
    momentum_corr_penalty = pd.DataFrame(0.0, index=M.index, columns=M.columns)
```

**4. Modify final score combination:**

Change from:
```python
SCORE = (rel * CONF) - risk_pen
```

To:
```python
SCORE = (rel * CONF) - risk_pen - cfg.momentum_corr_penalty_weight * momentum_corr_penalty
```

**Toy example:**

Consider three models in a ticker over 8 periods:

```
Mean momentum (ticker avg): [0.40, 0.45, 0.50, 0.55, 0.52, 0.48, 0.53, 0.58]

Model A momentum:           [0.42, 0.47, 0.52, 0.57, 0.53, 0.49, 0.55, 0.60]
- Correlation with mean: ~0.98 (moves almost identically)
- Penalty: (0.98 - 0.7) / 0.3 = 0.93

Model B momentum:           [0.35, 0.55, 0.45, 0.60, 0.40, 0.58, 0.48, 0.62]
- Correlation with mean: ~0.4 (choppy, independent path)
- Penalty: 0.0 (below threshold)

Model C momentum:           [0.50, 0.48, 0.55, 0.52, 0.58, 0.54, 0.60, 0.56]
- Correlation with mean: ~0.75 (moderately correlated)
- Penalty: (0.75 - 0.7) / 0.3 = 0.17
```

Even if Model A has the highest absolute momentum, its high correlation penalty would reduce its final score, potentially allowing Model B (with genuinely independent alpha) to be selected instead.

**5. Why this differs from existing uniqueness weighting:**

- **Uniqueness (placeholder)**: Intended to cluster models by return correlation and downweight redundant ones. Looks at correlation of *returns*.
- **Momentum independence**: Looks at correlation of *momentum trajectories* with the ticker average. A model could have unique returns but still be "riding the wave" in terms of when it improves/deteriorates.

These are complementary: returns correlation captures "do these models make money at the same time?", momentum correlation captures "do these models' relative standings improve/deteriorate together?"

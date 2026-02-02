IDEA: Add a "rank persistence" bonus that measures the autocorrelation of a model's percentile ranks over a lookback window, rewarding models whose cross-sectional position tends to persist from period to period and penalizing models with mean-reverting rank behavior.

RATIONALE: The current adaptive momentum (`M_{i,t}`) smooths ranks over time, and the confidence signal measures rank volatility (stability). However, neither directly captures whether a model's rank at time `t` is predictive of its rank at time `t+1`. A model with low rank volatility (high confidence) could still exhibit mean-reverting behavior where good periods are followed by bad periods, making momentum signals less actionable. Conversely, a model with moderate volatility but high positive autocorrelation (good periods tend to follow good periods) is more likely to continue performing well when selected. Rank autocorrelation provides a direct measure of "if this model is ranked highly now, how likely is it to remain highly ranked next period?"—exactly the predictive question the meta model cares about. This is training-free (computed from the already-available rank series), causal (uses only historical ranks), and captures something orthogonal to both confidence (level of volatility) and streak bonus (counting consecutive above-median periods without measuring statistical persistence strength).

REQUIRED_CHANGES:

**1. Add new config parameters to `MetaConfig` (in `config.py`):**
```python
enable_rank_persistence: bool = True
persistence_lookback: int = 10  # periods over which to measure rank autocorrelation
persistence_weight: float = 0.12  # contribution to final score
persistence_lag: int = 1  # lag for autocorrelation (1 = adjacent periods)
```

**2. Add rank persistence computation function in `scoring.py`:**

After computing percentile ranks `Q` in `compute_scores_for_ticker_v2`, compute the persistence bonus:

```python
def compute_rank_persistence(ranks_df, lookback=10, lag=1):
    """
    Compute the autocorrelation of each model's percentile ranks over a rolling window.
    
    High positive autocorrelation means ranks persist (good predictor of next period).
    Low or negative autocorrelation means ranks mean-revert (less predictable).
    
    Parameters:
    - ranks_df: DataFrame of percentile ranks (models x periods)
    - lookback: number of periods for rolling autocorrelation
    - lag: lag for autocorrelation (typically 1)
    
    Returns:
    - persistence_df: DataFrame of persistence scores in [-1, 1] (models x periods)
                      Positive = persistent ranks, Negative = mean-reverting ranks
    """
    persistence_df = pd.DataFrame(0.0, index=ranks_df.index, columns=ranks_df.columns)
    
    for t_idx, period in enumerate(ranks_df.columns):
        if t_idx < lookback + lag:
            # Not enough history for reliable autocorrelation
            continue
        
        # Get lookback window (excluding most recent 'lag' periods for the lagged series)
        start_idx = t_idx - lookback - lag + 1
        end_idx = t_idx + 1
        window_cols = ranks_df.columns[start_idx:end_idx]
        
        for i, model in enumerate(ranks_df.index):
            rank_series = ranks_df.loc[model, window_cols].values
            
            # Skip if too many NaNs
            valid_mask = ~np.isnan(rank_series)
            if valid_mask.sum() < lookback // 2:
                continue
            
            # Compute lag-1 autocorrelation
            # ranks[:-lag] vs ranks[lag:]
            series_current = rank_series[lag:]
            series_lagged = rank_series[:-lag]
            
            # Mask for pairs where both are valid
            pair_valid = ~(np.isnan(series_current) | np.isnan(series_lagged))
            if pair_valid.sum() < 3:
                continue
            
            autocorr = np.corrcoef(
                series_current[pair_valid],
                series_lagged[pair_valid]
            )[0, 1]
            
            if np.isnan(autocorr):
                autocorr = 0.0
            
            persistence_df.iloc[i, t_idx] = autocorr
    
    return persistence_df
```

**3. Integrate into `compute_scores_for_ticker_v2` in `scoring.py`:**

After computing percentile ranks `Q` and before final score assembly:

```python
# After: Q = percentile_ranks_across_models_v2(R)
# ...existing momentum, delta, confidence, risk computations...

# Compute rank persistence bonus if enabled
if cfg.enable_rank_persistence:
    rank_persistence = compute_rank_persistence(
        Q,
        lookback=cfg.persistence_lookback,
        lag=cfg.persistence_lag
    )
else:
    rank_persistence = pd.DataFrame(0.0, index=Q.index, columns=Q.columns)
```

**4. Modify final score combination:**

Change from:
```python
SCORE = (rel * CONF) - risk_pen
```

To:
```python
SCORE = (rel * CONF) - risk_pen + cfg.persistence_weight * rank_persistence
```

**Toy example:**

Consider two models with similar momentum and confidence:

```
Model A ranks (last 10 periods): [0.70, 0.72, 0.68, 0.74, 0.71, 0.73, 0.69, 0.75, 0.72, 0.74]
- Lag-1 autocorrelation: ~0.6 (high persistence - good periods follow good periods)
- persistence_score = 0.6

Model B ranks (last 10 periods): [0.70, 0.35, 0.75, 0.30, 0.72, 0.38, 0.68, 0.42, 0.73, 0.40]
- Lag-1 autocorrelation: ~-0.8 (strong mean reversion - good periods followed by bad)
- persistence_score = -0.8

Both have similar average rank (~0.55), similar volatility.
But Model A's good ranks persist, while Model B alternates wildly.
With persistence_weight=0.12: Model A gets +0.072 boost, Model B gets -0.096 penalty.
```

**5. Why this differs from existing signals:**

- **Confidence**: Measures rank *volatility* (std of ranks). High confidence = stable ranks. But doesn't distinguish between "stable at a good level" vs "stable but mean-reverting around median."
- **Streak bonus (idea 004)**: Counts consecutive above-median periods. Captures direction but not statistical strength of persistence.
- **Rank persistence**: Directly measures autocorrelation—the statistical relationship between rank at t and rank at t+1. A model could have:
  - High confidence (low volatility) but low persistence (ranks clustered but unpredictable period-to-period)
  - Low confidence (high volatility) but high persistence (volatile but trending consistently in one direction)
  - High streak count but low persistence (currently on a streak that's statistically unlikely to continue)

Autocorrelation is the most direct answer to "if I select this model because it's ranked well now, will it likely be ranked well next period?"

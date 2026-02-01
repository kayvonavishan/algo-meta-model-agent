IDEA: Add a "streak consistency" bonus that rewards models with consecutive periods of above-median rank performance, penalizing models that alternate frequently between good and bad periods even if their average momentum looks similar.

RATIONALE: The current adaptive momentum (`M_{i,t}`) smooths ranks over a lookback window, but two models can have identical momentum scores with very different paths: one steadily above median for 6 straight periods, another oscillating wildly but averaging to the same value. The steady model is more likely to continue performing well (persistence), while the oscillating model may just be lucky in timing. A streak bonus captures this "consistency of direction" that raw momentum averages wash out. This is training-free (just counting consecutive above/below-median periods in recent history) and complements existing confidence (which measures rank volatility, not directional persistence).

REQUIRED_CHANGES:

**1. Add new config parameters to `MetaConfig` (in `config.py`):**
```python
enable_streak_bonus: bool = True
streak_lookback: int = 8  # how many recent periods to check for streaks
streak_weight: float = 0.10  # how much to boost/penalize based on streak
streak_threshold: float = 0.5  # rank threshold for "good" period (0.5 = above median)
```

**2. Add streak computation function in `scoring.py`:**

```python
def compute_streak_bonus(ranks_df, lookback, threshold=0.5):
    """
    Compute a streak bonus for each model at each period.
    
    A "streak" is consecutive periods where rank >= threshold (above median).
    Returns a value in [-1, 1]:
    - Positive: recent streak of good performance
    - Negative: recent streak of poor performance
    - Near zero: alternating/mixed performance
    
    Parameters:
    - ranks_df: DataFrame of percentile ranks (models x periods)
    - lookback: number of recent periods to examine
    - threshold: rank threshold for "good" period
    
    Returns:
    - streak_df: DataFrame of streak scores (models x periods)
    """
    streak_df = pd.DataFrame(index=ranks_df.index, columns=ranks_df.columns, dtype=float)
    
    for t_idx, period in enumerate(ranks_df.columns):
        if t_idx < 1:
            streak_df[period] = 0.0
            continue
        
        # Get recent window
        start_idx = max(0, t_idx - lookback)
        window_cols = ranks_df.columns[start_idx:t_idx]  # exclude current period (causal)
        
        if len(window_cols) == 0:
            streak_df[period] = 0.0
            continue
        
        window_ranks = ranks_df[window_cols].values  # (n_models, window_len)
        above_threshold = (window_ranks >= threshold).astype(float)
        
        # Count longest recent streak (from most recent backward)
        # and compute a normalized streak score
        n_periods = above_threshold.shape[1]
        
        for i, model in enumerate(ranks_df.index):
            model_above = above_threshold[i, :]
            
            # Count current streak from end
            current_streak = 0
            streak_positive = True
            for j in range(n_periods - 1, -1, -1):
                if np.isnan(ranks_df[window_cols[j]].iloc[i]):
                    break
                if j == n_periods - 1:
                    streak_positive = model_above[j] == 1
                if model_above[j] == (1 if streak_positive else 0):
                    current_streak += 1
                else:
                    break
            
            # Normalize: streak of full lookback -> +/-1
            # Also factor in total good periods in window for stability
            total_good = np.nanmean(model_above)
            streak_norm = current_streak / lookback
            
            # Final score: weighted combo of streak length and overall consistency
            if streak_positive:
                streak_df.iloc[i, t_idx] = 0.6 * streak_norm + 0.4 * (total_good - 0.5) * 2
            else:
                streak_df.iloc[i, t_idx] = -0.6 * streak_norm + 0.4 * (total_good - 0.5) * 2
    
    return streak_df
```

**3. Integrate into `compute_scores_for_ticker_v2` in `scoring.py`:**

After computing percentile ranks `Q` and before final score assembly:

```python
# After: Q = percentile_ranks_across_models_v2(R)
# ...existing momentum, delta, confidence, risk computations...

# Compute streak bonus if enabled
if cfg.enable_streak_bonus:
    streak_bonus = compute_streak_bonus(
        Q, 
        lookback=cfg.streak_lookback,
        threshold=cfg.streak_threshold
    )
else:
    streak_bonus = pd.DataFrame(0.0, index=Q.index, columns=Q.columns)
```

**4. Modify final score combination:**

Change from:
```python
SCORE = (rel * CONF) - risk_pen
```

To:
```python
SCORE = (rel * CONF) - risk_pen + cfg.streak_weight * streak_bonus
```

**Toy example:**

Consider two models over 8 periods with identical average ranks:

```
Model A ranks: [0.55, 0.60, 0.58, 0.62, 0.57, 0.61, 0.59, 0.63]
- All above 0.5, consistent streak of 8
- streak_score ≈ +0.8

Model B ranks: [0.30, 0.80, 0.35, 0.75, 0.40, 0.70, 0.45, 0.65]
- Alternating above/below 0.5, streak of 1
- streak_score ≈ +0.1 (slightly positive due to last period good)

Both have similar mean rank (~0.57), but Model A gets a meaningful streak bonus.
```

**5. Optional sweep parameters:**

Add to `build_config_grid_v1` if you want to tune:
```python
'enable_streak_bonus': [True, False],
'streak_lookback': [6, 8, 12],
'streak_weight': [0.05, 0.10, 0.15],
```

IDEA: Add a regime-aware momentum decay that shortens the effective lookback window during high-volatility regimes and lengthens it during stable regimes.

RATIONALE: The current adaptive alpha adjusts how much weight recent periods get, but the lookback window itself (`momentum_lookback`) remains fixed. During volatile regimes (high dispersion), models' relative performance changes quickly, so older rankings become stale faster. Conversely, during stable regimes (low dispersion), longer histories provide more signal. By dynamically adjusting the effective lookback window—not just the weighting within it—we can be more responsive when needed and more stable when appropriate. This complements the existing adaptive alpha mechanism: alpha controls "how fast we update within the window," while regime-aware decay controls "how far back the window should reach."

REQUIRED_CHANGES:

**1. Add new config parameters to `MetaConfig` (in `config.py`):**
- `enable_regime_decay: bool = True` — flag to enable regime-aware lookback decay
- `lookback_min: int = 3` — minimum effective lookback periods (used in high-volatility regimes)
- `lookback_max: int = 12` — maximum effective lookback periods (used in stable regimes)
- `decay_sensitivity: float = 0.5` — controls how aggressively the lookback adjusts to dispersion z-scores (0 = no adjustment, 1 = full range)

**2. Modify dispersion calculation in `scoring.py` (`compute_scores_for_ticker_v2`):**

Currently, dispersion z-scores (`disp_z`) are computed in Step 1 to determine adaptive alpha. Extend this to also compute an effective lookback length per period:

```python
# After computing disp_z for each period:
if config.enable_regime_decay:
    # Map z-score to effective lookback length
    # High z (high dispersion) -> shorter lookback
    # Low z (low dispersion) -> longer lookback
    
    z_clipped = np.clip(disp_z, config.z_low, config.z_high)
    z_norm = (z_clipped - config.z_low) / (config.z_high - config.z_low)  # [0,1]
    
    # Invert: high dispersion (z_norm near 1) -> use lookback_min
    #         low dispersion (z_norm near 0) -> use lookback_max
    decay_frac = 1.0 - (config.decay_sensitivity * z_norm)
    effective_lookback = (
        config.lookback_min + 
        decay_frac * (config.lookback_max - config.lookback_min)
    )
    
    # Round to integer periods
    effective_lookback = np.round(effective_lookback).astype(int)
else:
    # Use fixed momentum_lookback for all periods
    effective_lookback = np.full(len(periods), config.momentum_lookback)
```

Toy example (with defaults `lookback_min=3`, `lookback_max=12`, `z_low=-1`, `z_high=1`, `decay_sensitivity=0.5`):

- Period with `disp_z = -1.0` (low dispersion):
  - `z_norm = 0.0`
  - `decay_frac = 1.0 - (0.5 * 0.0) = 1.0`
  - `effective_lookback = 3 + 1.0 * 9 = 12` periods
  
- Period with `disp_z = 0.0` (typical):
  - `z_norm = 0.5`
  - `decay_frac = 1.0 - (0.5 * 0.5) = 0.75`
  - `effective_lookback = 3 + 0.75 * 9 = 9.75 → 10` periods
  
- Period with `disp_z = 1.0` (high dispersion):
  - `z_norm = 1.0`
  - `decay_frac = 1.0 - (0.5 * 1.0) = 0.5`
  - `effective_lookback = 3 + 0.5 * 9 = 7.5 → 8` periods

**3. Update momentum calculation to use variable lookback:**

In `_adaptive_momentum_window` (within `scoring.py`), replace the fixed `lookback` parameter with the per-period `effective_lookback` array:

```python
def _adaptive_momentum_window(ranks_df, alpha_per_period, effective_lookback_per_period):
    """
    Compute momentum using a variable lookback window per period.
    
    Parameters:
    - ranks_df: DataFrame of percentile ranks (models x periods)
    - alpha_per_period: Series of adaptive alpha values per period
    - effective_lookback_per_period: array-like of integer lookback lengths per period
    
    Returns:
    - momentum_df: DataFrame of adaptive momentum scores (models x periods)
    """
    momentum = pd.DataFrame(index=ranks_df.index, columns=ranks_df.columns, dtype=float)
    
    for t_idx, period in enumerate(ranks_df.columns):
        if t_idx == 0:
            momentum[period] = ranks_df[period]
            continue
        
        # Get the lookback length for this period
        L = int(effective_lookback_per_period[t_idx])
        
        # Ensure we have enough history
        start_idx = max(0, t_idx - L + 1)
        window_cols = ranks_df.columns[start_idx : t_idx + 1]
        
        # Get alpha for this period
        alpha_t = alpha_per_period.iloc[t_idx]
        
        # Compute exponential weights (most recent period gets highest weight)
        window_len = len(window_cols)
        weights = np.array([(1 - alpha_t) ** (window_len - 1 - i) for i in range(window_len)])
        weights = weights / weights.sum()  # normalize
        
        # Weighted average of ranks in the window
        window_ranks = ranks_df[window_cols].values  # (n_models, window_len)
        momentum[period] = np.nansum(window_ranks * weights, axis=1)
    
    return momentum
```

**4. Integration with existing adaptive alpha:**

The key insight is that these two mechanisms are complementary:
- **Adaptive alpha** (existing): adjusts the exponential weighting *within* the lookback window
- **Regime decay** (new): adjusts the *size* of the lookback window itself

Together, they provide two degrees of freedom:
- In high-dispersion regimes: shorter window (fewer old periods included) + higher alpha (recent periods weighted more heavily within that shorter window)
- In low-dispersion regimes: longer window (more history included) + lower alpha (flatter weighting across that longer window)

**5. Update sweep producer (`adaptive_vol_momentum.py`):**

Add the new config parameters to the hyperparameter sweep grid if you want to tune them:

```python
config_sweep_params = {
    'enable_regime_decay': [True, False],
    'lookback_min': [2, 3, 4],
    'lookback_max': [10, 12, 15],
    'decay_sensitivity': [0.3, 0.5, 0.7],
    # ... existing params
}
```

Or keep them fixed at reasonable defaults initially to isolate the impact.

**6. Validation:**

- Ensure that when `enable_regime_decay=False`, behavior is identical to current implementation
- Add assertions that `effective_lookback` always satisfies `lookback_min ≤ effective_lookback ≤ lookback_max`
- Log the distribution of effective lookback values across periods to verify regime adaptation is working as expected
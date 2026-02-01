IDEA: Add a "momentum acceleration" signal that captures the second derivative of rank momentum (change in momentum trend), rewarding models whose momentum is accelerating upward and penalizing models whose momentum is decelerating, even if their current momentum level looks similar.

RATIONALE: The current pipeline computes adaptive momentum `M_{i,t}` (a smoothed first-order signal of ranks) and adds a delta term `D_{i,t} = Q_{i,t} - Q_{i,t-1}` (one-period rank change). However, the delta term only captures instantaneous velocity, not whether that velocity is itself improving or deteriorating. Two models can have identical momentum `M` and identical delta `D` at period `t`, yet one might be on an accelerating upward trajectory (momentum improving each period) while the other is on a decelerating path (momentum flattening or about to reverse). By computing a second derivative—the change in momentum itself over a short window—we can identify models that are "picking up steam" vs "losing steam" before it shows up in the first-order signals. This is training-free (just differencing the already-computed momentum series), complements existing signals without redundancy, and provides an early-warning signal for momentum regime changes.

REQUIRED_CHANGES:

**1. Add new config parameters to `MetaConfig` (in `config.py`):**
```python
enable_momentum_accel: bool = True
accel_lookback: int = 3  # periods over which to measure momentum change
accel_weight: float = 0.10  # contribution to final score
accel_smoothing: float = 0.5  # EMA smoothing on raw acceleration to reduce noise
```

**2. Add acceleration computation function in `scoring.py`:**

After computing adaptive momentum `M` in `compute_scores_for_ticker_v2`, compute the acceleration signal:

```python
def compute_momentum_acceleration(momentum_df, lookback=3, smoothing=0.5):
    """
    Compute the acceleration (second derivative) of momentum.
    
    Acceleration is positive when momentum is increasing over time,
    negative when momentum is decreasing.
    
    Parameters:
    - momentum_df: DataFrame of adaptive momentum (models x periods)
    - lookback: number of periods to measure momentum change
    - smoothing: EMA smoothing factor for the acceleration signal
    
    Returns:
    - accel_df: DataFrame of acceleration scores (models x periods)
    """
    # First difference of momentum (velocity of momentum)
    momentum_diff = momentum_df.diff(axis=1)
    
    # Rolling mean of momentum differences over lookback window
    # This captures sustained acceleration vs one-off jumps
    accel_raw = momentum_diff.rolling(window=lookback, axis=1, min_periods=1).mean()
    
    # Apply EMA smoothing to reduce noise
    accel_smooth = accel_raw.ewm(alpha=smoothing, axis=1).mean()
    
    # Normalize to comparable scale: convert to cross-sectional z-score per period
    accel_norm = accel_smooth.apply(
        lambda col: (col - col.mean()) / (col.std() + 1e-8), axis=0
    )
    
    # Clip to prevent extreme values from dominating
    accel_norm = accel_norm.clip(-3, 3) / 3  # scale to roughly [-1, 1]
    
    return accel_norm
```

**3. Integrate into `compute_scores_for_ticker_v2` in `scoring.py`:**

After computing adaptive momentum `M` and before final score assembly:

```python
# After: M = _adaptive_momentum_window(Q, alpha_per_period, lookback)
# ...existing delta, baseline, confidence, risk computations...

# Compute momentum acceleration if enabled
if cfg.enable_momentum_accel:
    accel = compute_momentum_acceleration(
        M,
        lookback=cfg.accel_lookback,
        smoothing=cfg.accel_smoothing
    )
else:
    accel = pd.DataFrame(0.0, index=M.index, columns=M.columns)
```

**4. Modify final score combination:**

Change from:
```python
SCORE = (rel * CONF) - risk_pen
```

To:
```python
SCORE = (rel * CONF) - risk_pen + cfg.accel_weight * accel
```

**Toy example:**

Consider two models with identical current momentum and delta:

```
Model A momentum history: [0.40, 0.45, 0.52, 0.61, 0.72]
- Momentum diffs: [+0.05, +0.07, +0.09, +0.11]
- Acceleration (mean of recent diffs): ~+0.09 (accelerating upward)

Model B momentum history: [0.55, 0.62, 0.67, 0.70, 0.72]
- Momentum diffs: [+0.07, +0.05, +0.03, +0.02]
- Acceleration (mean of recent diffs): ~+0.03 (decelerating)

Both end at M=0.72, but Model A has positive acceleration momentum,
Model B is losing steam. Model A gets a boost, Model B gets a smaller boost.
```

**5. Why this differs from existing delta term:**

- **Delta (`D_{i,t}`)**: Single-period rank change `Q_t - Q_{t-1}`. Captures instantaneous velocity but is noisy.
- **Acceleration**: Change in momentum over multiple periods. Captures whether the trend itself is strengthening or weakening.

The delta term tells you "did rank improve this period?" while acceleration tells you "is the rate of improvement itself improving?" They're complementary: a model could have a small positive delta but strong positive acceleration (early in a breakout), or large positive delta but negative acceleration (momentum exhaustion).

IDEA: Add a "Momentum Continuation" signal that amplifies delta contribution only when the current rank change continues the direction of recent rank changes (trend confirmation)**

RATIONALE: The current delta adjustment `D = Q_t - Q_{t-1}` rewards any positive rank change equally, whether it's a continuation of an existing uptrend or a sudden reversal from decline. However, rank changes that *continue* an existing trend are more likely to persist than isolated one-period jumps. If a model has been climbing ranks over the past few periods and continues climbing, that's a stronger signal than a model that was falling and suddenly spikes up (which is more likely mean-reversion or noise). By computing a "momentum continuation" factor — whether the current delta has the same sign as the rolling delta trend — and using it to scale the delta contribution, we can reduce noise from false reversals and amplify genuine trend continuation. This is training-free, uses only lookback data, and addresses the OOS degradation by filtering out unreliable delta signals.

REQUIRED_CHANGES: 1. **In `config.py`**, add new parameters:
   - `momentum_continuation_weight: float = 0.08` — weight for the momentum continuation signal
   - `momentum_continuation_lookback: int = 4` — lookback for computing the trend direction of delta

2. **In `scoring.py`, within `compute_scores_for_ticker_v2` (after computing delta D):**
   - Compute a rolling signed trend of D over the lookback window: for each model at time t, compute `trend_sign_t = sign(mean(D_{t-L:t-1}))` (the average direction of recent rank changes, excluding the current period to keep it causal)
   - Compute the current delta sign: `current_sign_t = sign(D_t)`
   - Create a continuation indicator: `continuation_t = 1 if current_sign_t == trend_sign_t and trend_sign_t != 0, else 0`
   - Normalize across models via percentile ranking to get `continuation_norm`
   
3. **Modify the base_forecast computation (around line ~494):**
   - Instead of `base_forecast = M + cfg.delta_weight * D`, modify to:
   - `continuation_boost = cfg.momentum_continuation_weight * continuation_norm * np.abs(D)`
   - `base_forecast = M + cfg.delta_weight * D + continuation_boost`
   - This adds extra weight to delta when it confirms the recent trend

4. **Implementation specifics:**
   - Use `pd.DataFrame(D.T).rolling(window=cfg.momentum_continuation_lookback, min_periods=2).mean()` to compute rolling mean of D
   - The continuation signal is computed per model per period
   - When the continuation condition is met (same sign as trend), the model gets a boost proportional to the magnitude of its current delta
   - Handle edge cases where trend_sign is 0 (no clear trend) by defaulting to no boost (continuation = 0)

IDEA: Add a "Peak Distance Discount" signal that penalizes models whose current rank is significantly below their recent peak rank over a lookback window, identifying models in decline even when their absolute rank remains high

RATIONALE: The current scoring system rewards models with high momentum (smoothed recent ranks) and various stability metrics, but doesn't explicitly detect models that are *declining from their recent best*. A model at rank percentile 0.70 that peaked at 0.95 four periods ago is showing a 25-point decline, suggesting deterioration or mean reversion, whereas a model at 0.70 that peaked at 0.72 is stable. The existing signals miss this because:
- Momentum M is a weighted average of recent ranks, not a peak comparison
- Delta D captures single-period change, not cumulative decline from peak
- Durability measures consecutive tenure above median, not distance from peak
- Velocity confirmation checks directional consistency, not magnitude of decline from best

The "baseline positive, meta negative" problem (~15% of periods) likely includes models that were selected based on decent momentum/rank but were actually in a declining trend from their recent highs. By computing `peak_distance = max(Q over lookback) - Q_current` and penalizing models with large peak distances, we can filter out these "past their prime" candidates while rewarding models that are at or near their peak performance. This follows the successful pattern of simple, threshold-based signals using only rank data.

REQUIRED_CHANGES: 1. **In `config.py`**, add new parameters:
   - `peak_distance_weight: float = -0.06` — **negative** weight because high peak distance is bad (penalizes models far from peak)
   - `peak_distance_lookback: int = 6` — window over which to find the peak rank

2. **In `scoring.py`, within `compute_scores_for_ticker_v2` (after computing Q percentile ranks, before base_forecast assembly):**
   - Compute rolling maximum rank over lookback:
     ```python
     # Q shape: (n_models, T)
     L = cfg.peak_distance_lookback
     Q_df = pd.DataFrame(Q.T)  # shape (T, n_models)
     rolling_peak = Q_df.rolling(window=L, min_periods=1).max().to_numpy().T  # shape (n_models, T)
     
     # Peak distance: how far current Q is below the recent peak
     peak_distance_raw = rolling_peak - Q  # shape (n_models, T), always >= 0
     
     # Normalize via percentile rank across models per period (higher = worse)
     peak_distance_norm = percentile_ranks_across_models_v2(peak_distance_raw, axis=0)
     ```

3. **Add to base_forecast computation (around line ~597-613 in scoring.py):**
   ```python
   if cfg.peak_distance_weight != 0 and peak_distance_norm is not None:
       base_forecast = base_forecast + cfg.peak_distance_weight * peak_distance_norm
   ```
   Note: Since the weight is negative, models with high normalized peak distance (far from peak) get *reduced* base_forecast, while models at/near their peak get less penalty.

4. **Implementation specifics:**
   - Use pandas rolling max for efficient computation
   - Peak distance is always non-negative (current Q cannot exceed recent max Q)
   - A model at its recent peak has peak_distance_raw = 0, leading to low percentile rank (reward)
   - A model 0.30 below its recent peak has high peak_distance_raw, leading to high percentile rank (penalty)
   - Handle NaN in Q by propagating NaN (pd.rolling handles this automatically)
   - The signal is causal: rolling_peak at t uses Q from t-L+1 to t, which is all available at decision time
   - The negative weight ensures this acts as a *penalty* for decline, not a reward

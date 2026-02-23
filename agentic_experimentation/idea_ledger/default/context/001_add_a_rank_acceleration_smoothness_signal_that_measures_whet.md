IDEA: Add a "Rank Acceleration Smoothness" signal that measures whether a model's recent rank changes (deltas) exhibit consistent directionality across a 2-step lag structure—specifically rewarding models where `sign(D_{t}) == sign(D_{t-2})` (current delta matches 2-period-ago delta), which indicates sustained trend rather than mean-reverting oscillation.

RATIONALE: The current velocity confirmation signal measures whether recent deltas share the same sign as the current delta, but it treats all lags equally within the lookback window. However, momentum literature suggests that the 1-period-ago delta is more likely to show noise/reversal (immediate mean reversion) while the 2-period-ago delta better reflects the underlying trend. Models exhibiting `sign(D_t) == sign(D_{t-2})` but `sign(D_t) != sign(D_{t-1})` are still on a sustained trajectory—they experienced a one-period pause/dip but resumed the prior trend direction. This "skip-lag" confirmation pattern captures trend persistence through short-term noise, which the current signals miss. The 15% "baseline positive, meta negative" problem likely includes models that were rising, had a one-period dip (triggering lower velocity confirmation scores), then resumed rising—but by then they'd been deselected. This signal is distinct from:
- Velocity confirmation (which weights all recent deltas equally)  
- Momentum/delta (which capture level and single-period change)
- Durability (which measures threshold tenure, not delta patterns)

It follows the successful pattern of simple, threshold-based logic using rank deltas.

REQUIRED_CHANGES: 1. **In `config.py`**, add new parameters:
   - `rank_acceleration_smoothness_weight: float = 0.06` — weight for the smoothness signal (similar to velocity confirmation)
   - `acceleration_skip_lag: int = 2` — the lag to compare against (default 2 means compare D_t with D_{t-2})
   - `acceleration_threshold: float = 0.015` — minimum delta magnitude to count as "moving"

2. **In `scoring.py`, within `compute_scores_for_ticker_v2` (after computing delta D):**
   - Compute acceleration smoothness score:
     ```python
     # D already computed: D[:, t] = Q[:, t] - Q[:, t-1]
     L = cfg.acceleration_skip_lag
     smoothness_raw = np.full((n_models, T), 0.5, dtype=float)  # neutral default
     
     for t in range(L, T):
         d_t = D[:, t]
         d_lag = D[:, t - L]  # delta from L periods ago
         
         # Both must be significant moves
         current_sig = np.abs(d_t) > cfg.acceleration_threshold
         lag_sig = np.abs(d_lag) > cfg.acceleration_threshold
         both_sig = current_sig & lag_sig
         
         # Smoothness = 1 if signs match (trend persisting), 0 if signs differ (reversal)
         sign_match = np.sign(d_t) == np.sign(d_lag)
         
         smoothness_raw[both_sig & sign_match, t] = 1.0
         smoothness_raw[both_sig & ~sign_match, t] = 0.0
         # Models without significant moves in either period stay at 0.5 (neutral)
     ```
   - Normalize via percentile ranking: `smoothness_norm = percentile_ranks_across_models_v2(smoothness_raw, axis=0)`

3. **Add to base_forecast computation (after existing feature additions):**
   ```python
   if cfg.rank_acceleration_smoothness_weight != 0 and smoothness_norm is not None:
       base_forecast = base_forecast + cfg.rank_acceleration_smoothness_weight * smoothness_norm
   ```

4. **Implementation specifics:**
   - The 2-period skip-lag captures whether the "fundamental" trend direction is maintained despite short-term noise
   - This is causal: only uses deltas from t and t-2 (both computed from prior returns)
   - Handle NaN by defaulting to 0.5 (neutral) when insufficient data
   - The signal rewards models whose trend direction at t matches direction at t-2, even if t-1 was flat/reversed (filtering noise)
   - Can be vectorized efficiently since it only needs D matrix slices at t and t-L

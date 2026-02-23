IDEA: Add a "Rank Velocity Confirmation" signal that boosts models where the direction of recent rank change (velocity sign) matches the direction of slightly older rank change (confirming trend), and penalizes models where velocity has reversed direction (early warning of mean reversion or regime shift)

RATIONALE: The current model has momentum M (smoothed rank), delta D (one-period change), and durability (time above median), but none specifically measure whether rank velocity is *consistent in direction* across multiple time scales. A model where `D_{t-1} > 0` AND `D_t > 0` (consecutive positive deltas) shows confirmed upward trajectory, while a model where `D_{t-1} > 0` but `D_t < 0` (velocity reversal) is showing early signs of mean reversion even if M and Q are still high. This "velocity confirmation" pattern is distinct from:
- Delta adjustment (which only uses the current single-period change)
- Momentum deceleration warning (proposed, which measures second derivative magnitude)
- Durability (which measures threshold tenure, not velocity direction)

The signal captures *directional consistency*—whether the rank is moving in the same direction it was moving last period—which is a simple but powerful filter for differentiating genuine trends from noise-driven spikes. This addresses the ~11% baseline-positive-meta-negative problem by catching models that are reversing direction before their absolute rank/score fully reflects the deterioration. It follows the successful pattern of simple boolean/threshold logic (like breakout and durability).

REQUIRED_CHANGES: 1. **In `config.py`**, add new parameters:
   - `velocity_confirmation_weight: float = 0.06` — weight for the velocity confirmation signal
   - `velocity_lookback: int = 3` — number of periods over which to measure velocity consistency
   - `velocity_threshold: float = 0.02` — minimum delta magnitude to count as "moving" (filters noise)

2. **In `scoring.py`, within `compute_scores_for_ticker_v2` (after computing delta D):**
   - Compute velocity confirmation score:
     ```python
     # D is already computed: D[:, t] = Q[:, t] - Q[:, t-1]
     # Velocity confirmation: count how many of the last L deltas have the same sign as D[t]
     
     L = cfg.velocity_lookback
     sign_current = np.sign(D)  # shape (n_models, T)
     velocity_confirmation_raw = np.zeros((n_models, T), dtype=float)
     
     for t in range(L, T):
         d_t = D[:, t]
         d_window = D[:, t-L+1:t+1]  # last L deltas including current
         
         # Only consider "significant" moves (above threshold)
         is_significant = np.abs(d_window) > cfg.velocity_threshold
         
         # For each model, count how many of the significant deltas match the sign of d_t
         sign_match = (np.sign(d_window) == np.sign(d_t)[:, None]) & is_significant
         match_count = sign_match.sum(axis=1)
         total_significant = is_significant.sum(axis=1)
         
         # Confirmation ratio = fraction of significant moves in same direction
         confirmation_ratio = np.divide(
             match_count, 
             total_significant,
             out=np.full_like(match_count, 0.5, dtype=float),
             where=total_significant > 0
         )
         velocity_confirmation_raw[:, t] = confirmation_ratio
     ```
   - Normalize via percentile ranking: `velocity_confirmation_norm = percentile_ranks_across_models_v2(velocity_confirmation_raw, axis=0)`

3. **Add to base_forecast computation:**
   ```python
   if cfg.velocity_confirmation_weight != 0 and velocity_confirmation_norm is not None:
       base_forecast = base_forecast + cfg.velocity_confirmation_weight * velocity_confirmation_norm
   ```

4. **Implementation specifics:**
   - The signal rewards models with consistent velocity direction (e.g., 3/3 recent deltas are positive = high confirmation)
   - The signal penalizes models with velocity reversal (e.g., was rising, now falling = low confirmation relative to peers)
   - The threshold parameter filters out tiny wiggles that aren't meaningful velocity
   - The signal is normalized across models per period, so it's relative ranking
   - Handle NaN by defaulting to 0.5 (neutral) when insufficient data
   - This is causal: only uses data available through period t

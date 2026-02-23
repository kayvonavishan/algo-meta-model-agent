IDEA: Add a "Rank Acceleration Consistency" signal that rewards models whose recent rank improvement rate (velocity) is positive across multiple sub-windows, rather than just measuring total momentum or single-period deltas

RATIONALE: The current delta_weight captures single-period rank changes, and momentum captures smoothed rank levels, but neither explicitly rewards *consistent directional movement* in ranks. A model that improves its rank in 4 out of 5 recent periods by small amounts is more reliable than one that jumped 30 percentile points in one period (which may be noise). The successful signals (breakout, durability) both use threshold-counting logic. This signal applies similar counting logic to rank velocity: count the number of recent periods where the model's rank increased (ΔQ > 0). Models with consistently positive rank velocity (4+ out of 5 periods) get higher scores than models with erratic rank movements. This addresses the low `rolling_hit_rate_min` problem by filtering out models that occasionally spike but have inconsistent improvement trajectories. Unlike the failed "consistency-adjusted momentum" (which used autocorrelation — complex and noisy), this uses simple counting (did rank go up? yes/no per period) which aligns with the successful pattern of threshold-based signals.

REQUIRED_CHANGES: 1. **In `config.py`**, add new parameters:
   - `rank_velocity_consistency_weight: float = 0.08` — weight for the rank velocity consistency signal (similar to breakout_weight and durability_weight)
   - `rank_velocity_lookback: int = 5` — how many periods of rank changes to examine
   - `rank_velocity_threshold: float = 0.6` — fraction of periods that must show positive rank change to get full credit (e.g., 0.6 means 3/5 periods must have ΔQ > 0)

2. **In `scoring.py`, within `compute_scores_for_ticker_v2` (after computing D = rank deltas):**
   - For each model and time t, count the number of periods in `[t-L+1, t]` where `D[:, k] > 0` (rank increased):
     - `positive_deltas = np.sum((D[:, max(0, t-L+1):t+1] > 0), axis=1)` for each t
     - `fraction_positive = positive_deltas / min(L, t+1)` — what fraction of recent periods showed rank improvement
   - Create `velocity_consistency_raw`:
     - If `fraction_positive >= cfg.rank_velocity_threshold`: score = fraction_positive (reward consistency)
     - If `fraction_positive < cfg.rank_velocity_threshold`: score = fraction_positive * 0.5 (dampen inconsistent models)
   - Normalize via percentile ranking across models per period: `velocity_norm = percentile_ranks_across_models_v2(velocity_consistency_raw, axis=0)`

3. **Add to base_forecast computation (after existing feature additions, around line ~577):**
   ```python
   if cfg.rank_velocity_consistency_weight != 0 and velocity_norm is not None:
       base_forecast = base_forecast + cfg.rank_velocity_consistency_weight * velocity_norm
   ```

4. **Implementation specifics:**
   - Vectorized: compute `(D > 0).astype(float)` then use `pd.DataFrame(...).rolling(window=L, min_periods=1).mean()` to get fraction of positive deltas
   - The threshold check creates a soft discontinuity: models consistently improving get proportionally more credit, oscillating models get dampened
   - Handle NaN in D by excluding from count (already handled by rolling mean's default NaN handling)
   - This is causal: uses only D values computed through t-1 (since D itself is computed from Q differences)

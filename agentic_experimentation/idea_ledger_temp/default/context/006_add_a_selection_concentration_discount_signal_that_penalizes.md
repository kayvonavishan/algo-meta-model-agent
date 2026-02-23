IDEA: Add a "Selection Concentration Discount" signal that penalizes models which have been continuously selected for too many consecutive periods, forcing some natural rotation and reducing the risk of extended drawdowns from stale selections**

RATIONALE: The current model shows a max drawdown duration of 22 periods (vs 8 for the all_models baseline), which is the single worst degradation metric. This suggests the selection process is "sticky" - once a model enters the top-N, it tends to stay selected even as its edge decays. The confidence signal rewards stability, and the momentum signal rewards recent high ranks, but neither explicitly addresses "time in portfolio" - how long a model has been continuously selected. In equity factor investing, "crowding" is a known risk: strategies that have been in favor too long tend to mean-revert once the crowd becomes too concentrated. Similarly, a model that has been top-ranked for 8+ consecutive periods may be "priced in" (all favorable conditions already reflected), making it vulnerable to sudden regime change. By computing how many consecutive periods a model has been in the top-tier (Q > 0.7, say) and applying a gentle discount to models with very long tenures, we encourage healthy rotation and reduce the probability of holding onto winners too long. This differs from rank durability (which is additive and rewards tenure) because this is a *subtractive* penalty that specifically targets *excessive* concentration. The signal is training-free, causal, and directly addresses the extended drawdown duration problem.

REQUIRED_CHANGES: 1. **In `config.py`**, add new parameters:
   - `concentration_discount_weight: float = 0.05` — weight for the concentration penalty (moderate, not too aggressive)
   - `concentration_lookback: int = 12` — how many periods to look back for tenure calculation
   - `concentration_tenure_threshold: int = 6` — consecutive periods above high rank before penalty activates
   - `concentration_rank_threshold: float = 0.7` — rank percentile defining "selected/top" status

2. **In `scoring.py`, within `compute_scores_for_ticker_v2` (after computing Q percentile ranks):**
   - For each model and time t, count consecutive periods ending at t-1 where Q > concentration_rank_threshold:
     - Similar to durability but we count tenure in TOP THIRD, not just above median
   - Apply exponential discount for tenure above threshold:
     - `discount_raw = max(0, tenure - concentration_tenure_threshold)` — no penalty until threshold reached
     - `discount_factor = 1.0 - concentration_discount_weight * log1p(discount_raw)` — logarithmic decay
     - Or simpler: `penalty = cfg.concentration_discount_weight * max(0, (tenure - threshold) / lookback)` — linear penalty scaled to [0, weight]
   - Normalize via percentile ranking (inverted): `concentration_penalty_norm = 1.0 - percentile_ranks_across_models_v2(discount_raw, axis=0)`
   - Models with short tenure in top tier get high normalized scores; models with extended tenure get discounted

3. **Add to base_forecast computation (after existing feature additions, around line ~548):**
   ```python
   if cfg.concentration_discount_weight != 0 and concentration_penalty_norm is not None:
       base_forecast = base_forecast + cfg.concentration_discount_weight * concentration_penalty_norm
   ```
   (Note: using `+` with `concentration_penalty_norm` inverted so high tenure = low score contribution)

4. **Implementation specifics:**
   - Compute `is_high_rank = Q > cfg.concentration_rank_threshold` (boolean)
   - For each column t, walk backwards to count consecutive Trues ending at t-1 (causal)
   - Cap at lookback window to avoid infinite tenure effects
   - This is NOT the same as penalizing all tenure (durability) - it specifically targets *extended* tenure above a high bar
   - The penalty activates only after `concentration_tenure_threshold` periods, so short-term winners are not penalized

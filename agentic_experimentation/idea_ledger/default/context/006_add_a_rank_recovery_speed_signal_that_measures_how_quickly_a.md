IDEA: Add a "Rank Recovery Speed" signal that measures how quickly a model returns to above-median rank (Q > 0.5) after temporarily dropping below the median, rewarding models that demonstrate quick rebound behavior rather than prolonged underperformance.

RATIONALE: The current scoring has durability (consecutive tenure above median) and breakout (crossing into top tier), but neither captures how a model *recovers* after falling below median. A model that dipped below median 2 periods ago but immediately rebounded demonstrates resilience and mean-reversion behavior, whereas a model that stayed below median for 6 periods before recovering is sluggish and may reflect structural issues. The max underperformance streak of 7 periods and rolling hit rate minimum of 16.7% suggest the meta model sometimes selects models that experience extended poor performance. By measuring time-since-last-drop (for models currently above median) or time-below-threshold (for models currently below), we can distinguish quick-recovery models from slow-recovery models. This follows the successful pattern of threshold-based counting (like breakout and durability) but captures the complementary dimension of recovery dynamics. Models that are currently above median AND recently recovered quickly get higher scores; models that have been below median for many periods get penalized. This is training-free, causal, and distinct from existing signals.

REQUIRED_CHANGES: 1. **In `config.py`**, add new parameters:
   - `rank_recovery_speed_weight: float = 0.08` — weight for the rank recovery speed signal (similar to breakout_weight and durability_weight)
   - `rank_recovery_lookback: int = 6` — how many periods to look back for recovery assessment
   - `rank_recovery_penalty_scale: float = 0.5` — how much to penalize slow recovery (longer time below median)

2. **In `scoring.py`, within `compute_scores_for_ticker_v2` (after computing Q percentile ranks):**
   - For each model and time t, compute the recovery profile:
     - `is_above = Q[:, t] > 0.5` (boolean: currently above median)
     - If above median: look back to find most recent period when model was below median. Recovery score = 1.0 - (periods_since_dip / lookback), clipped to [0, 1]. Models that never dipped get score 1.0. Models that just recovered get score close to 1.0. Models that recovered long ago (stable) get score 1.0.
     - If below median: look back to count consecutive periods below median. Score = cfg.rank_recovery_penalty_scale * (1.0 - periods_below_median / lookback), clipped to [0, 0.5]. Longer streaks below median get lower scores.
   - This produces `recovery_raw` matrix of shape (n_models, T)
   - Normalize via percentile ranking across models per period: `recovery_norm = percentile_ranks_across_models_v2(recovery_raw, axis=0)`

3. **Add to base_forecast computation (after existing feature additions):**
   ```python
   if cfg.rank_recovery_speed_weight != 0 and recovery_norm is not None:
       base_forecast = base_forecast + cfg.rank_recovery_speed_weight * recovery_norm
   ```

4. **Implementation specifics:**
   - Vectorized approach: for each column t, compute two masks: `is_above` and `is_below`. For `is_above` models, find `argmax` of last time they were below (or default to -lookback if never). For `is_below` models, count consecutive streak.
   - The signal is causal: uses only Q values through period t
   - Handle NaN in Q by treating as "below median" (conservative)
   - This signal distinguishes: (a) quick recoverers who recently bounced back, (b) stable performers who haven't needed to recover, (c) slow recoverers still below median, (d) prolonged underperformers stuck below median

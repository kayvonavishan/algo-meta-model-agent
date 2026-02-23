IDEA: Add a "Rank Durability" signal that measures how long a model has maintained an above-median rank (Q > 0.5) without dropping below the median, giving higher scores to models with longer unbroken "tenure" in the upper half

RATIONALE: The current scoring rewards momentum (recent rank trend) and confidence (rank stability/volatility), but neither explicitly captures "duration in top half without interruption." A model that has stayed above median for 8 consecutive periods is demonstrating sustained competence, not just a lucky streak followed by a drop. This differs from rank_persistence (which measures fraction of periods above 0.5 over a lookback) because it specifically tracks **continuous unbroken tenure** — a model that dipped below median 3 periods ago and recovered would have tenure=2, whereas rank_persistence might show 75% if it was above median 6/8 periods. Durability is a stronger predictor of future persistence because it filters out "oscillators" that frequently cross the median threshold. This is training-free, uses only causal lookback data, and captures a fundamentally different signal dimension than existing features. Unlike the rejected consistency-adjusted momentum (which tried autocorrelation of rank changes), this is a simple threshold-based counting signal similar in spirit to the successful breakout signal.

REQUIRED_CHANGES: 1. **In `config.py`**, add new parameters:
   - `rank_durability_weight: float = 0.08` — weight for the rank durability signal (similar to breakout_weight)
   - `rank_durability_cap: int = 12` — maximum tenure periods to count (caps the signal to avoid over-rewarding very long tenures)

2. **In `scoring.py`, within `compute_scores_for_ticker_v2` (after computing Q percentile ranks):**
   - For each model and time t, count consecutive periods ending at t-1 (causal) where Q > 0.5:
     - Initialize `tenure = 0`
     - Walk backwards from t-1: while Q[:, idx] > 0.5, increment tenure (cap at `rank_durability_cap`)
     - If Q[:, t-1] <= 0.5, tenure = 0
   - This produces `durability_raw` matrix of shape (n_models, T)
   - Normalize via percentile ranking across models per period: `durability_norm = percentile_ranks_across_models_v2(durability_raw, axis=0)`

3. **Add to base_forecast computation (around line ~548 in scoring.py, after existing feature additions):**
   ```python
   if cfg.rank_durability_weight != 0 and durability_norm is not None:
       base_forecast = base_forecast + cfg.rank_durability_weight * durability_norm
   ```

4. **Implementation specifics:**
   - Vectorized approach: create a boolean mask `is_above = Q > 0.5`, then for each column t, iterate backwards to count consecutive Trues ending at t-1
   - More efficient: compute `(Q > 0.5).astype(int)`, then use cumulative sum tricks with reset-on-zero logic
   - Handle NaN in Q by treating as "not above median" (tenure resets)
   - The causal shift ensures we use tenure computed through t-1 for scoring at t

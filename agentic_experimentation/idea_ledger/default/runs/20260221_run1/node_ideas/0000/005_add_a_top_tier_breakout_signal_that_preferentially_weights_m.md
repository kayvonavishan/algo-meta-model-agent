IDEA: Add a "Top-Tier Breakout" signal that preferentially weights models crossing into the upper rank quintile (Q > 0.8) from below, while dampening scores for models that frequently oscillate around the breakout threshold

RATIONALE: The current scoring rewards any positive momentum equally across the rank spectrum. However, a model moving from the 70th to 90th percentile (entering "top-tier" territory) is fundamentally different from a model moving from 10th to 30th percentile—the former is becoming a genuine contender while the latter may just be recovering from extreme underperformance. Furthermore, models that repeatedly cross back and forth around the 80th percentile threshold ("false breakouts") are noisy and unreliable for forward selection. By tracking whether a model has recently crossed into the top quintile AND stayed there (vs. oscillating), we can identify genuine breakout candidates versus lucky one-period spikes. This directly addresses the OOS degradation problem: many false positives likely come from models that briefly spike into high ranks but quickly revert. The signal is training-free, uses only lookback data, and captures a different aspect than the existing signals (which focus on consistency/autocorrelation, rank-level scaling, cross-sectional compression, or trend continuation—but none specifically target threshold-crossing behavior).

REQUIRED_CHANGES: 1. **In `config.py`**, add new parameters:
   - `breakout_weight: float = 0.08` — weight for the top-tier breakout signal (similar magnitude to other feature weights)
   - `breakout_threshold: float = 0.80` — rank percentile threshold defining "top tier"
   - `breakout_lookback: int = 4` — window for measuring breakout stability (how many periods must model stay above threshold)
   - `breakout_oscillation_penalty: float = 0.5` — penalty factor for models that cross the threshold multiple times (oscillators)

2. **In `scoring.py`, within `compute_scores_for_ticker_v2` (after computing Q percentile ranks):**
   - For each model and time t, compute:
     - `is_above_threshold_t = Q[:, t] > cfg.breakout_threshold` (boolean mask: currently in top tier)
     - `was_below_recently = any(Q[:, t-L:t-1] <= cfg.breakout_threshold)` over lookback window (was recently NOT in top tier)
     - `threshold_crossings = count of times Q crossed cfg.breakout_threshold in either direction over lookback`
   - Create `breakout_score_raw`:
     - If `is_above_threshold AND was_below_recently AND threshold_crossings <= 2`: full breakout score = 1.0 (genuine breakout, stayed above)
     - If `is_above_threshold AND threshold_crossings > 2`: penalized score = 1.0 - cfg.breakout_oscillation_penalty (oscillator)
     - If `NOT is_above_threshold`: score = 0.0 (not in top tier)
   - Normalize via percentile ranking across models per period to get `breakout_norm`

3. **Add to base_forecast computation (around line ~494-504 in scoring.py):**
   - After existing feature additions, add:
     ```python
     if cfg.breakout_weight != 0 and breakout_norm is not None:
         base_forecast = base_forecast + cfg.breakout_weight * breakout_norm
     ```

4. **Implementation specifics:**
   - To count threshold crossings: `crossings = np.sum(np.abs(np.diff((Q[:, t-L:t] > threshold).astype(int), axis=1)), axis=1)`
   - The "was_below_recently" check ensures we're rewarding genuine breakouts, not models that have been above threshold for a long time
   - This signal is computed per model per period, similar to existing feature signals
   - Handle NaN edge cases by defaulting to 0.0 (neutral) when insufficient data exists

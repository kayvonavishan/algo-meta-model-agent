IDEA: Add a "Score Spread Gating" signal that amplifies scores for models when the cross-sectional score spread (distance between top and median scores) is wide, and dampens them when the spread is narrow — essentially increasing selection confidence when differentiation is high.

RATIONALE: The current scoring produces a score per model per period, but doesn't account for the *quality of differentiation* in each period. In periods where top models have very similar scores (narrow spread), the selection is effectively choosing among near-equals — this introduces noise because small score differences may reflect randomness rather than true signal. In periods with wide score spread, the top models are clearly differentiated from median performers, making selection more reliable. By computing the ratio of a model's score to the cross-sectional IQR (or range from median to top), we can identify "high conviction" periods where differentiation is strong. Models in the top ranks during high-spread periods get a multiplicative boost; models selected during narrow-spread periods get dampened. This follows the successful pattern of simple threshold-based signals (like breakout and durability) and addresses a root cause of low rolling outperformance in certain windows: those may be periods where score spread was low and selection was essentially random. Unlike the failed regime-adaptive risk scaling (which modified penalties based on market regime), this modifies based on the *quality of the scoring signal itself*, which is more directly actionable. It's training-free, causal, and targets the inconsistency in selection quality across periods.

REQUIRED_CHANGES: 1. **In `config.py`**, add new parameters:
   - `score_spread_boost_weight: float = 0.10` — weight for score spread gating (how much to amplify scores in high-spread periods)
   - `score_spread_lookback: int = 4` — lookback for computing rolling median score spread
   - `score_spread_z_threshold: float = 0.5` — z-score threshold above which spread is considered "high"

2. **In `scoring.py`, within `compute_scores_for_ticker_v2` (after computing the base SCORE matrix but before the final causal shift):**
   - For each period t, compute the score spread:
     - `score_iqr_t = np.nanpercentile(SCORE[:, t], 75) - np.nanpercentile(SCORE[:, t], 25)` (IQR of scores)
     - Or simpler: `score_range_t = np.nanmax(SCORE[:, t]) - np.nanmedian(SCORE[:, t])` (top-to-median gap)
   - Compute a rolling z-score of `score_spread`:
     - `spread_z = (score_spread_t - rolling_mean(score_spread)) / (rolling_std(score_spread) + eps)`
   - Create a multiplicative gating factor:
     - `gate_t = 1.0 + cfg.score_spread_boost_weight * max(0, spread_z - cfg.score_spread_z_threshold)`
     - When spread z-score exceeds threshold, boost all scores proportionally for that period
     - When spread is below threshold, `gate_t = 1.0` (no change)
   - Apply: `SCORE[:, t] = SCORE[:, t] * gate_t`

3. **Implementation specifics:**
   - Compute `score_spread` as a 1D array of shape (T,) once per ticker
   - Use `pd.Series(score_spread).rolling(window=cfg.score_spread_lookback, min_periods=2)` for rolling stats
   - The multiplicative gating preserves relative ordering within each period while amplifying the signal when selection quality is high
   - Handle NaN in score spread by defaulting gate to 1.0
   - This is causal: uses only scores computed from data through period t

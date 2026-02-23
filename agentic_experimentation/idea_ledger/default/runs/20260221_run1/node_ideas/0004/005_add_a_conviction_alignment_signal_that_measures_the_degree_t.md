IDEA: Add a "Conviction Alignment" signal that measures the degree to which a model's high scores coincide with periods of strong selection differentiation (high score spread periods), rewarding models that rise to the top specifically when selection quality is high rather than during noisy low-differentiation periods.**

RATIONALE: The current score spread gating signal boosts ALL scores proportionally during high-spread periods, but it doesn't distinguish between models that *only* look good during low-spread (noisy) periods versus models that *also* look good during high-spread (high-conviction) periods. A model that consistently ranks highly during high-spread periods demonstrates genuine ability to differentiate from peers when conditions favor differentiation—this is a stronger signal than a model that performs well only when all scores are bunched together and the selection is essentially random among near-equals. The 15% baseline-positive-meta-negative problem may partly stem from selecting models that happened to score well during low-conviction periods but don't hold up during periods with clearer signal. By tracking each model's historical "alignment" with high-conviction periods—specifically measuring what fraction of its recent high-rank periods occurred when score spread was above average—we can identify models with durable, differentiable performance versus lucky beneficiaries of noise. This follows the successful pattern of simple threshold/counting logic (like durability and breakout) and builds on the score spread concept but applies it at the *model* level rather than the *period* level.

REQUIRED_CHANGES: 1. **In `config.py`**, add new parameters:
   - `conviction_alignment_weight: float = 0.07` — weight for the conviction alignment signal
   - `conviction_alignment_lookback: int = 8` — lookback window for measuring alignment
   - `conviction_high_spread_threshold: float = 0.0` — z-score threshold above which spread is considered "high" (0 = above average)
   - `conviction_high_rank_threshold: float = 0.60` — model Q threshold to count as "high performing" for this measure

2. **In `scoring.py`, within `compute_scores_for_ticker_v2` (after computing SCORE and the score_spread series):**
   - For each period t, compute whether it's a "high-conviction" period:
     - `spread_z_t = (score_spread_t - rolling_mean(score_spread)) / (rolling_std(score_spread) + eps)`
     - `is_high_conviction_t = spread_z_t > cfg.conviction_high_spread_threshold` (boolean)
   - For each model at time t, over the lookback window, compute:
     - `n_high_rank_periods = count of periods where model's Q > cfg.conviction_high_rank_threshold`
     - `n_high_rank_AND_high_conviction = count of periods where Q > threshold AND is_high_conviction`
     - `alignment_ratio = n_high_rank_AND_high_conviction / (n_high_rank_periods + eps)`
   - This ratio measures: "of the times this model performed well, what fraction occurred during high-conviction periods?"
   - Normalize via percentile ranking: `conviction_alignment_norm = percentile_ranks_across_models_v2(alignment_ratio, axis=0)`

3. **Add to base_forecast computation (after existing feature additions):**
   ```python
   if cfg.conviction_alignment_weight != 0 and conviction_alignment_norm is not None:
       base_forecast = base_forecast + cfg.conviction_alignment_weight * conviction_alignment_norm
   ```

4. **Implementation specifics:**
   - Reuse the `score_spread` computation already done for score_spread_boost_weight
   - The spread z-score can be computed once per period and reused
   - For each model, track (over lookback) the count of high-rank periods and the count that overlapped with high-conviction
   - Models with alignment_ratio close to 1.0 consistently rank well when selection quality is high (strong signal)
   - Models with alignment_ratio close to 0.0 only rank well during noisy/bunched periods (weak signal)
   - Handle NaN by defaulting to 0.5 (neutral) when insufficient lookback data
   - This is causal: uses Q and score_spread through period t-1 for scoring at t (via final causal shift)

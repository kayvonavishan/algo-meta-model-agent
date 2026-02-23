IDEA: Add a "Return-to-Rank Coherence" signal that measures whether a model's rank movements are proportionally aligned with its return magnitude, penalizing models whose high ranks are achieved primarily through small return differentials (unstable leaders) while rewarding models whose high ranks reflect genuinely large return advantages.

RATIONALE: The current percentile ranking system treats a model at the 95th percentile identically whether it beat the 50th percentile model by 5% or 0.1%. However, a model whose top rank is achieved by a razor-thin margin is much more likely to flip positions next period than one with a comfortable lead. The existing signals (momentum, durability, breakout, score spread) don't directly capture this "rank stability through return advantage" dimension. When returns are tightly clustered (even if ranks span [0,1] by definition), the top-ranked models are essentially interchangeable—selecting based on rank ordering adds noise. By computing the ratio of each model's return deviation from median to the cross-sectional return spread, we can identify models whose high ranks reflect genuine return separation versus statistical noise. This is distinct from:
- Score spread gating (which applies a period-level multiplier, not model-level)
- Confidence (which measures rank stability over time, not rank-return coherence within a period)
- Rank dispersion dampening (proposed, which also operates at period level)

This signal addresses the 15% "baseline-positive-meta-negative" problem by deprioritizing models that achieved top ranks through near-ties.

REQUIRED_CHANGES: 1. **In `config.py`**, add new parameters:
   - `return_rank_coherence_weight: float = 0.08` — weight for the coherence signal (similar to other feature weights)
   - `coherence_threshold: float = 0.25` — minimum return spread (as fraction of cross-sectional std) below which coherence penalty applies

2. **In `scoring.py`, within `compute_scores_for_ticker_v2` (after computing Q percentile ranks and before base_forecast assembly):**
   - For each period t, compute:
     - `cs_median_t = np.nanmedian(R[:, t])` — cross-sectional median return
     - `cs_std_t = np.nanstd(R[:, t])` — cross-sectional return spread
     - `return_deviation_t = R[:, t] - cs_median_t` — each model's return vs median
     - `normalized_deviation_t = return_deviation_t / (cs_std_t + 1e-8)` — z-score of return deviation
   - Compute coherence score:
     - For models with `Q[:, t] > 0.5` (above median rank), coherence = how much their normalized_deviation supports their rank
     - `coherence_raw[:, t] = where(Q[:, t] > 0.5, normalized_deviation_t, -normalized_deviation_t)` 
     - This rewards above-median models with large positive return deviations, and below-median models with appropriately negative deviations
   - Normalize via percentile ranking: `coherence_norm = percentile_ranks_across_models_v2(coherence_raw, axis=0)`

3. **Add to base_forecast computation:**
   ```python
   if cfg.return_rank_coherence_weight != 0 and coherence_norm is not None:
       base_forecast = base_forecast + cfg.return_rank_coherence_weight * coherence_norm
   ```

4. **Implementation specifics:**
   - The signal is computed per model per period using raw returns R (already available) and percentile ranks Q (already computed)
   - Models that are top-ranked AND have large return advantages get high coherence scores
   - Models that are top-ranked but achieved it with tiny return margins get low coherence scores
   - This filters out "lucky" top-ranked models whose positions are unstable
   - Handle NaN by defaulting to neutral (0.5) when insufficient data
   - The signal is causal: uses only data from period t to inform scoring at t+1

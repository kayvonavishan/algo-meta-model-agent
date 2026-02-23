IDEA: Add a "Positive-Environment Alignment" signal that measures how often a model achieves above-median rank (Q > 0.5) specifically during periods when the cross-sectional mean return is positive, penalizing models that underperform precisely when conditions are favorable

RATIONALE: The current model shows 15.15% of periods where baseline returns are positive but topN returns are negative (`pct_baseline_positive_meta_negative`). This indicates the meta model sometimes selects "contrarian" models that fail to capitalize when the overall environment is favorable. The existing downmarket resilience signal (if added) would address maintaining rank during stress, but this complementary signal ensures models also *participate in upside*. A model that consistently drops below median rank during positive market periods is either timing-mismatched or strategically defensive in an unhelpful way. By tracking the fraction of positive-return periods (cross-sectional mean > 0) where a model stays above median rank over a lookback window, we identify models that consistently participate in favorable regimes. This is training-free, uses only causal lookback data, and follows the successful pattern of threshold-based counting signals (breakout, durability). Unlike hit_asymmetry (which computes a ratio of hit rates at a higher threshold Q>0.75), this uses Q>0.5 and directly measures participation rate in favorable periods.

REQUIRED_CHANGES: 1. **In `config.py`**, add new parameters:
   - `positive_regime_alignment_weight: float = 0.08` — weight for the positive regime alignment signal (similar to breakout_weight and durability_weight)
   - `positive_regime_lookback: int = 8` — how many periods to examine for alignment assessment

2. **In `scoring.py`, within `compute_scores_for_ticker_v2` (after computing Q percentile ranks):**
   - Identify "positive regime periods": periods where the cross-sectional mean return (across models for this ticker) is positive: `is_positive_regime = np.nanmean(R, axis=0) > 0`
   - For each model and time t, over the lookback window `[t-L+1, t]`:
     - Count how many positive-regime periods occurred: `n_positive = sum(is_positive_regime[t-L+1:t+1])`
     - Count how many of those the model was above median: `n_aligned = sum((Q[:, t-L+1:t+1] > 0.5) & is_positive_regime[t-L+1:t+1])`
     - Compute `alignment_ratio = n_aligned / max(1, n_positive)` — fraction of positive periods where model stayed above median
     - If no positive periods in window, default to 0.5 (neutral)
   - Create `alignment_raw` matrix of shape (n_models, T)
   - Normalize via percentile ranking across models per period: `alignment_norm = percentile_ranks_across_models_v2(alignment_raw, axis=0)`

3. **Add to base_forecast computation (after existing feature additions):**
   ```python
   if cfg.positive_regime_alignment_weight != 0 and alignment_norm is not None:
       base_forecast = base_forecast + cfg.positive_regime_alignment_weight * alignment_norm
   ```

4. **Implementation specifics:**
   - Vectorized approach:
     - `is_positive_regime = (np.nanmean(R, axis=0) > 0).astype(float)` — shape (T,)
     - For each t, slice the window and compute the ratio using masked array operations
     - Alternatively, use `pd.DataFrame` rolling with a custom function
   - The signal is causal: uses only R and Q values through period t
   - Handle NaN in Q by treating as "not above median" (conservative)
   - This complements downmarket resilience by ensuring models participate in both good and bad regimes appropriately (downmarket resilience ensures models don't collapse during stress; positive regime alignment ensures they don't miss upside)

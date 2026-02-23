IDEA: Add a "Rank Stability Under Stress" signal that specifically measures how well a model maintains its rank during periods when the cross-sectional median return is negative (market stress periods), favoring models that demonstrate resilience rather than just overall stability.**

Add a "Rank Stability Under Stress" signal that measures a model's ability to maintain high ranks specifically during adverse cross-sectional periods (when median model return is negative), independently weighting defensive resilience versus broad stability

RATIONALE: The current confidence signal measures rank stability uniformly across all periods, treating good and bad market conditions equally. However, the downside capture ratio is nearly 1.0 (0.983), meaning the meta model doesn't effectively avoid downside. This suggests the selection is picking models that are stable on average but collapse during stress. By computing rank volatility separately for "stress periods" (when the cross-sectional median return is negative) versus normal periods, we can identify models that specifically maintain high ranks when most models are struggling. This is different from the rejected regime-adaptive risk scaling (which tried to adjust ALL scoring based on regime) because we're creating a separate additive feature signal that rewards stress resilience without disrupting the existing scoring. It's also different from hit_asymmetry (which measures hit-rate in good vs bad periods) because we're measuring rank *stability* in bad periods, not just frequency of being top-ranked. Models that are stable during stress are less likely to contribute to the extended drawdown duration (22 periods) observed in current metrics. The signal is training-free, causal, and captures a signal dimension orthogonal to existing features.

REQUIRED_CHANGES: 1. **In `config.py`**, add new parameters:
   - `stress_stability_weight: float = 0.08` — weight for the stress stability signal
   - `stress_stability_lookback: int = 8` — lookback window for computing stress-period stability
   - `stress_stability_threshold: float = 0.0` — threshold for defining "stress" (cross-sectional median return < threshold)

2. **In `scoring.py`, within `compute_scores_for_ticker_v2` (after computing Q percentile ranks and R returns matrix):**
   - Identify stress periods: `cs_median_return = np.nanmedian(R, axis=0)` then `is_stress = cs_median_return < cfg.stress_stability_threshold`
   - For each model and time t, over the lookback window:
     - Count stress periods in window: `n_stress = sum(is_stress[t-L:t])`
     - If `n_stress >= 2`, compute rank std only over stress periods: `stress_std = std(Q[:, stress_indices_in_window])`
     - If `n_stress < 2`, use neutral value (will be normalized to ~0.5)
   - Create `stress_stability_raw = 1.0 / (stress_std + cfg.conf_eps)` — inverse of stress-period rank volatility (lower vol = higher stability)
   - Normalize via percentile ranking: `stress_stability_norm = percentile_ranks_across_models_v2(stress_stability_raw, axis=0)`

3. **Add to base_forecast computation (after existing feature additions, around line ~548):**
   ```python
   if cfg.stress_stability_weight != 0 and stress_stability_norm is not None:
       base_forecast = base_forecast + cfg.stress_stability_weight * stress_stability_norm
   ```

4. **Implementation specifics:**
   - Vectorize by computing rolling windows and masking with the stress indicator
   - Use `pd.DataFrame(Q.T).rolling(window=L)` combined with conditional logic on stress mask
   - Handle edge cases where no stress periods exist in window (assign neutral 0.5 rank)
   - This differs from general confidence because it ONLY considers rank behavior during stress periods
   - The signal rewards models that don't drop in rank when times get tough, even if their overall rank volatility is normal

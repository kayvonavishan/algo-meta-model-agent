IDEA: Add a regime-aware baseline that adjusts the cross-sectional centering based on whether the ticker is in a trending vs mean-reverting state, rather than using a fixed median/mean baseline for all periods.

RATIONALE: The current baseline subtraction (Step 6.6 in the guide) uses a static method (median or mean of base scores) regardless of market regime. This can mis-handle trending and mean-reverting regimes. A regime-aware baseline would reduce baseline adjustment during trending (to preserve absolute strength) and amplify it during mean-reverting (to emphasize relative stability).

REQUIRED_CHANGES:

1. Regime detection: reuse dispersion z-scores already computed (Step 6.2.2) to classify periods:
   - Trending: z_t < z_trend_threshold (e.g., -0.5)
   - Mean-reverting: z_t > z_revert_threshold (e.g., +0.5)
   - Neutral: in between
   Add to MetaConfig:
   ```
   z_trend_threshold: float = -0.5
   z_revert_threshold: float = 0.5
   baseline_trend_damping: float = 0.3
   baseline_revert_amplify: float = 1.5
   ```

2. Regime-aware baseline in scoring.py (Step 6.6):
   - Compute standard baseline (median or mean).
   - Compute regime multipliers per period from dispersion z and thresholds.
   - Apply adjusted baseline: `baseline_adjusted = baseline_raw * regime_multiplier`
   - Use `rel_scores = base_scores - baseline_adjusted`.

3. Integration:
   - Ensure dispersion z-series is available where baseline is computed (`compute_scores_for_ticker_v2`).
   - Insert the regime logic before baseline subtraction; leave downstream steps unchanged.

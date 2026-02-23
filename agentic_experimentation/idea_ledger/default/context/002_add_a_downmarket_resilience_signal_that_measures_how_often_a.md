IDEA: Add a "Downmarket Resilience" signal that measures how often a model maintains above-median rank (Q > 0.5) specifically during periods when the cross-sectional median return is negative, rewarding models that hold their rank even when the overall environment is difficult

RATIONALE: The current model has a downside capture ratio of 0.937, which is good but not great—it means topN still captures 94% of baseline losses during down periods. The existing hit_asymmetry signal measures whether a model reaches top quartile in bad periods, but this is a high bar (Q > 0.75 during down markets). A gentler signal—simply tracking whether a model stays above median (Q > 0.5) during negative cross-sectional periods—captures a different property: "doesn't collapse when the market is tough." This aligns with the successful durability signal (threshold-based counting) but conditions on market regime. The OOS improvement (+1.27% delta per trade) suggests regime-aware signals have potential. Models that frequently drop below median precisely when the overall market is bad are "fair-weather performers" that create false positives. This signal identifies "all-weather" models that maintain respectable rank regardless of environment. It differs from hit_asymmetry (which looks at reaching top tier during stress) and from the failed regime-adaptive risk scaling (which modified the entire risk penalty dynamically, introducing instability). This is a simple additive feature signal using the successful counting-based pattern.

REQUIRED_CHANGES: 1. **In `config.py`**, add new parameters:
   - `downmarket_resilience_weight: float = 0.08` — weight for the downmarket resilience signal (similar to breakout_weight and durability_weight)
   - `downmarket_resilience_lookback: int = 8` — how many periods to examine for resilience assessment

2. **In `scoring.py`, within `compute_scores_for_ticker_v2` (after computing Q percentile ranks):**
   - Identify "downmarket periods": periods where the cross-sectional median return (across models for this ticker) is negative: `is_downmarket = np.nanmedian(R, axis=0) < 0`
   - For each model and time t, over the lookback window `[t-L+1, t]`:
     - Count how many downmarket periods occurred: `n_downmarket = sum(is_downmarket[t-L+1:t+1])`
     - Count how many of those the model was above median: `n_resilient = sum((Q[:, t-L+1:t+1] > 0.5) & is_downmarket[t-L+1:t+1])`
     - Compute `resilience_ratio = n_resilient / max(1, n_downmarket)` — fraction of downmarket periods where model stayed above median
     - If no downmarket periods in window, default to 0.5 (neutral)
   - Create `resilience_raw` matrix of shape (n_models, T)
   - Normalize via percentile ranking across models per period: `resilience_norm = percentile_ranks_across_models_v2(resilience_raw, axis=0)`

3. **Add to base_forecast computation (after existing feature additions):**
   ```python
   if cfg.downmarket_resilience_weight != 0 and resilience_norm is not None:
       base_forecast = base_forecast + cfg.downmarket_resilience_weight * resilience_norm
   ```

4. **Implementation specifics:**
   - Vectorized approach:
     - `is_downmarket = (np.nanmedian(R, axis=0) < 0).astype(float)` — shape (T,)
     - For each t, slice the window and compute the ratio using masked array operations
     - Alternatively, use `pd.DataFrame` rolling with a custom function
   - The signal is causal: uses only R and Q values through period t
   - Handle NaN in Q by treating as "not above median" (conservative)
   - This differs from hit_asymmetry in two key ways: (a) uses Q > 0.5 threshold instead of Q > 0.75, and (b) counts fraction rather than computing a ratio of hit rates between good/bad periods

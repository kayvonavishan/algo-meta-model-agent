IDEA: Add a "Rank-Level Weighted Confidence" signal that scales confidence not just by rank stability, but by the interaction between stability and absolute rank level

Add a "Regime-Conditional Rank Momentum" signal that boosts momentum scores during favorable market regimes (when the cross-sectional median return is positive) and dampens them during unfavorable regimes

Add a "Rank-Level Weighted Confidence" signal that multiplies confidence by rolling mean rank, preferentially selecting models that are both stable AND consistently high-ranking

RATIONALE: The current confidence measure (`1 / (rolling_std(Q) + eps)`) rewards rank stability regardless of where the model ranks. However, a model that's *stably bad* (consistently ranking low, e.g., Q ≈ 0.2 with low std) gets the same confidence boost as a model that's *stably excellent* (consistently ranking high, e.g., Q ≈ 0.9 with low std). For model selection, we care about models that are both **stable AND high-ranking**. By multiplying the confidence by the rolling mean rank (or a transformed version), we create a signal that preferentially selects models demonstrating sustained top-tier performance, rather than just consistent performance at any level. This targets false positives from stable mediocre models that might score well due to low volatility alone.

Looking at the baseline metrics, the OOS performance is significantly lower than full-sample performance (0.037% vs 0.14%), and the rolling min Sharpe can be quite negative (-0.87). This suggests the model may be selecting high-momentum models that perform well in benign conditions but fail in adverse regimes. By conditioning the momentum signal on the market regime (defined as periods where the cross-sectional median return is positive vs negative), we can adjust our confidence in momentum signals. When the overall market is favorable (positive median returns), momentum signals are more reliable; when the market is adverse, momentum may mean-revert. Adding a regime interaction term can help reduce false positives from momentum signals during regime transitions.

The current confidence measure rewards rank stability regardless of where the model ranks. A model that's stably mediocre (consistently ranking at Q ≈ 0.3 with low volatility) receives a similar confidence boost as a model that's stably excellent (Q ≈ 0.85 with low volatility). For optimal selection, we want models demonstrating sustained top-tier performance, not just consistency at any level. By multiplying confidence by rolling mean rank, we create a signal that filters out "stable losers" that might score well due to low volatility alone. This is training-free, uses only lookback data, and targets a specific false positive case in the current scoring: stable mediocrity being mistaken for reliable quality.

REQUIRED_CHANGES: 1. **In `config.py`**, add new parameters:
   - `rank_level_conf_weight: float = 0.10` — weight for the rank-level-adjusted confidence signal
   - `rank_level_conf_lookback: int = 8` — lookback for computing the rolling mean rank

2. **In `scoring.py`, within `compute_scores_for_ticker_v2`:**
   - After computing `CONF` (percentile-ranked confidence based on rank stability), compute a rolling mean of `Q` over `rank_level_conf_lookback` periods.
   - Create a new signal: `rank_level_conf = CONF * rolling_mean(Q)` (or equivalently, percentile-rank the product across models per period)
   - Normalize via percentile ranking across models for each period to get `rank_level_conf_norm`

3. **Add to base_forecast computation (around line ~494-504):**
   - `base_forecast += cfg.rank_level_conf_weight * rank_level_conf_norm`

4. **Implementation specifics:**
   - Use `pd.DataFrame(Q.T).rolling(window=cfg.rank_level_conf_lookback, min_periods=2).mean()` to get rolling mean ranks
   - Multiply element-wise with CONF, then apply `percentile_ranks_across_models_v2` for normalization
   - Handle NaN edge cases similarly to existing signals

---


1. **In `config.py`**, add new parameters:
   - `regime_momentum_interaction_weight: float = 0.08` — weight for regime-conditioned momentum
   - `regime_momentum_lookback: int = 6` — lookback for regime state smoothing

2. **In `scoring.py`, within `compute_scores_for_ticker_v2`:**
   - Compute a rolling indicator of "favorable regime" based on the cross-sectional median return being positive: `regime_good_t = EMA(cross_sectional_median(R) > 0, lookback)`
   - Create: `regime_momentum = M * regime_good_t` (momentum scaled by regime favorability)
   - Normalize via percentile ranking across models per period

3. **Add to base_forecast computation:**
   - `base_forecast += cfg.regime_momentum_interaction_weight * regime_momentum_norm`

---

Given the analysis and the rejected idea (autocorrelation of rank changes), I'll propose the **Rank-Level Weighted Confidence** idea as it addresses a concrete gap in the current model architecture.

---


1. **In `config.py`**, add new parameters:
   - `rank_level_conf_weight: float = 0.10` — weight applied to the rank-level-adjusted confidence signal (similar magnitude to other feature weights like `efficiency_weight=0.10`)
   - `rank_level_conf_lookback: int = 8` — window for computing rolling mean rank (should align with conf_lookback or be slightly shorter)

2. **In `scoring.py`, within `compute_scores_for_ticker_v2` (after CONF computation around line ~539):**
   - Compute rolling mean of Q over the lookback window:
     ```python
     Q_df = pd.DataFrame(Q)  # already exists for confidence
     rolling_mean_Q = Q_df.rolling(window=cfg.rank_level_conf_lookback, axis=1, min_periods=2).mean().to_numpy()
     ```
   - Create the rank-level-weighted confidence signal by element-wise multiplication:
     ```python
     rank_level_conf_raw = CONF * rolling_mean_Q
     ```
   - Normalize via percentile ranking:
     ```python
     rank_level_conf_norm = percentile_ranks_across_models_v2(rank_level_conf_raw, axis=0)
     ```

3. **Add to base_forecast computation (around line ~502-504):**
   - After existing feature additions, add:
     ```python
     if cfg.rank_level_conf_weight != 0:
         base_forecast = base_forecast + cfg.rank_level_conf_weight * rank_level_conf_norm
     ```

4. **Implementation note:**
   - This signal is computed after CONF but before the final scoring, so it can be injected into the existing feature addition block
   - Handle NaN propagation naturally—models with insufficient data get NaN which doesn't affect the base_forecast sum

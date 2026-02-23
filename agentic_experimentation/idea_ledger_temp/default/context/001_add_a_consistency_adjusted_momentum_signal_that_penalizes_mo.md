IDEA: Add a "Consistency-Adjusted Momentum" signal that penalizes momentum scores for models with high rank autocorrelation volatility (erratic rank reversals)**

RATIONALE: The current momentum signal rewards models with high recent ranks, but doesn't distinguish between models that climb ranks steadily versus those that spike up briefly due to luck. A model that persistently improves ranks over time (high positive autocorrelation of rank changes) is more likely to continue performing well than one with erratic rank reversals. By computing the rolling autocorrelation of rank changes (ΔQ), we can identify "steady climbers" versus "rank oscillators." Models with negative or volatile autocorrelation of rank changes tend to mean-revert (their good ranks don't persist), so penalizing/downweighting such models in the momentum score should improve selection quality and reduce false positives from temporary rank spikes. This is training-free and uses only lookback data available at selection time.

REQUIRED_CHANGES: 1. **In `scoring.py`, within `compute_scores_for_ticker_v2`:**
   - After computing the delta `D[:, 1:] = Q[:, 1:] - Q[:, :-1]` (rank changes), compute a rolling first-order autocorrelation of D over a lookback window (e.g., `rank_autocorr_lookback = 6-8 periods`).
   - Formula: For each model and time t, compute `autocorr(D_{t-L:t}) = corr(D[t-L:t-1], D[t-L+1:t])`.
   - Normalize the autocorrelation to [0, 1] using percentile ranks across models for each period.
   - Models with high positive autocorrelation (steady momentum) get higher normalized scores; models with negative autocorrelation (mean-reverting) get lower scores.

2. **In `config.py`, add new parameters:**
   - `rank_autocorr_weight: float = 0.08` — weight for the consistency signal (similar to other feature weights)
   - `rank_autocorr_lookback: int = 6` — lookback window for autocorrelation computation

3. **In the base_forecast computation (scoring.py line ~494-504):**
   - Add: `base_forecast += cfg.rank_autocorr_weight * rank_autocorr_norm` (similar to existing feature additions)

4. **Implementation Note:**
   - Use `pd.DataFrame(D.T).rolling(window=L).apply(lambda x: x[:-1].corr(x[1:]))` or a vectorized lag-correlation approach.
   - Handle edge cases where the window has insufficient data (return 0.5 or NaN, then fillna with 0.5).

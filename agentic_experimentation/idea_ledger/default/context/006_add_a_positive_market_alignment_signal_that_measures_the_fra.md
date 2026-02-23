IDEA: Add a "Positive Market Alignment" signal that measures the fraction of recent positive-market periods (where cross-sectional median return > 0) during which a model achieved a top-half rank (Q > 0.5), rewarding models that reliably rank well when the overall market environment is favorable.

RATIONALE: The ~9-15% "baseline positive, meta negative" problem indicates the meta model sometimes selects strategies that underperform when the broader universe is doing well. This could happen when selected models rank well during mixed/negative periods but fail during positive market regimes. By tracking each model's hit rate *specifically during positive market periods*, we can identify and favor models that are aligned with favorable market conditions - models that consistently rank well when conditions are good are more likely to contribute positive returns when selected. This is distinct from:
- Durability (measures consecutive tenure above median, not conditioned on market regime)
- Hit asymmetry (measures top-tier hit rate during bad vs good periods, focuses on asymmetry ratio rather than absolute alignment with good periods)
- Conviction alignment (measures alignment with high score-spread periods, not market return periods)

This follows the successful pattern of simple counting/threshold logic and rewards a positive pattern (ranking well during good times) rather than penalizing a negative one.

REQUIRED_CHANGES: 1. **In `config.py`**, add new parameters:
   - `positive_market_alignment_weight: float = 0.07` — weight for the positive market alignment signal
   - `positive_market_alignment_lookback: int = 8` — rolling window for measuring alignment
   - `positive_market_rank_threshold: float = 0.5` — Q threshold to count as "performing well" (top half)

2. **In `scoring.py`, within `compute_scores_for_ticker_v2` (after computing Q percentile ranks and having access to R returns matrix):**
   - Identify positive market periods per ticker using cross-sectional median return:
     ```python
     # R shape: (n_models, T)
     cs_median_return = np.nanmedian(R, axis=0)  # shape (T,)
     is_positive_market = cs_median_return > 0   # boolean array shape (T,)
     ```
   - For each model at time t, over the lookback window, compute:
     - Count of positive market periods in window
     - Count of positive market periods where model had Q > threshold
     - `alignment_rate = n_top_in_positive_market / (n_positive_market_periods + eps)`
   - Normalize via percentile ranking: `positive_market_alignment_norm = percentile_ranks_across_models_v2(alignment_rate, axis=0)`

3. **Add to base_forecast computation (after existing feature additions):**
   ```python
   if cfg.positive_market_alignment_weight != 0 and positive_market_alignment_norm is not None:
       base_forecast = base_forecast + cfg.positive_market_alignment_weight * positive_market_alignment_norm
   ```

4. **Implementation specifics:**
   - Create rolling counts efficiently:
     ```python
     L = cfg.positive_market_alignment_lookback
     # For each model, track (Q > threshold) AND (is_positive_market)
     is_top_in_positive = (Q > cfg.positive_market_rank_threshold) & is_positive_market[None, :]
     # Rolling count of positive market periods
     pos_mkt_df = pd.DataFrame(is_positive_market.astype(float).reshape(1, -1)).T
     n_pos_market = pos_mkt_df.rolling(window=L, min_periods=1).sum().to_numpy().flatten()
     # Rolling count of (top AND positive market) per model
     top_pos_df = pd.DataFrame(is_top_in_positive.T.astype(float))
     n_top_in_pos = top_pos_df.rolling(window=L, min_periods=1).sum().to_numpy().T
     # Alignment rate
     alignment_rate = np.divide(n_top_in_pos, n_pos_market[None, :] + 1e-8, where=n_pos_market[None, :] > 0, out=np.full_like(n_top_in_pos, 0.5))
     ```
   - Handle cases where no positive market periods exist in window by defaulting to 0.5 (neutral)
   - This is causal: uses data through period t-1 for scoring at t (via final causal shift)
   - The signal rewards models that are "fair weather performers" in a positive sense - they rank well when the market is favorable

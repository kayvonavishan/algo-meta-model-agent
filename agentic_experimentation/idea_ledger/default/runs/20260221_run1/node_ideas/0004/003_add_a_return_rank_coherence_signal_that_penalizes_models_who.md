IDEA: Add a "Return-Rank Coherence" signal that penalizes models whose recent raw return ranking differs significantly from their percentile rank (Q), identifying models with unstable position within the return distribution.**

Wait - I see that "return-to-rank coherence" was already tested (eval_id=0060, rejected). Let me think of something different.

Looking at the metrics:
- The `rolling_hit_rate_min` for topN is very low (16.7%) - there are windows where the model performs poorly
- The `max_underperformance_streak` is 3 periods
- `downside_capture` is 0.95 - nearly 1:1 capture of downside moves

A key insight: existing signals focus on rank levels and rank changes, but none specifically measure **how the model's performance compares to the cross-sectional median during bad market periods vs good market periods**. The `hit_asymmetry` signal exists but measures something different (hit rate of being in top tier during bad vs good periods).

Add a "Downside Outperformance Rate" signal that tracks the fraction of recent periods where a model's return exceeded the cross-sectional median specifically when the cross-sectional median was negative—rewarding models that relatively outperform during market stress periods.

Add a "Relative Return Stability" signal** that measures the coefficient of variation (CV) of a model's return *relative* to the cross-sectional median over a lookback window—models with lower relative volatility (more consistent relative performance) get higher scores.

Based on the analysis and avoiding what's already been tested, here's my recommendation:

Add a "Consecutive Outperformance Tenure" signal that counts the number of consecutive recent periods where a model's raw return exceeded the cross-sectional median return (not the absolute 0.5 rank threshold, but the actual per-period median return), rewarding models with longer unbroken streaks of being "above average"**

Add a "Rank Trough Recovery" signal that identifies models currently in the top ranks (Q > 0.7) that recently experienced a trough (dipped to Q < 0.4 in the past N periods) and then recovered, versus models that have stayed consistently high. Models that have proven they can recover from drawdowns may be more robust.

Actually this overlaps with the breakout signal. Let me think again...

The key insight from the rejected ideas: ideas that tried to penalize models (consistency penalties, drawdown recovery penalties, etc.) generally performed worse. The successful ideas (breakout, durability, velocity confirmation, score spread gating) generally *reward* positive patterns.

Here's my final idea:

---

Add a "Median-Relative Return Consistency" signal that computes, for each model over a lookback window, the fraction of periods where the model's return was within the top tercile (Q > 0.67) of returns AND the model's return was positive in absolute terms — rewarding models that combine high relative ranking with actual positive returns.

Add a "Positive Top-Tercile Consistency" signal that counts the fraction of recent periods where a model achieved BOTH a high relative rank (Q > 0.67) AND a positive absolute return, rewarding models that combine strong relative performance with actual profitability.

RATIONALE: The existing rank durability signal measures tenure above the 0.5 *percentile rank* threshold, but this is subtly different from measuring tenure above the *actual cross-sectional median return*. In periods with asymmetric return distributions (e.g., fat right tail), a model could have a rank > 0.5 but still have a return below the cross-sectional mean. This signal directly measures: "how many consecutive periods has this model beaten the average?" — a more intuitive measure of sustained above-average performance. This directly addresses the 15% baseline-positive-meta-negative problem by filtering out models that may have decent ranks but actually deliver below-median returns in recent periods. The signal uses simple counting logic similar to the successful durability signal.

Actually, wait - let me reconsider. The rank percentile Q is already based on returns, so a model with Q > 0.5 should by definition have beaten the median return. Let me think differently...

A better angle: The current signals don't explicitly track **rank trajectory shape**. Let me think about what patterns might indicate future underperformance:

- Models that have been stuck at mid-ranks (Q around 0.45-0.55) for several periods but suddenly spike - these may be noise spikes
- Models whose momentum M is high but whose most recent few Q values are declining (early reversal)
- Models that have high average rank but high rank variance in recent periods

Let me propose something that captures a missing pattern:

The current signals focus on relative ranks (Q), which measure performance vs other models, but don't directly verify that highly-ranked models are actually generating positive absolute returns. A model could rank well (Q > 0.5) by losing less than peers during a broad drawdown, but such a model is still losing money. The 15% "baseline positive, meta negative" problem suggests the model sometimes selects strategies that rank well but still deliver negative absolute returns. By adding a signal that rewards models achieving *both* high relative rank AND positive absolute return, we can filter toward models that are genuinely performing well rather than just "least bad." This is training-free, uses simple counting logic (similar to durability), and captures a fundamentally different dimension: absolute + relative performance conjunction rather than just relative rank. The tercile threshold (0.67) is stricter than median (0.50) to focus on truly top performers.

The current signals focus exclusively on relative ranks (percentile positions vs other models), but a model can rank well simply by losing less than peers during market drawdowns. The ~15% "baseline positive, meta negative" problem—where selected models underperform when the overall baseline is positive—suggests the meta model sometimes picks strategies that look good on rank metrics but aren't generating positive absolute returns. By rewarding models that are simultaneously (1) in the top third of performers AND (2) making money in absolute terms, we filter toward genuinely strong performers rather than "least bad" ones. This mirrors the successful pattern of simple counting/threshold signals (like durability and breakout) but captures a different dimension: the conjunction of relative rank excellence and absolute profitability. Unlike rank persistence (which just counts Q > 0.5) or hit asymmetry (which measures performance during market stress), this signal specifically targets the "top + profitable" combination that directly predicts whether selected models will deliver positive returns.

REQUIRED_CHANGES: 1. **In `config.py`**, add new parameters:
   - `positive_top_tercile_weight: float = 0.08` — weight for the positive top-tercile signal
   - `positive_top_tercile_lookback: int = 6` — lookback window for measuring consistency
   - `positive_top_tercile_rank_threshold: float = 0.67` — rank percentile threshold (top tercile)

2. **In `scoring.py`, within `compute_scores_for_ticker_v2` (after computing Q and having access to R):**
   - For each model and time t, count periods in the lookback window where BOTH:
     - `Q[:, k] > cfg.positive_top_tercile_rank_threshold` (in top tercile)
     - `R[:, k] > 0` (actual positive return)
   - Compute `fraction_qualified = count_qualified / lookback_length`
   - Normalize via percentile ranking: `positive_top_tercile_norm = percentile_ranks_across_models_v2(fraction_qualified, axis=0)`

3. **Add to base_forecast computation (after existing feature additions):**
   ```python
   if cfg.positive_top_tercile_weight != 0 and positive_top_tercile_norm is not None:
       base_forecast = base_forecast + cfg.positive_top_tercile_weight * positive_top_tercile_norm
   ```

4. **Implementation specifics:**
   - Create boolean mask: `is_top_tercile = Q > cfg.positive_top_tercile_rank_threshold`
   - Create boolean mask: `is_positive = R > 0`
   - Combined mask: `is_qualified = is_top_tercile & is_positive`
   - Use `pd.DataFrame(is_qualified.T).rolling(window=lookback, min_periods=1).mean()` to get fraction
   - Handle NaN by treating as not qualified (mask is False)
   - The signal is causal: uses data through period t-1 for scoring at t (via the final causal shift)

---

1. **In `config.py`**, add new parameters:
   - `positive_top_tercile_weight: float = 0.08` — weight for the signal (similar magnitude to other feature weights)
   - `positive_top_tercile_lookback: int = 6` — rolling window for counting qualified periods
   - `positive_top_tercile_rank_threshold: float = 0.67` — Q must exceed this to be "top tercile"

2. **In `scoring.py`, within `compute_scores_for_ticker_v2` (after computing Q percentile ranks and having R returns matrix):**
   - Build joint condition per model per period: `is_qualified = (Q > cfg.positive_top_tercile_rank_threshold) & (R > 0)`
   - Convert to float: `qualified_float = is_qualified.astype(float)`
   - Handle NaN in Q or R by setting qualified to False (NaN propagates as False in comparison)
   - Compute rolling mean over lookback:
     ```python
     L = cfg.positive_top_tercile_lookback
     qualified_df = pd.DataFrame(qualified_float.T)  # shape (T, n_models)
     fraction_qualified = qualified_df.rolling(window=L, min_periods=1).mean().to_numpy().T  # shape (n_models, T)
     ```
   - Normalize via percentile ranking: `positive_top_tercile_norm = percentile_ranks_across_models_v2(fraction_qualified, axis=0)`

3. **Add to base_forecast computation (after existing feature additions, around line ~613):**
   ```python
   if cfg.positive_top_tercile_weight != 0 and positive_top_tercile_norm is not None:
       base_forecast = base_forecast + cfg.positive_top_tercile_weight * positive_top_tercile_norm
   ```

4. **Implementation specifics:**
   - This is fully vectorized using pandas rolling operations
   - The threshold of 0.67 (top tercile) is stricter than the 0.50 (median) used by durability, focusing on truly top performers
   - Models that consistently rank highly AND generate positive returns score well; models that rank well but lose money get lower scores
   - This is causal: the final causal shift ensures scores for period t use data through t-1
   - Default to 0.5 (neutral) when insufficient lookback data exists

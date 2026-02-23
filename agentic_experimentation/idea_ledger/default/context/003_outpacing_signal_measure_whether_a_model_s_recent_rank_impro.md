IDEA: "Outpacing" Signal** — measure whether a model's recent rank improvement rate (average ΔQ over lookback) exceeds the cross-sectional median of improvement rates. Models that are "outpacing" peers get a bonus. This captures the competitive dynamics: it's not just about improving your rank, but about improving faster than others are improving.

This is different from existing signals because:
- Momentum (M) tracks smoothed rank levels, not improvement rates
- Delta (D) is single-period changes, not aggregated
- Durability counts consecutive tenure above 0.5, not improvement rate
- Breakout detects threshold crossings, not relative improvement speed

Let me formulate this properly:

---

Add an "Outpacing" signal that measures whether a model's recent rank improvement rate exceeds the cross-sectional median improvement rate, rewarding models that are gaining ground faster than their peers

RATIONALE: The current signals reward models based on their absolute rank momentum (M), single-period deltas (D), and threshold behaviors (breakout, durability). However, none explicitly capture *relative momentum* — whether a model is improving faster than the typical model. During regime transitions, many models may improve simultaneously, but the models that improve faster are more likely to emerge as leaders. Conversely, a model improving slowly while others surge is falling behind relatively, even if its absolute rank rises. By computing the rolling average of rank changes (ΔQ) for each model and comparing to the cross-sectional median of improvement rates, we identify "outpacing" models that are competitively gaining ground. This signal follows the successful pattern of simple threshold-based comparisons (like breakout and durability) rather than complex autocorrelation measures (which failed). It's training-free, causal, and captures a distinct dimension: competitive improvement dynamics.

REQUIRED_CHANGES: 1. **In `config.py`**, add new parameters:
   - `outpacing_weight: float = 0.08` — weight for the outpacing signal (similar to breakout_weight and durability_weight)
   - `outpacing_lookback: int = 5` — how many periods to average rank changes over

2. **In `scoring.py`, within `compute_scores_for_ticker_v2` (after computing D = rank deltas):**
   - Compute rolling average of rank changes for each model:
     ```python
     D_df = pd.DataFrame(D.T)  # shape: (T, n_models)
     avg_improvement = D_df.rolling(window=cfg.outpacing_lookback, min_periods=2).mean()
     ```
   - Compute cross-sectional median improvement rate per period:
     ```python
     cs_median_improvement = avg_improvement.median(axis=1)  # shape: (T,)
     ```
   - Create binary "outpacing" indicator: 1 if model's improvement rate > median, 0 otherwise:
     ```python
     outpacing_raw = (avg_improvement.gt(cs_median_improvement, axis=0)).astype(float)
     outpacing_raw = outpacing_raw.to_numpy().T  # back to (n_models, T)
     ```
   - Normalize via percentile ranking across models per period: `outpacing_norm = percentile_ranks_across_models_v2(outpacing_raw, axis=0)`

3. **Add to base_forecast computation (after existing feature additions):**
   ```python
   if cfg.outpacing_weight != 0 and outpacing_norm is not None:
       base_forecast = base_forecast + cfg.outpacing_weight * outpacing_norm
   ```

4. **Implementation specifics:**
   - Handle NaN in D by letting rolling mean handle them naturally (NaN if insufficient data)
   - The comparison to cross-sectional median is a simple threshold (> median), similar to successful signals
   - The percentile normalization ensures the signal integrates smoothly with other features
   - This is causal: uses only D values computed from Q history through period t

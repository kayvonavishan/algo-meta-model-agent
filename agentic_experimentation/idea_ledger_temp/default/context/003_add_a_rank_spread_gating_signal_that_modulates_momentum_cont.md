IDEA: Add a "Rank Spread Gating" signal that modulates momentum contribution based on the cross-sectional dispersion of ranks within each period**

RATIONALE: The current scoring weights momentum equally regardless of whether rankings are highly differentiated or tightly clustered. When the cross-sectional dispersion of ranks (Q values) is high in a period, it indicates clear performance differentiation among models—in these periods, top-ranked models genuinely outperformed, making momentum signals more reliable. Conversely, when dispersion is low (all models perform similarly), rankings are noisy and momentum signals are less meaningful. By computing a rolling measure of cross-sectional rank dispersion and using it to scale the momentum contribution, we can reduce exposure during periods when selections are essentially random and increase exposure when there's genuine signal. This directly addresses the OOS degradation problem: often, regime changes cause rankings to compress (all models fail or succeed together), and the current model continues trusting momentum despite this compression. This signal is training-free and uses only lookback data.

REQUIRED_CHANGES: 1. **In `config.py`**, add new parameters:
   - `rank_spread_gate_weight: float = 0.10` — controls how much cross-sectional rank spread affects the final score
   - `rank_spread_lookback: int = 4` — rolling window for smoothing the rank spread measure
   - `rank_spread_threshold: float = 0.15` — minimum cross-sectional std of Q below which the gate is fully suppressed (0 = full suppression, 1 = no effect)

2. **In `scoring.py`, within `compute_scores_for_ticker_v2`:**
   - After computing Q (percentile ranks), compute the cross-sectional standard deviation of Q for each period: `cs_std_Q_t = np.nanstd(Q[:, t])` for each t
   - Smooth with a rolling mean over `rank_spread_lookback` periods: `smoothed_spread = pd.Series(cs_std_Q).rolling(window=cfg.rank_spread_lookback, min_periods=1).mean()`
   - Normalize to [0, 1] using min-max scaling across the time dimension (or clipping to `[threshold, 0.35]` and rescaling, since theoretical max std of uniform [0,1] ranks is ~0.29)
   - Create `spread_gate = (smoothed_spread - cfg.rank_spread_threshold) / (0.30 - cfg.rank_spread_threshold)` clipped to [0, 1]

3. **Modify base_forecast computation (around line ~494):**
   - Scale the momentum contribution by the spread gate: `M_gated = M * spread_gate` 
   - Replace `base_forecast = M + cfg.delta_weight * D` with `base_forecast = M_gated + cfg.delta_weight * D`
   - Alternatively, add as an additive penalty: `base_forecast = base_forecast - cfg.rank_spread_gate_weight * (1 - spread_gate)` (penalizes scores during compressed-rank periods)

4. **Implementation specifics:**
   - The cross-sectional std is computed per period (across models), not per model
   - The resulting `spread_gate` is a 1D array of shape (T,), broadcast to all models in that period
   - This is different from existing signals which vary per model—this is a period-level gate affecting all models equally
   - When `spread_gate` is low (compressed rankings), all model scores in that period are dampened proportionally

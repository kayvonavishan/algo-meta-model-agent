IDEA: Add a "Cross-Sectional Rank Dispersion Dampening" signal that reduces confidence in model scores when the cross-sectional rank dispersion (spread of Q values) is unusually low within a period, indicating that model performance is highly clustered and differentiation is unreliable

RATIONALE: The current model has a concerning pattern: `rolling_hit_rate_min` at 16.7% and `max_underperformance_streak` of 7 periods suggests there are windows where selection is essentially random. The score_spread_boost addresses this from the score side, but doesn't address the underlying issue at the ranking stage. When percentile ranks (Q) have low dispersion within a period — meaning all models performed similarly — the entire ranking is noise-dominated. In these "flat" periods, even the "top" models are barely distinguishable from median performers in raw returns. By computing the within-period dispersion of Q values (which should ideally span [0,1] but may compress when returns cluster), we can identify periods where rankings are unreliable. Rather than boosting during high-spread periods (already done), this applies a multiplicative dampener to CONF during low-dispersion periods — essentially saying "we have less confidence in rankings when models are indistinguishable." This works upstream of score_spread (which operates on final SCORE) by modifying CONF based on Q dispersion. It's training-free, causal, and addresses the root cause of bad rolling windows: selecting from essentially identical candidates.

REQUIRED_CHANGES: 1. **In `config.py`**, add new parameters:
   - `rank_dispersion_confidence_weight: float = 0.15` — how much to dampen confidence when rank dispersion is low
   - `rank_dispersion_lookback: int = 4` — rolling window for rank dispersion statistics
   - `rank_dispersion_z_threshold: float = -0.5` — z-score threshold below which dispersion is "unusually low"

2. **In `scoring.py`, within `compute_scores_for_ticker_v2` (after computing Q percentile ranks, before computing CONF):**
   - Compute per-period rank dispersion: `rank_disp_t = np.nanstd(Q[:, t], ddof=1)` for each period t
   - Compute rolling z-score of rank dispersion:
     ```python
     rank_disp = np.nanstd(Q, axis=0)  # shape (T,)
     rank_disp_series = pd.Series(rank_disp).rolling(window=cfg.rank_dispersion_lookback, min_periods=2)
     rank_disp_z = (rank_disp - rank_disp_series.mean()) / (rank_disp_series.std(ddof=1) + 1e-8)
     ```
   - Create confidence dampening factor:
     ```python
     # When z < threshold (unusually low dispersion), apply dampening
     deficit = cfg.rank_dispersion_z_threshold - rank_disp_z
     deficit = np.where(np.isfinite(deficit), np.maximum(0.0, deficit), 0.0)
     dampener = 1.0 - cfg.rank_dispersion_confidence_weight * deficit
     dampener = np.clip(dampener, 0.5, 1.0)  # floor at 0.5 to avoid zeroing out
     ```

3. **Modify CONF computation (around line ~611-612):**
   - After computing `CONF = percentile_ranks_across_models_v2(raw.to_numpy(), axis=0)`:
     ```python
     if cfg.rank_dispersion_confidence_weight != 0:
         CONF = CONF * dampener[None, :]  # broadcast dampener across models
     ```

4. **Implementation specifics:**
   - The dampener operates on the CONF matrix multiplicatively, reducing confidence scores during low-dispersion periods
   - This differs from score_spread_boost which amplifies final scores in high-spread periods — this dampens confidence in low-spread ranking periods
   - The effect cascades: lower CONF → lower final SCORE → lower selection probability during unreliable periods
   - Handle NaN by defaulting dampener to 1.0 (no change)
   - The 0.5 floor ensures CONF is never completely eliminated

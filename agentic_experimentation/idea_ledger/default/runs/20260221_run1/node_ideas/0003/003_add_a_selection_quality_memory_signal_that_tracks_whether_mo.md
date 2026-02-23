IDEA: Add a "Selection Quality Memory" signal that tracks whether models selected in recent periods subsequently delivered positive returns in the following period, and rewards models that come from tickers with consistently good post-selection outcomes while penalizing models from tickers with poor recent selection track records.

RATIONALE: The current meta model selects top-N models each period based on scores computed from historical rank behavior. However, it doesn't learn from whether those selections *worked* in the immediately following period. Tickers have different levels of "predictability" — some tickers have strong signal-to-noise ratios where high-scoring models consistently deliver, while others are noisy and selections randomly fail. By tracking a per-ticker rolling "selection hit rate" (fraction of recent periods where selected models from that ticker beat the baseline), we can identify which tickers are currently "hot" (good predictability) versus "cold" (noisy/regime-shifted). This directly addresses the 15% `pct_baseline_positive_meta_negative` problem: when a ticker's recent selections have been failing despite positive baseline, the model should reduce allocation to that ticker. Unlike regime-adaptive risk scaling (which failed because it modified the existing risk penalty in complex ways), this is a simple additive/multiplicative bonus based on empirical selection outcomes — similar in spirit to the successful threshold-based signals. It's training-free, causal (uses only past selection outcomes available at t), and targets the root cause of selection quality variance across tickers and time.

REQUIRED_CHANGES: 1. **In `config.py`**, add new parameters:
   - `selection_quality_weight: float = 0.10` — weight for the selection quality signal
   - `selection_quality_lookback: int = 6` — how many past periods of selection outcomes to track
   - `selection_quality_threshold: float = 0.5` — hit rate threshold below which ticker gets penalized

2. **In `scoring.py`, within `compute_scores_for_ticker_v2` (after computing the final SCORE but before causal shift):**
   - Track per-period whether the current ticker's previously-selected models (top scores at t-1) delivered positive returns at t
   - Compute `ticker_selection_hit_rate` = rolling mean over lookback of: "did top-M scored models from this ticker beat baseline return at t?"
   - Create a multiplicative gate:
     ```python
     # For each period t, check if t-1's top-M scored models had positive returns at t
     # This requires access to the returns matrix R
     hit_history = []
     M = cfg.top_m_for_ticker_gate
     for t in range(1, T):
         # Get the M models with highest scores at t-1 (pre-shift)
         scores_prev = SCORE[:, t-1]
         if np.sum(np.isfinite(scores_prev)) < M:
             hit_history.append(np.nan)
             continue
         top_m_idx = np.argsort(scores_prev)[::-1][:M]
         # Check their returns at t
         returns_t = R[top_m_idx, t]
         baseline_t = np.nanmedian(R[:, t])
         hit = np.mean(returns_t > baseline_t) if np.any(np.isfinite(returns_t)) else 0.5
         hit_history.append(hit)
     
     # Rolling hit rate
     hit_series = pd.Series([np.nan] + hit_history)  # align with T periods
     selection_hit_rate = hit_series.rolling(window=cfg.selection_quality_lookback, min_periods=2).mean()
     
     # Multiplicative gate: boost when hit rate > threshold, penalize when below
     excess = selection_hit_rate - cfg.selection_quality_threshold
     gate = 1.0 + cfg.selection_quality_weight * excess.clip(-0.5, 0.5)
     gate = gate.where(gate.isfinite(), 1.0)
     
     # Apply to SCORE
     SCORE = SCORE * gate.to_numpy()[None, :]
     ```

3. **Implementation specifics:**
   - This must be computed AFTER the main SCORE matrix but BEFORE the causal shift, so it uses scores that would have been used for selection at t-1
   - The gate is per-period (shape T), broadcast across all models in that ticker
   - The signal is ticker-level, not model-level, which simplifies computation and captures ticker predictability
   - When a ticker's recent selections have failed (hit rate < 50%), all models in that ticker get dampened; when successful, they get boosted
   - The clip to [-0.5, 0.5] prevents extreme gates from outlier hit rates

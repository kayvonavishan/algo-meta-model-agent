IDEA: Add a "Cross-Sectional Momentum Decay" signal that penalizes models whose recent momentum (improvement in rank) is decelerating relative to peers, favoring models with sustained or accelerating rank improvement

RATIONALE: The current delta signal captures the one-period change in rank (`D = Q_t - Q_{t-1}`), but doesn't distinguish between models that are *accelerating* (ranks improving faster) versus *decelerating* (ranks improving slower or starting to decline). The breakout signal helps identify models crossing into top-tier, but many false positives likely come from models whose breakout is already "stalling out" — they crossed a threshold but their rank momentum is decelerating (second derivative is negative). This is different from the rejected autocorrelation idea because we're not measuring consistency of direction, but rather the *change in velocity* of rank improvement. Specifically, we compute `acceleration = D_t - D_{t-1}` (second derivative of rank), normalize it, and penalize models with negative acceleration (decelerating momentum) while boosting models with positive acceleration (accelerating momentum). Models whose rank improvement is slowing down are more likely to mean-revert, contributing to the max underperformance streak of 7 periods. By incorporating this acceleration signal, we can filter out models at the "peak" of their rank momentum curve before they start declining. This is training-free, causal, and captures a signal dimension not addressed by existing features. The approach is analogous to momentum crash risk in equity factor research — momentum strategies often underperform when buying stocks whose momentum is peaking rather than still building.

REQUIRED_CHANGES: 1. **In `config.py`**, add new parameters:
   - `momentum_acceleration_weight: float = 0.08` — weight for the momentum acceleration/deceleration signal (similar magnitude to breakout_weight)
   - `momentum_acceleration_lookback: int = 3` — lookback for smoothing the acceleration signal (to reduce noise)

2. **In `scoring.py`, within `compute_scores_for_ticker_v2` (after computing D = rank delta):**
   - Compute the "second derivative" of ranks — the change in momentum:
     ```python
     # D[:, 1:] = Q[:, 1:] - Q[:, :-1] is already computed (first derivative)
     # Compute acceleration (second derivative): change in D
     accel_raw = np.zeros_like(D, dtype=float)
     accel_raw[:, 2:] = D[:, 2:] - D[:, 1:-1]  # accel[t] = D[t] - D[t-1]
     ```
   - Optionally smooth acceleration over a short lookback to reduce period-to-period noise:
     ```python
     # Rolling mean of acceleration over last momentum_acceleration_lookback periods
     accel_df = pd.DataFrame(accel_raw.T)
     accel_smooth = accel_df.rolling(window=cfg.momentum_acceleration_lookback, min_periods=1).mean().to_numpy().T
     ```
   - Normalize via percentile ranking across models per period:
     ```python
     accel_norm = percentile_ranks_across_models_v2(accel_smooth, axis=0)
     ```
   - Models with positive acceleration (ranks improving faster) get higher normalized scores; models with negative acceleration (decelerating or reversing) get lower scores.

3. **Add to base_forecast computation (after existing feature additions, around line ~548 in scoring.py):**
   ```python
   if cfg.momentum_acceleration_weight != 0 and accel_norm is not None:
       base_forecast = base_forecast + cfg.momentum_acceleration_weight * accel_norm
   ```

4. **Implementation specifics:**
   - The signal is computed from the same Q (percentile ranks) matrix already available
   - Use `np.diff` or manual subtraction to compute second derivative
   - Handle NaN edge cases by filling with 0.5 (neutral) or np.nan and let the percentile ranking handle it
   - The causal shift already applied to scores ensures this uses only past data for selection at time t
   - A short smoothing window (2-3 periods) reduces noise while still capturing meaningful deceleration patterns

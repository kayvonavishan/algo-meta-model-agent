IDEA: Add a "Momentum Deceleration Warning" signal that detects models whose rank improvement is slowing down (second derivative of rank trajectory is negative), penalizing models that appear to be "topping out" before their scores fully reflect the deterioration

RATIONALE: The current model has a 15% `pct_baseline_positive_meta_negative` rate (selecting losers when baseline is positive), and a `rolling_hit_rate_min` of 16.7%. These metrics suggest the model is sometimes "late" to detect reversals—it keeps selecting models that have already peaked. The existing signals (momentum M, delta D, durability, breakout) all reward models that are currently in good standing or improving, but none specifically detect *decelerating* improvement, which is an early warning sign of imminent reversal. A model whose rank rose from 0.3→0.5→0.6→0.65 is slowing down (Δ=0.2, 0.1, 0.05) even though its current rank (0.65) and recent momentum are still positive. The second derivative of rank trajectory—measuring whether rank gains are accelerating or decelerating—captures this "topping out" pattern. Unlike the rejected "consistency-adjusted momentum" (which measured autocorrelation of rank changes and added complexity), this is a simple numeric calculation: compare recent rank change to prior rank change. Penalizing models where `D_t < D_{t-1}` (deceleration) should help the model exit positions earlier in regime transitions. This complements the existing durability and breakout signals, which reward sustained performance, by adding an early-exit mechanism for models whose momentum is fading.

REQUIRED_CHANGES: 1. **In `config.py`**, add new parameters:
   - `momentum_decel_penalty_weight: float = 0.06` — penalty weight for momentum deceleration (similar magnitude to other feature weights)
   - `momentum_decel_lookback: int = 3` — how many periods of delta history to compare (minimum 2)
   - `momentum_decel_threshold: float = 0.05` — minimum deceleration magnitude to trigger penalty (filters noise)

2. **In `scoring.py`, within `compute_scores_for_ticker_v2` (after computing delta D but before base_forecast assembly):**
   - Compute the second derivative of rank trajectory (change in delta):
     ```python
     # D is already computed: D[:, t] = Q[:, t] - Q[:, t-1]
     # Compute delta-of-delta (acceleration/deceleration)
     DD = np.zeros_like(D, dtype=float)
     DD[:, 2:] = D[:, 2:] - D[:, 1:-1]  # DD_t = D_t - D_{t-1}
     ```
   - Identify decelerating models: `DD < -cfg.momentum_decel_threshold` (rank gains slowing significantly)
   - Compute a rolling deceleration score:
     ```python
     L = cfg.momentum_decel_lookback
     # For each t, count how many of the last L periods had significant deceleration
     decel_count = np.zeros((n_models, T), dtype=float)
     for t in range(L, T):
         decel_window = DD[:, t-L+1:t+1] < -cfg.momentum_decel_threshold
         decel_count[:, t] = np.sum(decel_window, axis=1) / L
     ```
   - Normalize via percentile ranking: `decel_penalty_norm = percentile_ranks_across_models_v2(decel_count, axis=0)`
   - This gives high values (approaching 1.0) to models with frequent recent deceleration

3. **Add penalty to the SCORE computation (after computing SCORE but before score_spread gating):**
   ```python
   if cfg.momentum_decel_penalty_weight != 0 and decel_penalty_norm is not None:
       SCORE = SCORE - cfg.momentum_decel_penalty_weight * decel_penalty_norm
   ```
   - Note: This is a **penalty** (subtraction), not a bonus, because we want to reduce scores for decelerating models

4. **Implementation specifics:**
   - The signal uses only D (already computed) and applies simple differencing, so computational overhead is minimal
   - The threshold parameter filters out normal noise in rank changes—only significant deceleration triggers the penalty
   - The penalty is normalized across models per period, so it's relative (penalizes the *most decelerating* models that period)
   - Handle NaN edge cases by defaulting to 0 penalty (conservative)
   - The signal is causal: uses delta computed from t-2, t-1 data to inform t's score

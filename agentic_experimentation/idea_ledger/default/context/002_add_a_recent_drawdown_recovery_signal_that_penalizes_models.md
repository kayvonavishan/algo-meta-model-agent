IDEA: Add a "Recent Drawdown Recovery" signal that penalizes models currently in or just emerging from a significant rank drawdown, favoring models that have maintained stable high ranks or fully recovered and consolidated**

RATIONALE: The current scoring rewards momentum and recent high ranks, but doesn't distinguish between models that are genuinely strong versus models that are "bouncing back" from a recent rank collapse. When a model's rank drops significantly (e.g., from 90th to 40th percentile) and then rebounds, the momentum signal treats this rebound positively. However, such models often exhibit mean reversion—the recovery is temporary, and they may fall again. This creates "trap" selections that contribute to the extended max drawdown duration (22 periods in current metrics). By computing the worst rank drawdown (peak-to-trough in Q) over a recent lookback and penalizing models with large recent drawdowns, we can avoid selecting models that are in "volatile recovery" mode. This differs from the confidence signal (which measures rank stability/std) because confidence penalizes all volatility equally, whereas this specifically targets asymmetric downside rank events. It also differs from the CVaR risk penalty (which operates on residuals) because this focuses on the drawdown structure specifically. Models with clean rank histories (no recent deep drops) get higher scores, while recently-crashed models get penalized even if they've recovered in the most recent period.

REQUIRED_CHANGES: 1. **In `config.py`**, add new parameters:
   - `rank_drawdown_penalty_weight: float = 0.10` — weight for the drawdown recovery penalty (higher than typical feature weights since it's a penalty)
   - `rank_drawdown_lookback: int = 6` — lookback window for measuring rank drawdown (how many periods to look back for peak rank)
   - `rank_drawdown_recovery_threshold: float = 0.8` — fraction of drawdown that must be recovered for penalty to start fading (e.g., 0.8 means model must recover 80% of rank loss)

2. **In `scoring.py`, within `compute_scores_for_ticker_v2` (after computing Q percentile ranks):**
   - For each model and time t, compute:
     - `peak_rank[t] = max(Q[:, max(0, t-L):t+1])` — highest rank in recent lookback (L = rank_drawdown_lookback)
     - `drawdown[t] = peak_rank[t] - Q[:, t]` — how far current rank is below recent peak (0 to ~1 scale)
   - Create `drawdown_penalty_raw`:
     - `drawdown_penalty_raw[:, t] = drawdown[:, t]` — raw drawdown magnitude
   - Normalize via percentile ranking (inverted, so high drawdown = low score):
     - `drawdown_norm = 1.0 - percentile_ranks_across_models_v2(drawdown_penalty_raw, axis=0)`
   - This produces a [0,1] signal where models with NO recent drawdown get ~1.0 and models with severe recent drawdowns get ~0.0

3. **Add to base_forecast computation (after existing feature additions, around line ~548):**
   ```python
   if cfg.rank_drawdown_penalty_weight != 0 and drawdown_norm is not None:
       base_forecast = base_forecast + cfg.rank_drawdown_penalty_weight * drawdown_norm
   ```

4. **Implementation specifics:**
   - Vectorized approach: use `pd.DataFrame(Q.T).rolling(window=L, min_periods=1).max()` to get peak ranks efficiently
   - The drawdown is simply `peak - current`, which is always >= 0
   - Handle NaN by treating as neutral (0.5) after percentile ranking
   - This signal is causal: at time t, we only use Q values through t
   - The signal naturally decays as the model recovers and enough time passes (the old peak falls out of the lookback window)

IDEA: Add a "Rank Relative Strength" signal that measures whether a model's current rank is above or below its own rolling median rank (self-relative performance), normalizing against cross-sectional peers to identify models outperforming their own baseline

RATIONALE: The current scoring uses cross-sectional percentile ranks (Q) to compare models against each other, but doesn't capture whether a model is performing better or worse *relative to its own historical norm*. A model currently at the 60th percentile might seem mediocre, but if its rolling median rank was 40th percentile, it's actually in an upswing. Conversely, a model at 70th percentile that usually ranks at 85th is underperforming its own baseline. This "self-relative" signal captures mean-reversion versus trend-continuation at the individual model level. The breakout signal captures crossing absolute thresholds; momentum captures trend direction; but neither captures deviation from self-baseline. Models performing above their own norm are likely in a favorable regime for that specific strategy, while models underperforming their norm may be entering a degradation phase. This is particularly useful for filtering out "always mediocre" models that happen to spike momentarily versus "normally excellent" models having a brief dip. The signal is training-free, causal, and orthogonal to existing features. It addresses the OOS degradation by avoiding models that are lucky outliers versus genuinely improving.

REQUIRED_CHANGES: 1. **In `config.py`**, add new parameters:
   - `rank_relative_strength_weight: float = 0.08` — weight for the self-relative rank signal
   - `rank_relative_strength_lookback: int = 8` — lookback window for computing rolling median rank per model

2. **In `scoring.py`, within `compute_scores_for_ticker_v2` (after computing Q percentile ranks):**
   - Compute rolling median of each model's rank over the lookback window:
     ```python
     Q_df = pd.DataFrame(Q.T)  # columns = models, rows = time
     rolling_median_rank = Q_df.rolling(window=cfg.rank_relative_strength_lookback, min_periods=2).median().to_numpy().T
     ```
   - Compute deviation from self-baseline: `rank_deviation = Q - rolling_median_rank` (positive = above own norm, negative = below)
   - Normalize via percentile ranking across models per period: `rank_rel_strength_norm = percentile_ranks_across_models_v2(rank_deviation, axis=0)`
   - This produces a [0,1] signal where models performing above their own historical baseline get higher scores

3. **Add to base_forecast computation (after existing feature additions, around line ~548):**
   ```python
   if cfg.rank_relative_strength_weight != 0 and rank_rel_strength_norm is not None:
       base_forecast = base_forecast + cfg.rank_relative_strength_weight * rank_rel_strength_norm
   ```

4. **Implementation specifics:**
   - The signal is computed per model (row) as deviation from its own rolling median rank
   - Use `pd.DataFrame(Q.T).rolling(window=L, min_periods=2).median()` for efficiency
   - Handle NaN by letting percentile ranking assign neutral scores
   - This differs from momentum (which measures trend direction across all models) because it's self-referential
   - The causal shift already applied to scores ensures this uses only past data for selection at time t

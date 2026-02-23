IDEA: Add a "Rank Quantile Transition Probability" signal that rewards models whose recent rank transitions predominantly move toward higher quantiles (e.g., Q→higher Q) rather than lower quantiles, using a quantile-binned transition analysis over a short lookback

Add a "Quantile Transition Probability" signal that rewards models whose recent rank transitions predominantly move toward higher quantile bins (e.g., from bottom-half to top-half) rather than lower bins, capturing systematic upward drift in rank distribution

RATIONALE: The current signals track rank velocity (delta), velocity confirmation (consistency of direction), and durability (tenure above threshold), but none measure the *distribution of rank transitions across quantile bins*. A model might have net positive delta but achieve it via volatile swings (e.g., going from 0.3→0.9→0.4→0.8 has net positive delta but shows unstable quantile positioning). By binning ranks into quantiles (e.g., 4 bins: 0-25%, 25-50%, 50-75%, 75-100%) and tracking the fraction of recent transitions that moved to a higher bin vs. lower bin, we can identify models with systematic upward drift in the rank distribution. This is distinct from:
- Delta (single-period change magnitude, not direction persistence across quantiles)
- Velocity confirmation (direction consistency, but doesn't account for magnitude of bin jumps)
- Durability (threshold tenure, not transition patterns)

Models that show strong "upward quantile transition probability" are systematically improving their relative standing, not just having lucky one-period spikes. This should help with the 15% baseline-positive-meta-negative problem by filtering out models that oscillate across quantiles rather than consistently climbing.

The current model tracks rank velocity (delta), velocity confirmation (directional consistency), and durability (threshold tenure), but none explicitly measure the *pattern of rank transitions across coarse quantile bins*. A model might show positive average delta but achieve it through volatile swings across the rank spectrum (e.g., 0.3→0.9→0.4→0.8), which indicates instability rather than genuine improvement. By discretizing ranks into quantile bins (e.g., quartiles) and measuring the fraction of recent transitions that moved upward vs. downward across bins, we capture "systematic rank improvement" distinct from noise. This directly addresses the 15% baseline-positive-meta-negative problem: models selected based on momentum/delta might be oscillators that happened to be on an upswing, while models with consistently upward quantile transitions are more likely to continue performing well. The signal follows the successful pattern of simple threshold/counting logic (like durability and breakout).

REQUIRED_CHANGES: 1. **In `config.py`**, add new parameters:
   - `quantile_transition_weight: float = 0.07` — weight for the quantile transition signal
   - `quantile_transition_lookback: int = 5` — lookback window for counting transitions
   - `quantile_transition_n_bins: int = 4` — number of quantile bins (default 4 = quartiles)

2. **In `scoring.py`, within `compute_scores_for_ticker_v2` (after computing Q percentile ranks):**
   - Bin each Q value into one of `n_bins` quantile bins (0, 1, 2, ..., n_bins-1)
   - For each model at time t, look at the last `lookback` periods and count:
     - `n_up = count of transitions where bin[t_k] > bin[t_{k-1}]` (moved to higher quantile)
     - `n_down = count of transitions where bin[t_k] < bin[t_{k-1}]` (moved to lower quantile)
     - `n_same = count of transitions where bin[t_k] == bin[t_{k-1}]` (stayed in same quantile)
   - Compute `upward_transition_rate = n_up / (n_up + n_down + eps)` (fraction of non-static transitions that were upward)
   - Normalize via percentile ranking: `quantile_transition_norm = percentile_ranks_across_models_v2(upward_transition_rate, axis=0)`

3. **Add to base_forecast computation:**
   ```python
   if cfg.quantile_transition_weight != 0 and quantile_transition_norm is not None:
       base_forecast = base_forecast + cfg.quantile_transition_weight * quantile_transition_norm
   ```

4. **Implementation specifics:**
   - Bin computation: `Q_binned = np.floor(Q * cfg.quantile_transition_n_bins).clip(0, cfg.quantile_transition_n_bins - 1).astype(int)`
   - Transition direction: `bin_diff = Q_binned[:, 1:] - Q_binned[:, :-1]` → classify as up (>0), down (<0), or same (0)
   - Use rolling window sum over the lookback to count up/down transitions efficiently
   - Handle NaN by treating as "static" (no transition counted)
   - Models with strong upward quantile drift get high normalized scores; oscillators or declining models get low scores

---

1. **In `config.py`**, add new parameters:
   - `quantile_transition_weight: float = 0.07` — weight for the quantile transition signal
   - `quantile_transition_lookback: int = 5` — lookback window for counting transitions
   - `quantile_transition_n_bins: int = 4` — number of quantile bins (quartiles by default)

2. **In `scoring.py`, within `compute_scores_for_ticker_v2` (after computing Q percentile ranks):**
   - Bin Q values: `Q_binned = np.floor(Q * n_bins).clip(0, n_bins - 1).astype(int)` to get discrete bin indices (0 to 3 for quartiles)
   - For each model at time t, over the lookback window:
     - Count transitions where `Q_binned[t_k] > Q_binned[t_{k-1}]` (upward)
     - Count transitions where `Q_binned[t_k] < Q_binned[t_{k-1}]` (downward)
   - Compute `upward_transition_rate = n_up / (n_up + n_down + eps)` (ignoring same-bin transitions)
   - Normalize via percentile ranking: `quantile_transition_norm = percentile_ranks_across_models_v2(upward_transition_rate, axis=0)`

3. **Add to base_forecast computation (after existing feature additions):**
   ```python
   if cfg.quantile_transition_weight != 0 and quantile_transition_norm is not None:
       base_forecast = base_forecast + cfg.quantile_transition_weight * quantile_transition_norm
   ```

4. **Implementation specifics:**
   - Compute bin differences: `bin_diff = Q_binned[:, 1:] - Q_binned[:, :-1]`
   - Rolling window count over lookback: `n_up = rolling_sum(bin_diff > 0)`, `n_down = rolling_sum(bin_diff < 0)`
   - Handle NaN in Q by propagating to binned values, then exclude from counts
   - Default to 0.5 (neutral) when insufficient transitions exist in the window
   - This is causal: uses data through t-1 for scoring at t (via final causal shift)

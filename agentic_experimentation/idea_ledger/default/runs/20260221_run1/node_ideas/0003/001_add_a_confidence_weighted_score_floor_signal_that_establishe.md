IDEA: Add a "Confidence-Weighted Score Floor" signal that establishes a minimum score threshold tied to a model's historical confidence level—models with high past confidence but currently low scores are penalized more aggressively, filtering out formerly-reliable models that have begun deteriorating before the standard signals detect the regime shift.

RATIONALE: The current system has a concerning pattern: `pct_baseline_positive_meta_negative` is 15.2%, meaning we select losers 15% of the time when the baseline actually has positive returns. This happens because momentum and durability signals have lagged responses—a model that was consistently good (high confidence) can take several periods to show up as deteriorating in the standard signals. The current confidence signal (CONF) measures stability of past ranks, but it's always additive/positive—high CONF always helps the score. This ignores the possibility that a "confident" model (stable historical ranks) that is now showing *declining* scores may be entering a regime change. By comparing a model's current score rank against its historical confidence rank, we can identify models that are "disappointing relative to their track record." If a model had top-quartile confidence but now has bottom-quartile score, it's underperforming its own established reliability—a red flag. This creates an asymmetric penalty: high-confidence models that slip get penalized more than low-confidence models, because the former represent unexpected deterioration. This directly addresses the 15% "wrong-when-baseline-is-right" problem without adding complexity to the risk penalty (which failed when made regime-adaptive). It's training-free, causal, and captures a distinct signal: relative performance disappointment given established reliability.

REQUIRED_CHANGES: 1. **In `config.py`**, add new parameters:
   - `conf_floor_penalty_weight: float = 0.12` — weight for the confidence-floor penalty (higher because it's a penalty, not a bonus)
   - `conf_floor_lookback: int = 6` — how many periods of score/confidence history to use for comparison
   - `conf_floor_threshold: float = 0.25` — percentile threshold below which a score triggers the penalty check

2. **In `scoring.py`, within `compute_scores_for_ticker_v2` (after computing SCORE and CONF, before score_spread gating):**
   - Compute rolling percentile rank of SCORE for each model: `score_rank_t = percentile_ranks_across_models_v2(SCORE, axis=0)`
   - Compute rolling percentile rank of CONF from the lookback window (this represents the model's "expected reliability rank")
   - For each model at time t:
     - `expected_reliability = rolling_mean(CONF[:, t-L:t])` over lookback periods, then percentile rank
     - `current_score_rank = score_rank_t[:, t]`
     - `disappointment = max(0, expected_reliability - current_score_rank - 0.25)` — how much the model is underperforming its reliability track record (with 0.25 tolerance)
   - Create penalty: `floor_penalty = cfg.conf_floor_penalty_weight * disappointment`
   - Apply: `SCORE[:, t] = SCORE[:, t] - floor_penalty[:, t]`

3. **Implementation specifics:**
   - The penalty only applies when `current_score_rank < cfg.conf_floor_threshold` AND `expected_reliability > 0.5` (was historically above-average confidence)
   - This creates an asymmetric filter: high-confidence models that suddenly underperform get penalized, but low-confidence models aren't penalized further for continuing to be low-confidence
   - Handle NaN by defaulting to no penalty (conservative)
   - The penalty is multiplicative with the gap between expected reliability and current performance, creating proportional downweighting
